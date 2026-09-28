"""Connectome-constrained rate model of the fly's fighting circuit.

Wiring (who connects to whom, how many synapses, excitatory or inhibitory) is
fixed by the connectome. Training only tunes what the connectome does not
tell us:
    - a gain and bias per cell type (shared by every neuron of that type),
    - how strongly each sensory channel drives its sensory neurons,
    - a linear readout from descending neurons to the motor command vector.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from flymt import motor
from flymt.fight_env import OBS_FIELDS

TAU = 0.02  # s, membrane-ish time constant
DT = 0.005  # s, brain integration step (2 per 10 ms env tick)
SUBSTEPS = 2

_F = {k: i for i, k in enumerate(OBS_FIELDS)}


def _relu(x):
    return np.maximum(x, 0.0)


def _prox(o) -> float:
    """0 when the opponent is across the ring, 1 when touching."""
    return float(np.clip(1.2 - 2.0 * o[_F["opp_dist"]], 0, 1))


# Sensory channels: each turns observations into a drive for one group of
# sensory neurons (the group is chosen in connectome.py by cell type).
CHANNELS = {
    # Visual projection neurons that track a nearby fly (LC10-like), per eye.
    "vis_target_left": lambda o: _relu(o[_F["opp_bearing_sin"]]) * _prox(o),
    "vis_target_right": lambda o: _relu(-o[_F["opp_bearing_sin"]]) * _prox(o),
    "vis_target_front": lambda o: _relu(o[_F["opp_bearing_cos"]]) * _prox(o),
    # Looming detectors (LPLC2 / LC4-like): something is coming at us fast.
    "looming": lambda o: min(1.0, (np.clip(o[_F["opp_closing_speed"]] * 2, 0, 1)
                                   + max(o[_F[f"opp_{s}"]] for s in motor.STRIKES))
                             * _prox(o)),
    # Mechanosensory bristles: being hit.
    "touch_head": lambda o: o[_F["touch_head"]],
    "touch_body": lambda o: max(o[_F["touch_thorax"]], o[_F["touch_abdomen"]]),
    "touch_legs": lambda o: o[_F["touch_legs"]],
    # Gravity / balance (Johnston's organ-like): tilted or flipped.
    "balance": lambda o: float(np.clip(1 - o[_F["up"]], 0, 1)),
    # Male aggression drive (P1-like), tonic during a bout.
    "aggression": lambda o: 1.0,
}


@dataclass
class Circuit:
    """A connectome subgraph ready to simulate."""

    body_id: np.ndarray  # (N,) int64 neuPrint bodyId
    type: np.ndarray  # (N,) str cell type
    side: np.ndarray  # (N,) str "L" / "R" / ""
    role: np.ndarray  # (N,) str: a CHANNELS key, "dn", or "inter"
    soma_xyz: np.ndarray  # (N, 3) float, nm; NaN where unknown
    W: sp.csr_matrix  # (N, N) post x pre, signed synapse counts
    dataset: str = ""

    @property
    def n(self) -> int:
        return len(self.body_id)

    def save(self, path):
        np.savez_compressed(
            path, body_id=self.body_id, type=self.type, side=self.side, role=self.role,
            soma_xyz=self.soma_xyz, W_data=self.W.data, W_indices=self.W.indices,
            W_indptr=self.W.indptr, W_shape=self.W.shape, dataset=self.dataset)

    @classmethod
    def load(cls, path) -> "Circuit":
        f = np.load(path, allow_pickle=False)
        W = sp.csr_matrix((f["W_data"], f["W_indices"], f["W_indptr"]),
                          shape=tuple(f["W_shape"]))
        return cls(f["body_id"], f["type"], f["side"], f["role"], f["soma_xyz"], W,
                   str(f["dataset"]))


class Layout:
    """Where each trainable parameter lives in the flat vector ES optimizes."""

    def __init__(self, c: Circuit):
        self.types, self.type_idx = np.unique(c.type, return_inverse=True)
        self.channels = list(CHANNELS)
        dn = np.flatnonzero(c.role == "dn")
        # Read out from DN types with left and right kept apart (turning needs it).
        keys = [f"{c.type[i]}|{c.side[i]}" for i in dn]
        self.dn_keys, dn_group = np.unique(keys, return_inverse=True)
        pool = sp.csr_matrix((np.ones(len(dn)), (dn_group, dn)),
                             shape=(len(self.dn_keys), c.n))
        counts = np.asarray(pool.sum(1)).ravel()
        self.dn_pool = (sp.diags(1 / counts) @ pool).tocsr()  # mean rate per group
        self.sensory = {ch: np.flatnonzero(c.role == ch) for ch in self.channels}
        self.n_dn, self.n_cmd = len(self.dn_keys), len(motor.COMMANDS)
        sizes = (("gain", len(self.types)), ("bias", len(self.types)),
                 ("input", len(self.channels)), ("w_scale", 1),
                 ("readout", self.n_cmd * self.n_dn), ("readout_b", self.n_cmd))
        self.slices, at = {}, 0
        for name, size in sizes:
            self.slices[name] = slice(at, at + size)
            at += size
        self.size = at

    def init(self, rng: np.random.Generator) -> np.ndarray:
        p = np.zeros(self.size)
        p[self.slices["input"]] = 1.0
        ro = rng.normal(0, 0.3, (self.n_cmd, self.n_dn))
        # Seed with what biology already tells us about a few DN types.
        cmd = {k: i for i, k in enumerate(motor.COMMANDS)}
        for j, key in enumerate(self.dn_keys):
            t, side = key.split("|")
            if t.startswith("DNa02"):  # steering: left DNa02 turns left
                ro[cmd["turn"], j] += 2.0 if side == "L" else -2.0
            if t.startswith(("DNp09", "oDN1")):  # forward walking
                ro[cmd["forward"], j] += 2.0
            if t.startswith("MDN"):  # moonwalker: backward walking
                ro[cmd["forward"], j] -= 2.0
        p[self.slices["readout"]] = ro.ravel()
        return p


class Brain:
    """Simulates the circuit and turns descending-neuron activity into commands."""

    def __init__(self, circuit: Circuit, layout: Layout, params: np.ndarray):
        self.c, self.L = circuit, layout
        s = layout.slices
        gain = np.exp(np.clip(params[s["gain"]], -3, 3))[layout.type_idx]
        self.bias = params[s["bias"]][layout.type_idx]
        # Normalize each neuron's total input so wiring scale is comparable
        # across neurons, then apply the per-type gain on the postsynaptic side.
        W = circuit.W.astype(np.float64)
        in_abs = np.asarray(abs(W).sum(1)).ravel()
        norm = 1.0 / np.sqrt(in_abs + 1.0)
        self.W = (sp.diags(gain * norm * np.exp(params[s["w_scale"]][0])) @ W).tocsr()
        self.input_gain = params[s["input"]]
        self.readout = params[s["readout"]].reshape(layout.n_cmd, layout.n_dn)
        self.readout_b = params[s["readout_b"]]
        self.reset()

    def reset(self):
        self.rates = np.zeros(self.c.n)
        self.dn = np.zeros(self.L.n_dn)

    def step(self, obs: np.ndarray) -> np.ndarray:
        drive = np.zeros(self.c.n)
        for k, ch in enumerate(self.L.channels):
            idx = self.L.sensory[ch]
            if len(idx):
                drive[idx] = self.input_gain[k] * CHANNELS[ch](obs)
        alpha = DT / TAU
        for _ in range(SUBSTEPS):
            x = self.W @ self.rates + self.bias + drive
            self.rates += alpha * (np.tanh(_relu(x)) - self.rates)
        self.dn = self.L.dn_pool @ self.rates
        z = self.readout @ self.dn + self.readout_b
        cmd = np.empty(self.L.n_cmd)
        cmd[:2] = np.tanh(z[:2])  # forward, turn
        cmd[2:] = 1 / (1 + np.exp(-z[2:]))  # strike / hold drive
        return cmd

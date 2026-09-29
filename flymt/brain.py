"""Connectome-constrained rate model of the fly's fighting circuit (PyTorch).

Wiring (who connects to whom, how many synapses, excitatory or inhibitory) is
fixed by the connectome. Training only tunes what the connectome does not
tell us:
    - a gain and bias per cell type (shared by every neuron of that type),
    - how strongly each sensory channel drives its sensory neurons,
    - a linear readout from descending neurons to the motor command vector.

The brain is the PPO policy: descending-neuron activity parameterizes a
distribution over commands (Gaussian walk/turn, Bernoulli strikes and holds).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from flymt import motor
from flymt.fight_env import OBS_FIELDS

TAU = 0.02  # s, membrane-ish time constant
DT = 0.005  # s, brain integration step
SUBSTEPS = 2  # per 10 ms env tick

_F = {k: i for i, k in enumerate(OBS_FIELDS)}
CHANNELS = ("vis_target_left", "vis_target_right", "looming", "touch_head",
            "touch_body", "touch_legs", "balance", "aggression", "dopamine")
DN_REST_BIAS = -0.5
# Initial recurrent weight scale (log). At full strength the circuit saturates
# (right DNa02 pinned on regardless of vision); at half strength DNa02 and other
# DNs respond to where the opponent is.
W_SCALE_INIT = float(np.log(0.5))
N_CONT = 2  # forward, turn
# Strikes and postures are each ONE choice per tick, with "none" as the default,
# so doing nothing is a single decision rather than five coin flips.
STRIKE_CHOICES = ("none",) + motor.STRIKES
POSTURE_CHOICES = ("none", "guard", "box", "clinch")
N_OUT = N_CONT + len(STRIKE_CHOICES) + len(POSTURE_CHOICES)
ACTION_DIM = N_CONT + 2  # [forward, turn, strike index, posture index]
_CMD = {k: i for i, k in enumerate(motor.COMMANDS)}


def sensory_drive(obs: torch.Tensor) -> torch.Tensor:
    """(B, obs) -> (B, channels): what each sensory neuron group feels."""
    o = lambda k: obs[:, _F[k]]  # noqa: E731
    prox = (1.2 - 2.0 * o("opp_dist")).clamp(0, 1)  # 1 = touching, 0 = far
    size = (0.1 / o("opp_dist").clamp(min=1e-3)).clamp(max=1)  # apparent size of the rival
    incoming = torch.stack([o(f"opp_{s}") for s in motor.STRIKES], 1).amax(1)
    return torch.stack([
        # LC10 visual projection neurons track another fly. Each eye covers its
        # own side and they overlap in front; drive scales with apparent size.
        (0.5 + o("opp_bearing_sin")).clamp(0, 1) * size,
        (0.5 - o("opp_bearing_sin")).clamp(0, 1) * size,
        # Looming detectors (LPLC2 / LC4-like): something is coming at us fast.
        (((o("opp_closing_speed") * 2).clamp(0, 1) + incoming) * prox).clamp(max=1),
        # Mechanosensory bristles: being hit.
        o("touch_head"),
        torch.maximum(o("touch_thorax"), o("touch_abdomen")),
        o("touch_legs"),
        # Gravity / balance (Johnston's organ-like): tilted or flipped.
        (1 - o("up")).clamp(0, 1),
        # Male aggression drive (pC1 / P1-class), tonic during a bout.
        torch.ones_like(prox),
        # Reward: PAM dopamine neurons fire when this fly scores.
        o("reward_signal").clamp(0, 1),
    ], 1)


@dataclass
class Circuit:
    """A connectome subgraph ready to simulate."""

    body_id: np.ndarray  # (N,) int64 neuPrint bodyId
    type: np.ndarray  # (N,) str cell type
    side: np.ndarray  # (N,) str "L" / "R" / ""
    role: np.ndarray  # (N,) str: a CHANNELS name, "dn", or "inter"
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


def _to_torch_sparse(m: sp.spmatrix) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)),
                                   m.shape).coalesce()


class TorchBrain(nn.Module):
    def __init__(self, circuit: Circuit):
        super().__init__()
        self.c = circuit
        types, type_idx = np.unique(circuit.type, return_inverse=True)
        self.types = types
        self.register_buffer("type_idx", torch.from_numpy(type_idx.astype(np.int64)))
        # Normalize each neuron's total input so wiring scale is comparable.
        W = circuit.W.astype(np.float64)
        in_abs = np.asarray(abs(W).sum(1)).ravel()
        self.register_buffer("W", _to_torch_sparse(sp.diags(1 / np.sqrt(in_abs + 1)) @ W))
        # Sensory neurons: which channel (if any) drives each neuron.
        S = np.zeros((len(CHANNELS), circuit.n), np.float32)
        for k, ch in enumerate(CHANNELS):
            S[k, circuit.role == ch] = 1.0
        self.register_buffer("S", torch.from_numpy(S))
        # Descending neurons pooled by type and side (turning needs L vs R).
        dn = np.flatnonzero(circuit.role == "dn")
        keys = np.array([f"{circuit.type[i]}|{circuit.side[i]}" for i in dn])
        self.dn_keys, group = np.unique(keys, return_inverse=True)
        counts = np.bincount(group)
        pool = sp.csr_matrix((1 / counts[group], (group, dn)),
                             shape=(len(self.dn_keys), circuit.n))
        self.register_buffer("P", _to_torch_sparse(pool))

        n_t, n_dn = len(types), len(self.dn_keys)
        self.log_gain = nn.Parameter(torch.zeros(n_t))
        self.bias = nn.Parameter(torch.zeros(n_t))
        self.input_gain = nn.Parameter(torch.ones(len(CHANNELS)))
        self.w_scale = nn.Parameter(torch.full((), W_SCALE_INIT))
        self.readout = nn.Linear(n_dn, N_OUT)
        self.log_std = nn.Parameter(torch.full((N_CONT,), -1.2))  # std 0.3
        self._seed_readout()
        with torch.no_grad():
            # Descending neurons are quiet at rest in real flies; start them there.
            is_dn = np.zeros(n_t, bool)
            is_dn[np.unique(type_idx[circuit.role == "dn"])] = True
            self.bias[torch.from_numpy(is_dn)] = DN_REST_BIAS
        self._calibrate_steering()

    @torch.no_grad()
    def _calibrate_steering(self):
        """Offset the turn command so an opponent dead ahead means 'no turn'.

        At rest the right DNa02 is more active than the left, which alone would
        make the fly circle right. This only sets the starting point; training
        is free to change it.
        """
        obs = torch.zeros(1, len(OBS_FIELDS))
        obs[0, _F["opp_dist"]] = 0.25
        obs[0, _F["opp_bearing_cos"]] = 1.0
        obs[0, _F["up"]] = 1.0
        h = self.init_hidden(1)
        for _ in range(40):
            out, h = self(obs, h)
        self.readout.bias[1] -= torch.atanh(out[0][0, 1].clamp(-0.999, 0.999))

    @torch.no_grad()
    def _seed_readout(self):
        """Start from what biology already tells us about a few DN types."""
        # Scale by fan-in so ~1000 DN groups don't sum to saturated commands.
        nn.init.normal_(self.readout.weight, 0, 0.3 / np.sqrt(self.readout.in_features))
        nn.init.zeros_(self.readout.bias)
        s0, p0 = N_CONT, N_CONT + len(STRIKE_CHOICES)
        self.readout.bias[s0] = 4.5  # "no strike" is the default (~5% strike per tick)
        self.readout.bias[p0] = 2.0  # "no posture" is the default
        for j, key in enumerate(self.dn_keys):
            t, side = key.split("|")
            if t.startswith("DNa02"):  # steering: left DNa02 turns left
                self.readout.weight[1, j] += 2.0 if side == "L" else -2.0
            if t.startswith(("DNp09", "oDN1")):  # forward walking
                self.readout.weight[0, j] += 2.0

    def init_hidden(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.c.n, device=self.S.device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        """One 10 ms tick. obs (B, F), h (B, N) -> (mean, log_std, strike, posture), h."""
        drive = (sensory_drive(obs) * self.input_gain) @ self.S  # (B, N)
        gain = torch.exp(self.log_gain.clamp(-3, 3))[self.type_idx] * torch.exp(self.w_scale)
        bias = self.bias[self.type_idx]
        r = h.T  # (N, B) for sparse matmul
        for _ in range(SUBSTEPS):
            x = gain[:, None] * torch.sparse.mm(self.W, r) + bias[:, None] + drive.T
            r = r + (DT / TAU) * (torch.tanh(x.relu()) - r)
        h = r.T
        dn = torch.sparse.mm(self.P, r).T  # (B, n_dn)
        z = self.readout(dn)
        s0, p0 = N_CONT, N_CONT + len(STRIKE_CHOICES)
        return (torch.tanh(z[:, :N_CONT]), self.log_std.expand(len(z), -1),
                z[:, s0:p0], z[:, p0:]), h

    @staticmethod
    def distribution(out):
        mean, log_std, strike, posture = out
        return (torch.distributions.Normal(mean, log_std.exp()),
                torch.distributions.Categorical(logits=strike),
                torch.distributions.Categorical(logits=posture))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, h: torch.Tensor, deterministic: bool = False):
        """Choose an action. Returns (action (B, ACTION_DIM), log_prob (B,), h).

        The action is [forward, turn, strike index, posture index]; turn it into
        a motor command with `to_command`.
        """
        out, h = self(obs, h)
        cont, strike, posture = self.distribution(out)
        if deterministic:
            a_c, a_s, a_p = out[0], out[2].argmax(1), out[3].argmax(1)
        else:
            a_c, a_s, a_p = cont.sample(), strike.sample(), posture.sample()
        logp = cont.log_prob(a_c).sum(1) + strike.log_prob(a_s) + posture.log_prob(a_p)
        # Raw walk/turn sample (not clipped) so PPO scores exactly the action taken;
        # the motor system bounds walk/turn itself.
        return torch.cat([a_c, a_s[:, None].float(), a_p[:, None].float()], 1), logp, h

    def log_prob(self, out, action):
        cont, strike, posture = self.distribution(out)
        return (cont.log_prob(action[:, :N_CONT]).sum(1)
                + strike.log_prob(action[:, N_CONT].long())
                + posture.log_prob(action[:, N_CONT + 1].long()))

    def entropy(self, out):
        cont, strike, posture = self.distribution(out)
        return cont.entropy().sum(1) + strike.entropy() + posture.entropy()


def to_command(action: np.ndarray) -> np.ndarray:
    """[forward, turn, strike index, posture index] rows -> motor command rows."""
    action = np.atleast_2d(action)
    cmd = np.zeros((len(action), len(motor.COMMANDS)), np.float32)
    cmd[:, :N_CONT] = action[:, :N_CONT]
    for i, (s, p) in enumerate(action[:, N_CONT:].astype(int)):
        if s:
            cmd[i, _CMD[STRIKE_CHOICES[s]]] = 1.0
        if p:
            cmd[i, _CMD[POSTURE_CHOICES[p]]] = 1.0
    return cmd

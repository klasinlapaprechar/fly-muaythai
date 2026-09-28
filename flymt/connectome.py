"""Build the fighting circuit from the male Drosophila CNS connectome (neuPrint).

Every neuron is a real, individually reconstructed neuron and every connection
is a real synapse count from dataset male-cns:v1.0. Selection:
    1. sensory groups for each input channel (vision, looming, touch, balance,
       aggression drive, dopamine),
    2. all descending neurons (the brain's command lines to the body),
    3. interneurons that sit on real synaptic paths between the two,
    4. all synapses among the selected neurons, signed by neurotransmitter.

Usage:
    python -m flymt.connectome            # needs NEUPRINT_APPLICATION_CREDENTIALS in .env
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flymt.brain import CHANNELS, Circuit

DATASET = "male-cns:v1.0"
OUT = Path("data/connectome/circuit.npz")
MIN_SYNAPSES = 3  # ignore connections weaker than this (reconstruction noise)
SENSORY_CAP = 250  # per channel: keep the neurons with the most output synapses

SENSORY = {
    "vis_target_left": "n.type STARTS WITH 'LC10' AND n.somaSide = 'L'",
    "vis_target_right": "n.type STARTS WITH 'LC10' AND n.somaSide = 'R'",
    "looming": "n.type IN ['LPLC2', 'LC4']",
    "touch_head": "n.type IN ['BM_InOm', 'BM']",
    "touch_body": "n.class = 'mechanosensory_tactile' AND "
                  "n.entryNerve IN ['ADMN', 'PDMN', 'DMetaN', 'DProN']",
    "touch_legs": "n.class = 'mechanosensory_tactile' AND "
                  "n.entryNerve IN ['ProLN', 'MesoLN', 'MetaLN']",
    "balance": "n.type STARTS WITH 'JO-C' OR n.type STARTS WITH 'JO-E'",
    "aggression": "n.type STARTS WITH 'pC1'",
    "dopamine": "n.type STARTS WITH 'PAM'",
}
assert set(SENSORY) == set(CHANNELS)
DN_WHERE = "n.superclass = 'descending_neuron'"
# Interneuron budget: bridges (on sensory->DN paths), sensory processors, premotor.
N_BRIDGE, N_SENSORY_SIDE, N_PREMOTOR = 1500, 500, 700

NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
           # Monoamines act as slow neuromodulators, not fast synapses.
           "dopamine": 0.0, "serotonin": 0.0, "octopamine": 0.0}


def _client():
    from neuprint import Client
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("NEUPRINT_APPLICATION_CREDENTIALS="):
                token = line.split("=", 1)[1].strip()
    if not token:
        raise SystemExit("Set NEUPRINT_APPLICATION_CREDENTIALS (see README).")
    return Client("neuprint.janelia.org", dataset=DATASET, token=token)


def _ids(xs) -> str:
    return "[" + ",".join(str(int(x)) for x in xs) + "]"


def _chunks(xs, n):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def build_circuit(c, verbose: bool = True) -> Circuit:
    log = print if verbose else (lambda *a: None)
    role: dict[int, str] = {}
    for ch, where in SENSORY.items():
        df = c.fetch_custom(f"MATCH (n:Neuron) WHERE {where} RETURN n.bodyId AS id, "
                            f"n.pre AS pre ORDER BY pre DESC LIMIT {SENSORY_CAP}")
        for i in df.id:
            role.setdefault(int(i), ch)
        log(f"  sensory {ch:17s} {len(df):4d} neurons")
    sensory = list(role)
    dns = c.fetch_custom(f"MATCH (n:Neuron) WHERE {DN_WHERE} RETURN n.bodyId AS id").id
    for i in dns:
        role[int(i)] = "dn"
    log(f"  descending neurons      {len(dns):4d}")

    # Interneuron candidates: synaptic input from our sensory set, output to DNs.
    w_in = pd.concat([c.fetch_custom(
        f"MATCH (s:Neuron)-[w:ConnectsTo]->(x:Neuron) WHERE s.bodyId IN {_ids(ch)} "
        f"AND w.weight >= {MIN_SYNAPSES} RETURN x.bodyId AS id, sum(w.weight) AS w")
        for ch in _chunks(sensory, 500)]).groupby("id").w.sum()
    w_out = pd.concat([c.fetch_custom(
        f"MATCH (x:Neuron)-[w:ConnectsTo]->(d:Neuron) WHERE d.bodyId IN {_ids(ch)} "
        f"AND w.weight >= {MIN_SYNAPSES} RETURN x.bodyId AS id, sum(w.weight) AS w")
        for ch in _chunks(dns, 500)]).groupby("id").w.sum()
    cand = pd.DataFrame({"w_in": w_in, "w_out": w_out}).fillna(0)
    cand = cand[~cand.index.isin(list(role))]
    cand["bridge"] = np.sqrt(cand.w_in * cand.w_out)
    picked = set(cand.nlargest(N_BRIDGE, "bridge").index)
    picked |= set(cand.drop(list(picked)).nlargest(N_SENSORY_SIDE, "w_in").index)
    picked |= set(cand.drop(list(picked)).nlargest(N_PREMOTOR, "w_out").index)
    for i in picked:
        role[int(i)] = "inter"
    log(f"  interneurons            {len(picked):4d} "
        f"(from {len(cand)} candidates on sensory/DN paths)")

    ids = np.array(sorted(role), dtype=np.int64)
    props = pd.concat([c.fetch_custom(
        f"MATCH (n:Neuron) WHERE n.bodyId IN {_ids(ch)} RETURN n.bodyId AS id, "
        "n.type AS type, n.somaSide AS side, n.consensusNt AS nt, "
        "n.celltypePredictedNt AS nt_type, n.somaLocation AS soma, "
        "n.superclass AS superclass")
        for ch in _chunks(ids, 2000)]).set_index("id").loc[ids]

    edges = pd.concat([c.fetch_custom(
        f"MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN {_ids(ch)} "
        f"AND b.bodyId IN {_ids(ids)} AND w.weight >= {MIN_SYNAPSES} "
        "RETURN a.bodyId AS pre, b.bodyId AS post, w.weight AS w")
        for ch in _chunks(ids, 300)])
    log(f"  synaptic connections   {len(edges)} (>= {MIN_SYNAPSES} synapses each), "
        f"{int(edges.w.sum())} synapses total")

    nt = props.nt.where(props.nt.notna() & (props.nt != "unclear"), props.nt_type)
    sign = nt.map(NT_SIGN).fillna(0.0).to_numpy()  # unknown transmitter: no effect
    pos = {b: k for k, b in enumerate(ids)}
    pre = edges.pre.map(pos).to_numpy()
    post = edges.post.map(pos).to_numpy()
    W = sp.csr_matrix((edges.w.to_numpy() * sign[pre], (post, pre)),
                      shape=(len(ids), len(ids)))
    W.eliminate_zeros()

    def xyz(s):
        if isinstance(s, dict):
            s = s.get("coordinates")
        return s if isinstance(s, (list, tuple)) and len(s) == 3 else [np.nan] * 3

    types = props.type.fillna("untyped").astype(str).to_numpy()
    sides = props.side.fillna("").astype(str).to_numpy()
    roles = np.array([role[int(i)] for i in ids])
    soma = np.array([xyz(s) for s in props.soma], dtype=float)
    log("  neurotransmitters: " + ", ".join(f"{k} {v}" for k, v in nt.value_counts().items()))
    return Circuit(ids, types.astype("<U64"), sides.astype("<U4"), roles.astype("<U32"),
                   soma, W, DATASET)


ROLE_COLORS = {"vis_target_left": "#4fc3f7", "vis_target_right": "#0288d1",
               "looming": "#7e57c2", "touch_head": "#ffb74d", "touch_body": "#ffa726",
               "touch_legs": "#fb8c00", "balance": "#aed581", "aggression": "#e53935",
               "dopamine": "#ffee58", "inter": "#78909c", "dn": "#f06292"}


def display_positions(circ: Circuit, seed: int = 0) -> np.ndarray:
    """2D drawing positions (seen from behind: fly's left on the left).

    Sensory neurons with cell bodies outside the CNS (bristles, legs, antenna)
    have no soma location; draw them at the mean soma position of the neurons
    they synapse onto, with a little jitter.
    """
    xy = np.column_stack([-circ.soma_xyz[:, 0], -circ.soma_xyz[:, 1]])
    known = np.isfinite(xy).all(1)
    A = abs(circ.W).T.tocsr()  # pre x post
    rng = np.random.default_rng(seed)
    for i in np.flatnonzero(~known):
        row = A.getrow(i)
        tgt = row.indices[known[row.indices]]
        if len(tgt):
            w = np.asarray(row[:, tgt].todense()).ravel()
            xy[i] = (xy[tgt] * w[:, None]).sum(0) / w.sum()
    spread = np.nanstd(xy[known], 0) * 0.03
    xy[~known] += rng.normal(0, 1, (int((~known).sum()), 2)) * spread
    return xy


def plot_circuit(circ: Circuit, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = display_positions(circ)
    ok = np.isfinite(xy).all(1)
    fig, ax = plt.subplots(figsize=(9, 10), facecolor="#101418")
    ax.set_facecolor("#101418")
    for r in ["inter", *CHANNELS, "dn"]:
        m = ok & (circ.role == r)
        ax.scatter(xy[m, 0], xy[m, 1], s=4 if r == "inter" else 9,
                   c=ROLE_COLORS[r], alpha=0.8, linewidths=0,
                   label=f"{r} ({(circ.role == r).sum()})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8, labelcolor="w", markerscale=2.5, frameon=False)
    ax.set_title(f"Fighting circuit from {circ.dataset}: {circ.n} real neurons, "
                 f"{circ.W.nnz} connections\n(cell bodies seen from behind; peripheral sensory "
                 "neurons drawn at their synaptic targets)",
                 color="w", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plot-only", action="store_true", help="re-plot a saved circuit")
    args = ap.parse_args()
    if args.plot_only:
        plot_circuit(Circuit.load(args.out), Path("docs/img/circuit.png"))
        return
    print(f"Building fighting circuit from {DATASET} ...")
    circ = build_circuit(_client())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    circ.save(out)
    print(f"saved {out}: {circ.n} neurons, {circ.W.nnz} signed connections")
    plot_circuit(circ, Path("docs/img/circuit.png"))
    print("saved docs/img/circuit.png")


if __name__ == "__main__":
    main()

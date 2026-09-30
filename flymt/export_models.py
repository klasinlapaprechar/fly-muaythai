"""Package the trained fighters so the game runs right after download.

Training checkpoints are ~8 MB each (critic, optimizer state, fixed wiring).
The fighter itself is just the ~15K learned parameters; the wiring is rebuilt
from the circuit file. This writes models/ with the circuit and one slim file
per difficulty level.

Usage:
    python -m flymt.export_models
"""

import shutil
from pathlib import Path

import torch

from flymt.brain import Circuit, TorchBrain

LEVELS = {"easy": "checkpoints/update_0100.pt", "medium": "checkpoints/update_0500.pt",
          "hard": "checkpoints/latest.pt"}
OUT = Path("models")


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copy("data/connectome/circuit.npz", OUT / "circuit.npz")
    names = {n for n, _ in TorchBrain(Circuit.load(OUT / "circuit.npz")).named_parameters()}
    for level, path in LEVELS.items():
        ck = torch.load(path, map_location="cpu")
        params = {k: v.clone() for k, v in ck["brain"].items() if k in names}
        torch.save({"brain": params, "update": ck["update"], "level": level},
                   OUT / f"fighter_{level}.pt")
        n = sum(v.numel() for v in params.values())
        print(f"{level:6s} <- {path} (update {ck['update']}): {n} learned parameters")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()

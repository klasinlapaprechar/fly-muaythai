"""Smoke tests. The circuit here is random and only exercises code paths."""

import argparse

import numpy as np
import scipy.sparse as sp

from flymt import motor, train
from flymt.brain import CHANNELS, Brain, Circuit, Layout
from flymt.fight_env import FightEnv


def synthetic_circuit(n=300, seed=0) -> Circuit:
    rng = np.random.default_rng(seed)
    roles = np.array(["inter"] * n, dtype="<U32")
    for k, ch in enumerate(CHANNELS):
        roles[k * 5:(k + 1) * 5] = ch
    roles[-40:] = "dn"
    types = np.array([f"T{i % 25}" for i in range(n)])
    types[-40:] = [f"DNx{i % 10:02d}" for i in range(40)]
    sides = np.array(["L", "R"] * (n // 2))
    W = sp.random(n, n, density=0.05, random_state=seed, format="csr")
    W.data = np.round(W.data * 20) * rng.choice([-1, 1], W.nnz, p=[0.3, 0.7])
    return Circuit(np.arange(n), types, sides, roles, rng.normal(size=(n, 3)), W, "synthetic")


def test_env_bout_runs():
    env = FightEnv(seed=0)
    obs = env.reset()
    assert obs["red"].shape == (env.obs_size,)
    for _ in range(20):
        obs, rew, done, ev = env.step({n: np.zeros(env.cmd_size) for n in obs})
    assert np.isfinite(obs["red"]).all()


def test_brain_step_and_roundtrip(tmp_path):
    c = synthetic_circuit()
    c.save(tmp_path / "c.npz")
    c2 = Circuit.load(tmp_path / "c.npz")
    assert (c2.W != c.W).nnz == 0 and (c2.role == c.role).all()
    L = Layout(c2)
    brain = Brain(c2, L, L.init(np.random.default_rng(0)))
    obs = FightEnv(seed=0).reset()["red"]
    for _ in range(10):
        cmd = brain.step(obs)
    assert cmd.shape == (len(motor.COMMANDS),)
    assert np.all((brain.rates >= 0) & (brain.rates <= 1))


def test_es_generation(tmp_path):
    c = synthetic_circuit()
    c.save(tmp_path / "c.npz")
    args = argparse.Namespace(
        circuit=str(tmp_path / "c.npz"), out=str(tmp_path / "ck"), generations=2,
        pairs=2, bouts=1, sigma=0.05, lr=0.03, decay=0.005, workers=2, stage=0,
        save_every=1, seed=0, resume=False)
    train.EPISODE_SECONDS[0] = 0.3
    train.train(args)
    ck = np.load(tmp_path / "ck" / "latest.npz")
    assert int(ck["gen"]) == 1 and np.isfinite(ck["params"]).all()

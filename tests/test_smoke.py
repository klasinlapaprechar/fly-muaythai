"""Smoke tests. The circuit here is random and only exercises code paths."""

import argparse

import numpy as np
import scipy.sparse as sp

import torch

from flymt import motor, ppo
from flymt.brain import CHANNELS, Circuit, TorchBrain
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
    brain = TorchBrain(c2)
    obs = torch.from_numpy(FightEnv(seed=0).reset()["red"])[None]
    h = brain.init_hidden(1)
    for _ in range(10):
        cmd, logp, h = brain.act(obs, h)
    assert cmd.shape == (1, len(motor.COMMANDS)) and torch.isfinite(logp).all()
    assert ((h >= 0) & (h <= 1)).all()


def test_ppo_updates(tmp_path):
    c = synthetic_circuit()
    c.save(tmp_path / "c.npz")
    args = argparse.Namespace(
        circuit=str(tmp_path / "c.npz"), out=str(tmp_path / "ck"), updates=2, workers=2,
        envs_per_worker=1, rollout=16, chunk=8, minibatch=4, epochs=1, lr=3e-4,
        gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad=0.5,
        promote_window=4, stage=0, save_every=1, seed=0, cpu=False, resume=False)
    ppo.train(args)
    ck = torch.load(tmp_path / "ck" / "latest.pt")
    assert ck["update"] == 1 and ck["steps"] == 2 * 16 * 2

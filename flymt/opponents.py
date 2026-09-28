"""Scripted sparring partners for the training curriculum."""

import numpy as np

from flymt import motor
from flymt.fight_env import OBS_FIELDS

STAGES = ("bag", "mover", "sparring", "selfplay")
_F = {k: i for i, k in enumerate(OBS_FIELDS)}
_C = {k: i for i, k in enumerate(motor.COMMANDS)}


class Opponent:
    """Heavy bag: stands still."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(len(motor.COMMANDS))


class Mover(Opponent):
    """Wanders around the ring and turns off the ropes."""

    def __init__(self, rng):
        super().__init__(rng)
        self.cmd = np.zeros(len(motor.COMMANDS))

    def act(self, obs):
        if self.rng.random() < 0.02:  # new heading every ~0.5 s
            self.cmd[_C["forward"]] = self.rng.uniform(-0.3, 0.8)
            self.cmd[_C["turn"]] = self.rng.uniform(-0.8, 0.8)
        if obs[_F["rope_dist"]] < 0.15:
            self.cmd[_C["turn"]] = 1.0
        return self.cmd


class Sparring(Opponent):
    """Closes the distance, throws a random strike every ~0.25 s, sometimes guards."""

    def act(self, obs):
        c = np.zeros(len(motor.COMMANDS))
        bearing = np.arctan2(obs[_F["opp_bearing_sin"]], obs[_F["opp_bearing_cos"]])
        c[_C["turn"]] = np.clip(2 * bearing, -1, 1)
        if obs[_F["opp_dist"]] > 0.2:
            c[_C["forward"]] = 0.8 if abs(bearing) < 0.5 else 0.2
        elif self.rng.random() < 0.04:
            c[_C[self.rng.choice(motor.STRIKES)]] = 1.0
        if any(obs[_F[f"opp_{s}"]] for s in motor.STRIKES) and self.rng.random() < 0.3:
            c[_C["guard"]] = 1.0
        return c


def make(stage: int, rng: np.random.Generator) -> Opponent | None:
    """Scripted opponent for a stage, or None when the brain's past self fights."""
    return {0: Opponent, 1: Mover, 2: Sparring}.get(stage, lambda r: None)(rng)

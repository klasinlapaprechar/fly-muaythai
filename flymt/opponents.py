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
    """A sparring partner that makes you earn it.

    Circles at the edge of range, steps in to throw one or two strikes, then
    backs out. Raises its guard when it sees a strike coming.
    """

    def __init__(self, rng):
        super().__init__(rng)
        self.circle = rng.choice([-1.0, 1.0])  # which way it circles
        self.retreat = 0  # ticks left backing off after an attack
        self.guard = 0  # ticks left holding the guard up
        self.attack = 0  # ticks left in the current attack

    def act(self, obs):
        c = np.zeros(len(motor.COMMANDS))
        bearing = np.arctan2(obs[_F["opp_bearing_sin"]], obs[_F["opp_bearing_cos"]])
        dist = obs[_F["opp_dist"]]
        incoming = any(obs[_F[f"opp_{s}"]] for s in motor.STRIKES)
        if incoming and self.guard == 0 and self.rng.random() < 0.6:
            self.guard = 20
        if self.retreat > 0:  # back out of range, still facing the opponent
            self.retreat -= 1
            c[_C["forward"]] = -0.5
            c[_C["turn"]] = np.clip(2 * bearing, -1, 1)
        elif self.attack > 0:  # step in and throw
            self.attack -= 1
            c[_C["forward"]] = 0.8
            c[_C["turn"]] = np.clip(2 * bearing, -1, 1)
            if dist < 0.2 and self.rng.random() < 0.15:
                c[_C[self.rng.choice(motor.STRIKES)]] = 1.0
            if self.attack == 0:
                self.retreat = 30
        else:  # circle just outside range, waiting for an opening
            c[_C["turn"]] = np.clip(2 * bearing + 0.6 * self.circle, -1, 1)
            c[_C["forward"]] = 0.6 if dist > 0.35 else (-0.3 if dist < 0.25 else 0.3)
            if self.rng.random() < 0.03:
                self.circle = -self.circle
            if dist < 0.4 and self.rng.random() < 0.04:
                self.attack = 40
        if obs[_F["rope_dist"]] < 0.12:
            c[_C["forward"]] = max(c[_C["forward"]], 0.3)
        if self.guard > 0:
            self.guard -= 1
            c[_C["guard"]] = 1.0
        return c


def make(stage: int, rng: np.random.Generator) -> Opponent | None:
    """Scripted opponent for a stage, or None when the brain's past self fights."""
    return {0: Opponent, 1: Mover, 2: Sparring}.get(stage, lambda r: None)(rng)

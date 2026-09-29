"""Scripted sparring partners for the training curriculum."""

import numpy as np

from flymt import motor
from flymt.fight_env import OBS_FIELDS

# Stage 0 teaches steering against a wandering target that never strikes.
# Stage 4 mixes opponents: half the bouts against the Veteran, half against
# past versions of the fly's own brain.
STAGES = ("orient", "bag", "mover", "sparring", "mixed")
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


COMBOS = (("jab_l", "kick_r"), ("jab_r", "jab_l", "lunge"), ("jab_l", "jab_r"),
          ("kick_l", "jab_r"), ("jab_r", "kick_l", "kick_r"))


class Veteran(Opponent):
    """A scripted fighter that fights back like an experienced opponent.

    Always faces you; circles at the edge of range and steps in and out;
    guards against incoming strikes and counters right after them; punishes
    misses; throws combinations; uses the boxing stance and the clinch.
    """

    def __init__(self, rng):
        super().__init__(rng)
        self.circle = rng.choice([-1.0, 1.0])
        self.queue: list[str] = []  # strikes still to throw in the current combo
        self.fire = False  # strike command raised last tick (need a rising edge)
        self.opp_striking = False
        self.guard = self.box = self.clinch = self.retreat = 0

    def act(self, obs):
        c = np.zeros(len(motor.COMMANDS))
        bearing = np.arctan2(obs[_F["opp_bearing_sin"]], obs[_F["opp_bearing_cos"]])
        dist = obs[_F["opp_dist"]]
        c[_C["turn"]] = np.clip(3 * bearing, -1, 1)  # always square up
        incoming = any(obs[_F[f"opp_{s}"]] for s in motor.STRIKES)
        busy = any(obs[_F[f"self_{s}"]] for s in motor.STRIKES)
        in_range = dist < 0.22 and abs(bearing) < 0.6

        # Defense: guard the incoming strike; the moment it ends, counter.
        if incoming and not self.opp_striking and self.guard == 0 and self.rng.random() < 0.7:
            self.guard = 12
        if self.opp_striking and not incoming and in_range and not self.queue:
            self.queue = list(COMBOS[self.rng.integers(len(COMBOS))])
        self.opp_striking = incoming

        # Offense: open with a combo when in range and the moment is right.
        if not self.queue and in_range and self.retreat == 0 and self.rng.random() < 0.03:
            self.queue = list(COMBOS[self.rng.integers(len(COMBOS))])
        if self.fire:
            self.fire = False  # release so the next strike gets a rising edge
        elif self.queue and not busy and dist < 0.25:
            c[_C[self.queue.pop(0)]] = 1.0
            self.fire = True
            if not self.queue:
                self.retreat = 25  # step out after the combo

        # Footwork.
        if self.retreat > 0:
            self.retreat -= 1
            c[_C["forward"]] = -0.4
        elif self.queue:
            c[_C["forward"]] = 0.7 if dist > 0.18 else 0.0
        else:
            c[_C["turn"]] = np.clip(3 * bearing + 0.5 * self.circle, -1, 1)
            c[_C["forward"]] = 0.6 if dist > 0.32 else (-0.3 if dist < 0.22 else 0.2)
            if self.rng.random() < 0.02:
                self.circle = -self.circle
        if obs[_F["rope_dist"]] < 0.12:
            c[_C["forward"]] = max(c[_C["forward"]], 0.3)

        # Stances: rear up to box at range; clinch when chest to chest.
        if self.box == 0 and 0.2 < dist < 0.35 and self.rng.random() < 0.01:
            self.box = 50
        if self.clinch == 0 and dist < 0.14 and self.rng.random() < 0.02:
            self.clinch = 30
        for name in ("guard", "box", "clinch"):
            left = getattr(self, name)
            if left > 0:
                setattr(self, name, left - 1)
                c[_C[name]] = 1.0
                break
        return c


def make(stage: int, rng: np.random.Generator, slot: int = 0) -> Opponent | None:
    """Scripted opponent for a stage, or None when the brain's past self fights.

    In the mixed stage, even-numbered fight slots get the Veteran and
    odd-numbered slots get a past version of the brain.
    """
    if stage == 4:
        return Veteran(rng) if slot % 2 == 0 else None
    return {0: Mover, 1: Opponent, 2: Mover, 3: Sparring}[stage](rng)

"""Motor system: the fly's "ventral nerve cord".

The brain does not drive 59 joints directly. Like descending neurons in a real
fly, it sends a small command vector and this module expands it into joint
targets: a tripod walking gait plus a library of Muay Thai strikes.

Command vector (all floats):
    forward  [-1, 1]  walk speed/direction
    turn     [-1, 1]  + = turn left
    jab_l, jab_r      foreleg punch      (fires on rising edge > 0.5)
    kick_l, kick_r    mid-leg roundhouse (fires on rising edge > 0.5)
    lunge             rear up and slam forward, the fly's "knee"
    clinch            forelegs grab + claw adhesion while > 0.5
    guard             forelegs raised in front while > 0.5
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from flymt.arena import FighterIndex
from flymt.ik import IK_JOINTS, LEGS, LegIK

COMMANDS = ("forward", "turn", "jab_l", "jab_r", "kick_l", "kick_r",
            "lunge", "clinch", "guard")
STRIKES = ("jab_l", "jab_r", "kick_l", "kick_r", "lunge")

GROUND_Z = -0.124  # claw height at rest, thorax frame
STRIDE = 0.07  # cm, full-speed stride length
LIFT = 0.04  # cm, swing height
GAIT_HZ = 10.0
N_PHASE = 32
STRIDE_SCALES = np.linspace(-1, 1, 5)
TRIPOD_A = ("T1_left", "T2_right", "T3_left")
CACHE = Path(__file__).resolve().parent.parent / "cache" / "motor_tables.npz"


@dataclass(frozen=True)
class Move:
    """A strike: claw keyframes for the legs it uses, written for the left side."""

    legs: tuple[str, ...]  # segment names ("T1", "T2", "T3")
    keys: tuple[tuple[float, tuple[tuple[float, float, float], ...]], ...]
    duration: float  # seconds
    bilateral: bool = False  # both sides move together (lunge)


# Keyframes: (time fraction, claw target per leg) in the thorax frame. y is for
# the left leg and is mirrored for right-side moves.
MOVES = {
    "jab": Move(("T1",), (
        (0.0, ((0.09, 0.087, GROUND_Z),)),
        (0.25, ((0.11, 0.07, -0.03),)),  # chamber: lift the foreleg
        (0.55, ((0.23, 0.03, -0.02),)),  # extend straight at the head
        (1.0, ((0.09, 0.087, GROUND_Z),)),
    ), duration=0.09),
    "kick": Move(("T2",), (
        (0.0, ((0.025, 0.164, GROUND_Z),)),
        (0.3, ((0.03, 0.2, -0.05),)),  # raise the knee out to the side
        (0.6, ((0.14, 0.12, -0.05),)),  # sweep forward and in
        (1.0, ((0.025, 0.164, GROUND_Z),)),
    ), duration=0.13),
    # Rear up on the hind legs with forelegs high, then drop forward onto the
    # opponent. Real male flies lunge exactly like this.
    "lunge": Move(("T1", "T3"), (
        (0.0, ((0.09, 0.087, GROUND_Z), (-0.181, 0.105, GROUND_Z))),
        (0.4, ((0.14, 0.07, -0.02), (-0.12, 0.09, -0.19))),
        (0.7, ((0.22, 0.05, -0.1), (-0.26, 0.1, -0.15))),
        (1.0, ((0.09, 0.087, GROUND_Z), (-0.181, 0.105, GROUND_Z))),
    ), duration=0.15, bilateral=True),
}
HOLDS = {
    "clinch": (0.2, 0.045, -0.03),  # forelegs reach forward to grab
    "guard": (0.14, 0.05, 0.0),  # forelegs up in front of the head
}


def _leg(segment: str, side: str) -> str:
    return f"{segment}_{'left' if side == 'l' else 'right'}"


def _flip(p, side: str) -> np.ndarray:
    return np.array([p[0], p[1] if side == "l" else -p[1], p[2]])


def build_tables(model: mujoco.MjModel, verbose: bool = False) -> dict[str, np.ndarray]:
    """Solve IK for the gait cycle and every strike keyframe."""
    ik = LegIK(model)
    rest = {leg: ik.claw(leg) for leg in LEGS}
    tables = {}

    gait = np.zeros((len(STRIDE_SCALES), N_PHASE, len(LEGS), len(IK_JOINTS)))
    worst = 0.0
    for si, s in enumerate(STRIDE_SCALES):
        for li, leg in enumerate(LEGS):
            init = None
            offset = 0.0 if leg in TRIPOD_A else 0.5
            for pi in range(N_PHASE):
                ph = (pi / N_PHASE + offset) % 1.0
                target = rest[leg].copy()
                target[2] = GROUND_Z
                half = s * STRIDE / 2
                if ph < 0.5:  # stance: claw sweeps back along the ground
                    target[0] += half * (1 - 4 * ph)
                else:  # swing: lift and bring the claw forward
                    u = (ph - 0.5) / 0.5
                    target[0] += half * (2 * u - 1)
                    target[2] += LIFT * np.sin(np.pi * u)
                ang, r = ik.solve(leg, target, init=init)
                worst = max(worst, r)
                init = ang
                gait[si, pi, li] = [ang[j] for j in IK_JOINTS]
    tables["gait"] = gait
    if verbose:
        print(f"gait IK worst residual {worst:.4f} cm")

    standing = gait[len(STRIDE_SCALES) // 2, 0]
    for name, mv in MOVES.items():
        for side in "lr":
            arr = np.repeat(standing[None], len(mv.keys), axis=0)
            for ki, (_, targets) in enumerate(mv.keys):
                for seg, p in zip(mv.legs, targets):
                    for sd in ("lr" if mv.bilateral else side):
                        leg = _leg(seg, sd)
                        ang, r = ik.solve(leg, _flip(p, sd))
                        if verbose and r > 0.02:
                            print(f"  {name}_{side} key{ki} {leg}: residual {r:.3f} cm")
                        arr[ki, LEGS.index(leg)] = [ang[j] for j in IK_JOINTS]
            tables[f"move_{name}_{side}"] = arr
    for name, p in HOLDS.items():
        arr = standing.copy()
        for sd in "lr":
            leg = _leg("T1", sd)
            ang, _ = ik.solve(leg, _flip(p, sd))
            arr[LEGS.index(leg)] = [ang[j] for j in IK_JOINTS]
        tables[f"hold_{name}"] = arr
    return tables


def load_tables(model: mujoco.MjModel, rebuild: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rebuild:
        with np.load(CACHE) as f:
            return dict(f)
    tables = build_tables(model, verbose=True)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **tables)
    return tables


class MotorSystem:
    """Turns a command vector into actuator targets for one fighter."""

    def __init__(self, fighter: FighterIndex, tables: dict[str, np.ndarray]):
        self.t = tables
        self.joint_ctrl = np.array([[fighter.ctrl[f"{j}_{leg}"] for j in IK_JOINTS]
                                    for leg in LEGS])
        self.adhesion_ctrl = np.array([fighter.ctrl[f"adhere_claw_{leg}"]
                                       for leg in LEGS])
        self.reset()

    def reset(self):
        self.phase = 0.0
        self.active: tuple[str, float] | None = None  # (strike, elapsed seconds)
        self.prev_cmd = np.zeros(len(COMMANDS))

    def _gait_pose(self, forward: float, turn: float) -> np.ndarray:
        """Joint angles (legs x joints) for the current gait phase."""
        pi = int(self.phase * N_PHASE) % N_PHASE
        pose = np.zeros((len(LEGS), len(IK_JOINTS)))
        for li, leg in enumerate(LEGS):
            s = forward + (-turn if leg.endswith("left") else turn)
            x = (np.clip(s, -1, 1) + 1) / 2 * (len(STRIDE_SCALES) - 1)
            i0 = min(int(x), len(STRIDE_SCALES) - 2)
            w = x - i0
            g = self.t["gait"]
            pose[li] = (1 - w) * g[i0, pi, li] + w * g[i0 + 1, pi, li]
        return pose

    def step(self, cmd: np.ndarray, dt: float, ctrl: np.ndarray) -> str | None:
        """Write actuator targets into `ctrl`. Returns a strike name when one starts."""
        c = dict(zip(COMMANDS, cmd))
        started = None
        if self.active is None:
            for name in STRIKES:
                i = COMMANDS.index(name)
                if cmd[i] > 0.5 and self.prev_cmd[i] <= 0.5:
                    self.active, started = (name, 0.0), name
                    break
        self.prev_cmd = np.array(cmd, dtype=float)

        # Fast backward walking tips the fly over, so cap it like a real fly.
        forward = max(c["forward"], -0.5)
        speed = min(abs(forward) + abs(c["turn"]), 1.0)
        self.phase = (self.phase + GAIT_HZ * dt * speed) % 1.0
        pose = self._gait_pose(forward, c["turn"])
        adhesion = np.zeros(len(LEGS))
        for li, leg in enumerate(LEGS):
            offset = 0.0 if leg in TRIPOD_A else 0.5
            in_stance = ((self.phase + offset) % 1.0) < 0.5 or speed < 0.05
            adhesion[li] = 0.6 if in_stance else 0.0

        # Held postures take over the forelegs.
        for hold in ("clinch", "guard"):
            if c[hold] > 0.5:
                pose[:2] = self.t[f"hold_{hold}"][:2]
                adhesion[:2] = 1.0 if hold == "clinch" else 0.0
                break

        if self.active is not None:
            name, el = self.active
            base, _, side = name.partition("_")
            mv = MOVES[base]
            frac = el / mv.duration
            if frac >= 1.0:
                self.active = None
            else:
                arr = self.t[f"move_{base}_{side or 'l'}"]
                times = np.array([k[0] for k in mv.keys])
                k = min(int(np.searchsorted(times, frac, side="right")) - 1,
                        len(times) - 2)
                w = (frac - times[k]) / (times[k + 1] - times[k])
                for seg in mv.legs:
                    for sd in ("lr" if mv.bilateral else side):
                        li = LEGS.index(_leg(seg, sd))
                        pose[li] = (1 - w) * arr[k, li] + w * arr[k + 1, li]
                        adhesion[li] = 0.0
                self.active = (name, el + dt)

        ctrl[self.joint_ctrl] = pose
        ctrl[self.adhesion_ctrl] = adhesion
        return started

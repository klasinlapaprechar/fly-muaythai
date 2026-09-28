"""Two-fly Muay Thai bout: physics, referee, observations and rewards.

Timing: physics at 0.5 ms, motor system at 1 ms, brain decisions at 10 ms.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from flymt import arena, motor

PHYSICS_DT = 5e-4
MOTOR_DT = 1e-3
BRAIN_DT = 1e-2
ROUND_SECONDS = 6.0

HIT_POINTS = {"jab_l": 1.0, "jab_r": 1.0, "kick_l": 1.5, "kick_r": 1.5, "lunge": 2.0}
HIT_FORCE = 0.02  # contact force (g*cm/s^2) that counts as a clean strike; real hits land ~0.1-1.5
TOUCH_SCALE = 0.05  # force that saturates the touch sensors
KNOCKDOWN_UP = 0.3  # thorax "up" z below this = on its side or back
KNOCKDOWN_SECONDS = 0.15
KNOCKDOWN_POINTS = 5.0

OBS_FIELDS = (
    # Vision: where the opponent is (egocentric).
    "opp_dist", "opp_bearing_sin", "opp_bearing_cos", "opp_elev",
    "opp_facing_sin", "opp_facing_cos", "opp_closing_speed",
    # Vision: what the opponent is doing (motion cues of an incoming strike).
    *(f"opp_{s}" for s in motor.STRIKES), "opp_guard", "opp_clinch",
    # Mechanosensation: being touched / hit.
    "touch_head", "touch_thorax", "touch_abdomen", "touch_legs",
    # Proprioception and balance.
    "up", "vel_fwd", "vel_side", "yaw_rate",
    *(f"self_{s}" for s in motor.STRIKES), "rope_dist",
)
PARTS = ("head", "thorax", "abdomen", "legs")


@dataclass
class Score:
    points: float = 0.0
    hits: dict[str, int] = field(default_factory=dict)
    knockdowns: int = 0


class FightEnv:
    def __init__(self, seed: int | None = None):
        self.model, self.idx = arena.build()
        self.model.opt.timestep = PHYSICS_DT
        self.model.opt.noslip_iterations = 0
        self.data = mujoco.MjData(self.model)
        tables = motor.load_tables(self.model)
        self.motor = {n: motor.MotorSystem(self.idx[n], tables) for n in arena.FIGHTERS}
        self.rng = np.random.default_rng(seed)
        self._geom_owner = np.full(self.model.ngeom, -1)
        self._geom_part = np.full(self.model.ngeom, 3)
        for k, name in enumerate(arena.FIGHTERS):
            prefix = f"{name}/"
            for g in range(self.model.ngeom):
                b = self.model.geom_bodyid[g]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if bname.startswith(prefix):
                    self._geom_owner[g] = k
                    part = bname[len(prefix):]
                    self._geom_part[g] = (0 if part.startswith("head") else
                                          1 if part == "thorax" else
                                          2 if part.startswith("abdomen") else 3)
        self._strike_set = [set(self.idx[n].strike_geoms.tolist()) for n in arena.FIGHTERS]
        self._target_set = [set(self.idx[n].target_geoms.tolist()) for n in arena.FIGHTERS]
        self.obs_size = len(OBS_FIELDS)
        self.cmd_size = len(motor.COMMANDS)

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        # Random stance: each fly somewhere on its half, roughly facing the other.
        for k, name in enumerate(arena.FIGHTERS):
            f = self.idx[name]
            side = -1 if k == 0 else 1
            x = side * self.rng.uniform(0.2, 0.4)
            y = self.rng.uniform(-0.2, 0.2)
            yaw = (0.0 if k == 0 else np.pi) + self.rng.uniform(-0.6, 0.6)
            d.qpos[f.qpos_adr:f.qpos_adr + 3] = [x, y, 0.105]
            d.qpos[f.qpos_adr + 3:f.qpos_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for m in self.motor.values():
            m.reset()
        self._touch = {n: np.zeros(4) for n in arena.FIGHTERS}
        idle = np.zeros(self.cmd_size)
        for _ in range(60):  # let both flies settle onto their feet
            self._motor_tick({n: idle for n in arena.FIGHTERS})
        self.t = 0.0
        self.score = {n: Score() for n in arena.FIGHTERS}
        self._down_time = {n: 0.0 for n in arena.FIGHTERS}
        self._landed = {n: False for n in arena.FIGHTERS}  # one hit per strike
        return self.observe()

    def step(self, cmds: dict[str, np.ndarray]):
        """Advance one brain tick (10 ms). Returns obs, rewards, done, events."""
        events = []
        rewards = {n: 0.0 for n in arena.FIGHTERS}
        self._touch = {n: np.zeros(4) for n in arena.FIGHTERS}
        dist_before = self._distance()
        for _ in range(int(round(BRAIN_DT / MOTOR_DT))):
            started = self._motor_tick(cmds)
            for n, s in started.items():
                if s:
                    self._landed[n] = False
                    events.append(("throw", n, s))
            for attacker, victim, force in self._scan_hits():
                strike = self.motor[attacker].active
                if strike is None or self._landed[attacker] or force < HIT_FORCE:
                    continue
                pts = HIT_POINTS[strike[0]]
                self._landed[attacker] = True
                sc = self.score[attacker]
                sc.points += pts
                sc.hits[strike[0]] = sc.hits.get(strike[0], 0) + 1
                rewards[attacker] += pts
                rewards[victim] -= pts
                events.append(("hit", attacker, strike[0], victim, force))
        self.t += BRAIN_DT

        done = self.t >= ROUND_SECONDS
        for k, n in enumerate(arena.FIGHTERS):
            other = arena.FIGHTERS[1 - k]
            self._down_time[n] = self._down_time[n] + BRAIN_DT if self._up(n) < KNOCKDOWN_UP else 0.0
            if self._down_time[n] >= KNOCKDOWN_SECONDS:
                self.score[other].points += KNOCKDOWN_POINTS
                self.score[n].knockdowns += 1
                rewards[other] += KNOCKDOWN_POINTS
                rewards[n] -= KNOCKDOWN_POINTS
                events.append(("knockdown", other, n))
                done = True
        # Light shaping: close the distance and face the opponent.
        closing = dist_before - self._distance()
        for n in arena.FIGHTERS:
            rewards[n] += 2.0 * closing + 0.002 * self._facing(n)
        return self.observe(), rewards, done, events

    def _motor_tick(self, cmds) -> dict[str, str | None]:
        started = {n: self.motor[n].step(cmds[n], MOTOR_DT, self.data.ctrl)
                   for n in arena.FIGHTERS}
        for _ in range(int(round(MOTOR_DT / PHYSICS_DT))):
            mujoco.mj_step(self.model, self.data)
        return started

    def _scan_hits(self):
        """Yield (attacker, victim, force) for strike-geom-on-target contacts.

        Also accumulates per-body-part touch for the mechanosensory inputs.
        """
        d = self.data
        force = np.zeros(6)
        hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            o1, o2 = self._geom_owner[c.geom1], self._geom_owner[c.geom2]
            if o1 < 0 or o2 < 0 or o1 == o2:
                continue
            mujoco.mj_contactForce(self.model, d, i, force)
            fn = abs(force[0])
            for ga, oa, gb, ob in ((c.geom1, o1, c.geom2, o2), (c.geom2, o2, c.geom1, o1)):
                victim = arena.FIGHTERS[ob]
                self._touch[victim][self._geom_part[gb]] += fn
                if ga in self._strike_set[oa] and gb in self._target_set[ob]:
                    hits.append((arena.FIGHTERS[oa], victim, fn))
        return hits

    def _frame(self, n):
        f = self.idx[n]
        return self.data.xpos[f.thorax_body], self.data.xmat[f.thorax_body].reshape(3, 3)

    def _up(self, n) -> float:
        return float(self._frame(n)[1][2, 2])

    def _distance(self) -> float:
        a, _ = self._frame("red")
        b, _ = self._frame("blue")
        return float(np.linalg.norm((a - b)[:2]))

    def _facing(self, n) -> float:
        k = arena.FIGHTERS.index(n)
        p, R = self._frame(n)
        q, _ = self._frame(arena.FIGHTERS[1 - k])
        rel = R.T @ (q - p)
        return float(rel[0] / (np.linalg.norm(rel[:2]) + 1e-6))

    def observe(self) -> dict[str, np.ndarray]:
        out = {}
        for k, n in enumerate(arena.FIGHTERS):
            other = arena.FIGHTERS[1 - k]
            p, R = self._frame(n)
            q, Ro = self._frame(other)
            rel = R.T @ (q - p)
            dist = np.linalg.norm(rel[:2]) + 1e-6
            opp_fwd = R.T @ Ro[:, 0]  # where the opponent is pointing, relative to us
            v = self.data.cvel[self.idx[n].thorax_body]  # [angular(3), linear(3)]
            v_o = self.data.cvel[self.idx[other].thorax_body]
            vel = R.T @ v[3:]
            closing = -float(np.dot((q - p)[:2] / dist, (v_o[3:] - v[3:])[:2]))
            opp_active = self.motor[other].active
            my_active = self.motor[n].active
            oc = self.motor[other].prev_cmd
            o = [
                dist, rel[1] / dist, rel[0] / dist, rel[2],
                opp_fwd[1], opp_fwd[0], closing,
                *(float(opp_active is not None and opp_active[0] == s) for s in motor.STRIKES),
                float(oc[motor.COMMANDS.index("guard")] > 0.5),
                float(oc[motor.COMMANDS.index("clinch")] > 0.5),
                *np.tanh(self._touch[n] / TOUCH_SCALE),
                R[2, 2], vel[0], vel[1], v[2],
                *(float(my_active is not None and my_active[0] == s) for s in motor.STRIKES),
                arena.RING_RADIUS - np.linalg.norm(p[:2]),
            ]
            out[n] = np.asarray(o, dtype=np.float32)
        return out

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
STRIKE_COST = 0.1  # per strike thrown (fatigue)
MISS_PENALTY = 0.1  # extra when a strike ends without landing
LUNGE_MISS_PENALTY = 0.7  # extra on top for a missed lunge: an overcommitted miss
HEAD_MULTIPLIER = 1.5  # strikes that land on the head score more
COMBO_WINDOW = 0.5  # s: a different strike landing this soon after a hit...
COMBO_BONUS = 0.5  # ...earns this on top
BOX_STANCE_REWARD = 0.005  # per tick fighting from the boxing stance at range, facing
# Fight-craft shaping. Per tick = per 10 ms; hits (1-2) and knockdowns (5) dominate.
APPROACH_REWARD = 5.0  # per cm this fighter itself moves toward the opponent
FACING_REWARD = 0.02  # per tick, scaled by cos(bearing)
IN_RANGE_REWARD = 0.02  # per tick within striking range and facing
RANGE_CM = 0.25
CLEAN_FACING = 0.707  # a strike only scores if the attacker faces the target within 45 deg
SIDEWAYS_PENALTY = 0.06  # per tick in range, scaled from 0 (head-on) to full (back turned)
UNFACED_STRIKE_PENALTY = 0.3  # throwing a strike while not facing within 45 deg
# Orienting stage: the reward is only about keeping the opponent dead ahead.
ORIENT_REWARD = 0.02  # per tick, scaled by cos(bearing)
ORIENT_AWAY_PENALTY = 0.02  # per tick with the opponent behind
ENGAGED_CM = 0.5
CIRCLE_REWARD = 0.01  # per tick of sideways movement around the opponent, in range and facing
CIRCLE_SPEED = 0.3  # cm/s of sideways speed that earns the full circling reward
DEFEND_REWARD = 0.5  # opponent's strike misses while within striking range (slipped)
BLOCK_BONUS = 0.3  # ...and it was thrown into a guard or boxing stance (blocked)
# Resets: when an exchange goes bad (in range but not facing), back out and square up.
BAD_FACING = 0.5  # facing below this (~60 deg off) while in range is a bad position
RESET_OUT_REWARD = 3.0  # per cm backed out of a bad position
RESET_DONE_DIST = 0.3  # cm: far enough out to count as reset...
RESET_DONE_FACING = 0.87  # ...and squared up again (within 30 deg)
RESET_BONUS = 0.4  # for completing a reset
RESET_MAX_SECONDS = 1.5  # a reset that takes longer than this is abandoned
# Clinch: control the opponent chest to chest and strike from there (knees).
CLINCH_DIST = 0.18  # cm
CLINCH_CONTROL_REWARD = 0.02  # per tick clinched, close and facing
CLINCH_STRIKE_MULTIPLIER = 1.5  # strikes landed from the clinch score more
RETREAT_GRACE = 1.0  # s of backing off *without throwing a strike* before it counts as running
RETREAT_PENALTY = 0.02  # per tick of running (retreating without fighting back)
PASSIVE_SECONDS = 1.0  # going this long without a strike, and not closing in...
PASSIVE_PENALTY = 0.05  # ...costs this per tick (must outweigh all positional rewards)

OBS_FIELDS = (
    # Vision: where the opponent is (egocentric).
    "opp_dist", "opp_bearing_sin", "opp_bearing_cos", "opp_elev",
    "opp_facing_sin", "opp_facing_cos", "opp_closing_speed",
    # Vision: what the opponent is doing (motion cues of an incoming strike).
    *(f"opp_{s}" for s in motor.STRIKES), "opp_guard", "opp_clinch", "opp_box",
    # Mechanosensation: being touched / hit.
    "touch_head", "touch_thorax", "touch_abdomen", "touch_legs",
    # Proprioception and balance.
    "up", "vel_fwd", "vel_side", "yaw_rate",
    *(f"self_{s}" for s in motor.STRIKES), "rope_dist",
    # Reward: points this fighter scored in the last tick (drives PAM dopamine neurons).
    "reward_signal",
)
PARTS = ("head", "thorax", "abdomen", "legs")


@dataclass
class Score:
    points: float = 0.0
    hits: dict[str, int] = field(default_factory=dict)
    knockdowns: int = 0


class FightEnv:
    def __init__(self, seed: int | None = None, mode: str = "fight"):
        self.mode = mode  # "fight", or "orient" (reward only for facing the opponent)
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
            if self.mode == "orient":  # start pointing anywhere: facing must be earned
                yaw = self.rng.uniform(-np.pi, np.pi)
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
        self._scored = {n: 0.0 for n in arena.FIGHTERS}
        self._retreat_time = {n: 0.0 for n in arena.FIGHTERS}
        self._since_strike = {n: 0.0 for n in arena.FIGHTERS}
        self._last_hit = {n: (-1.0, None) for n in arena.FIGHTERS}  # (time, strike)
        self._resetting = {n: 0.0 for n in arena.FIGHTERS}  # seconds into a reset, 0 = none
        return self.observe()

    def step(self, cmds: dict[str, np.ndarray]):
        """Advance one brain tick (10 ms). Returns obs, rewards, done, events."""
        events = []
        rewards = {n: 0.0 for n in arena.FIGHTERS}
        self._scored = {n: 0.0 for n in arena.FIGHTERS}
        self._touch = {n: np.zeros(4) for n in arena.FIGHTERS}
        pos_before = {n: self._frame(n)[0][:2].copy() for n in arena.FIGHTERS}
        threw = set()
        for _ in range(int(round(BRAIN_DT / MOTOR_DT))):
            before = {n: self.motor[n].active for n in arena.FIGHTERS}
            started = self._motor_tick(cmds)
            for k, n in enumerate(arena.FIGHTERS):
                # A strike that ended without landing was slipped or blocked.
                prev, cur = before[n], self.motor[n].active
                ended = prev is not None and (cur is None or cur[1] < prev[1])
                if ended and not self._landed[n]:
                    rewards[n] -= MISS_PENALTY + (LUNGE_MISS_PENALTY if prev[0] == "lunge" else 0)
                    events.append(("miss", n, prev[0]))
                if ended and not self._landed[n] and self._distance() < RANGE_CM * 1.4:
                    defender = arena.FIGHTERS[1 - k]
                    rewards[defender] += DEFEND_REWARD
                    blocked = self.motor[defender].hold in ("guard", "box")
                    if blocked:
                        rewards[defender] += BLOCK_BONUS
                    events.append(("block" if blocked else "slip", defender, prev[0]))
            for n, s in started.items():
                if s:
                    self._landed[n] = False
                    self._since_strike[n] = 0.0
                    threw.add(n)
                    rewards[n] -= STRIKE_COST
                    if self._facing(n) < CLEAN_FACING:
                        rewards[n] -= UNFACED_STRIKE_PENALTY  # not picking its shot
                    events.append(("throw", n, s))
            for attacker, victim, force, part in self._scan_hits():
                strike = self.motor[attacker].active
                if strike is None or self._landed[attacker] or force < HIT_FORCE:
                    continue
                if self._facing(attacker) < CLEAN_FACING:
                    continue  # side-on or backward contact is not a clean strike
                pts = HIT_POINTS[strike[0]] * (HEAD_MULTIPLIER if part == 0 else 1.0)
                if self.motor[attacker].hold == "clinch":
                    pts *= CLINCH_STRIKE_MULTIPLIER
                    events.append(("clinch_strike", attacker, strike[0]))
                self._landed[attacker] = True
                t_last, s_last = self._last_hit[attacker]
                if s_last is not None and s_last != strike[0] and self.t - t_last <= COMBO_WINDOW:
                    rewards[attacker] += COMBO_BONUS
                    events.append(("combo", attacker, s_last, strike[0]))
                self._last_hit[attacker] = (self.t, strike[0])
                sc = self.score[attacker]
                sc.points += pts
                sc.hits[strike[0]] = sc.hits.get(strike[0], 0) + 1
                rewards[attacker] += pts
                rewards[victim] -= pts
                self._scored[attacker] += pts
                events.append(("hit", attacker, strike[0], victim, force,
                               ("head", "thorax", "abdomen", "legs")[part]))
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
                self._scored[other] += KNOCKDOWN_POINTS
                events.append(("knockdown", other, n))
                done = True
        if self.mode == "orient":
            for n in arena.FIGHTERS:
                facing = self._facing(n)
                kd = (KNOCKDOWN_POINTS if any(e[0] == "knockdown" and e[2] == n for e in events)
                      else 0.0)
                rewards[n] = (ORIENT_REWARD * facing
                              - (ORIENT_AWAY_PENALTY if facing < 0 else 0.0) - kd)
            return self.observe(), rewards, done, events
        # Fight-craft shaping, scored on each fighter's own movement.
        dist = self._distance()
        for k, n in enumerate(arena.FIGHTERS):
            other = arena.FIGHTERS[1 - k]
            p = self._frame(n)[0][:2]
            to_opp = self._frame(other)[0][:2] - p
            u = to_opp / (np.linalg.norm(to_opp) + 1e-6)
            moved = p - pos_before[n]
            toward = float(moved @ u)  # cm this fighter moved toward the opponent
            sideways = abs(float(moved @ np.array([-u[1], u[0]]))) / BRAIN_DT  # cm/s
            facing = self._facing(n)
            # Positional rewards only count while actively fighting (a strike in
            # the last second); otherwise standing in range doing nothing pays.
            active = self._since_strike[n] <= PASSIVE_SECONDS
            r = APPROACH_REWARD * toward + FACING_REWARD * facing
            if dist < RANGE_CM and facing > 0.8 and active:
                r += IN_RANGE_REWARD
            if dist < ENGAGED_CM:
                r -= SIDEWAYS_PENALTY * (1 - facing) / 2  # stay face to face
            if dist < ENGAGED_CM * 0.8 and facing > 0.7 and active:
                r += CIRCLE_REWARD * min(sideways / CIRCLE_SPEED, 1.0)
                if self.motor[n].hold == "box":
                    r += BOX_STANCE_REWARD
            # Resets: a bad position (in range, not facing) should be escaped by
            # backing out and squaring up again, like breaking off a scramble.
            if dist < RESET_DONE_DIST and facing < BAD_FACING and not self._resetting[n]:
                self._resetting[n] = BRAIN_DT
            if self._resetting[n]:
                r += RESET_OUT_REWARD * max(-toward, 0.0)
                self._resetting[n] += BRAIN_DT
                if dist >= RESET_DONE_DIST and facing >= RESET_DONE_FACING:
                    self._resetting[n] = 0.0  # out and squared up: reset complete
                    r += RESET_BONUS
                    events.append(("reset", n))
                elif self._resetting[n] > RESET_MAX_SECONDS:
                    self._resetting[n] = 0.0
            if (self.motor[n].hold == "clinch" and dist < CLINCH_DIST and facing > CLEAN_FACING
                    and active):
                r += CLINCH_CONTROL_REWARD
            # Running: backing off without fighting back. Retreating while still
            # throwing strikes is fine (fighting off the back foot), so any strike
            # resets the clock, as does stopping or coming forward. A reset is not running.
            retreating = toward < -0.05 * BRAIN_DT
            self._retreat_time[n] = (self._retreat_time[n] + BRAIN_DT
                                     if retreating and n not in threw
                                     and not self._resetting[n] else 0.0)
            if self._retreat_time[n] > RETREAT_GRACE:
                r -= RETREAT_PENALTY
            # Not fighting back: in range but no strikes for too long.
            self._since_strike[n] += BRAIN_DT
            # At any distance: standing off (not closing in) without striking is
            # passive. Walking in to engage and resetting are not.
            if (self._since_strike[n] > PASSIVE_SECONDS and toward <= 0.05 * BRAIN_DT
                    and not self._resetting[n]):
                r -= PASSIVE_PENALTY
            rewards[n] += r
        return self.observe(), rewards, done, events

    def _motor_tick(self, cmds) -> dict[str, str | None]:
        started = {n: self.motor[n].step(cmds[n], MOTOR_DT, self.data.ctrl)
                   for n in arena.FIGHTERS}
        for _ in range(int(round(MOTOR_DT / PHYSICS_DT))):
            mujoco.mj_step(self.model, self.data)
        return started

    def _scan_hits(self):
        """Return (attacker, victim, force, body part) for strike-on-target contacts.

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
                    hits.append((arena.FIGHTERS[oa], victim, fn, int(self._geom_part[gb])))
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
                float(oc[motor.COMMANDS.index("box")] > 0.5),
                *np.tanh(self._touch[n] / TOUCH_SCALE),
                R[2, 2], vel[0], vel[1], v[2],
                *(float(my_active is not None and my_active[0] == s) for s in motor.STRIKES),
                arena.RING_RADIUS - np.linalg.norm(p[:2]),
                np.tanh(self._scored[n]),
            ]
            out[n] = np.asarray(o, dtype=np.float32)
        return out

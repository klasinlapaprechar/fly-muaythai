"""Per-leg inverse kinematics in the thorax frame.

Used offline to turn claw target positions (gait cycles, strike keyframes) into
joint-angle tables that the motor system plays back at runtime.
"""

import mujoco
import numpy as np

from flymt.arena import root_joint_name

LEGS = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur",
              "tibia", "tarsus")
# Joints the solver may move; the rest keep their rest angle.
IK_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia")


class LegIK:
    """Damped least-squares IK for one fighter's legs with the thorax pinned."""

    def __init__(self, model: mujoco.MjModel, fighter: str = "red"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.prefix = f"{fighter}/"
        self.thorax = self._body("thorax")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                root_joint_name(fighter))
        adr = model.jnt_qposadr[jid]
        self.d.qpos[adr:adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(model, self.d)
        self.rest_qpos = self.d.qpos.copy()

    def _body(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.prefix + name)
        assert i >= 0, name
        return i

    def _joint(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name)
        assert i >= 0, name
        return i

    def claw(self, leg: str, angles: dict[str, float] | None = None) -> np.ndarray:
        """Claw position in the thorax frame (x forward, y left, z up)."""
        self.d.qpos[:] = self.rest_qpos
        for j, v in (angles or {}).items():
            self.d.qpos[self.m.jnt_qposadr[self._joint(f"{j}_{leg}")]] = v
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self._body(f"claw_{leg}")].copy()

    def solve(self, leg: str, target: np.ndarray, init: dict[str, float] | None = None,
              iters: int = 200, damping: float = 1e-3, rest_pull: float = 0.02
              ) -> tuple[dict[str, float], float]:
        """Return joint angles putting the claw at `target`, and the residual (cm)."""
        joints = [self._joint(f"{j}_{leg}") for j in IK_JOINTS]
        qadr = np.array([self.m.jnt_qposadr[j] for j in joints])
        dadr = np.array([self.m.jnt_dofadr[j] for j in joints])
        lo, hi = self.m.jnt_range[joints].T
        claw = self._body(f"claw_{leg}")

        q = self.rest_qpos.copy()
        for j, v in (init or {}).items():
            q[self.m.jnt_qposadr[self._joint(f"{j}_{leg}")]] = v
        rest = self.rest_qpos[qadr]
        jacp = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            err = target - self.d.xpos[claw]
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacBody(self.m, self.d, jacp, None, claw)
            J = jacp[:, dadr]
            # Secondary objective: stay near the rest pose.
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            dq += rest_pull * (rest - q[qadr])
            q[qadr] = np.clip(q[qadr] + dq, lo, hi)
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        resid = float(np.linalg.norm(target - self.d.xpos[claw]))
        return {n: float(q[a]) for n, a in zip(IK_JOINTS, qadr)}, resid

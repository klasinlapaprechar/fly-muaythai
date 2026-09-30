"""Two-fly fighting ring built from flybody's fruit fly MJCF model.

Units follow flybody: centimeters, grams, seconds. A fly is ~0.25 cm long.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from dm_control import mjcf
from flybody.fruitfly.fruitfly import FruitFly

FIGHTERS = ("red", "blue")
RING_RADIUS = 0.8  # cm, a few body lengths across
WALL_SEGMENTS = 32
START_GAP = 0.45  # cm between thoraces at the bell


@dataclass
class FighterIndex:
    """Index lookups into the compiled model for one fighter."""

    name: str
    qpos_adr: int  # start of the freejoint qpos (7 values)
    qvel_adr: int  # start of the freejoint qvel (6 values)
    thorax_body: int
    ctrl: dict[str, int]  # actuator short name -> ctrl index
    body_ids: np.ndarray  # every body belonging to this fighter
    strike_geoms: np.ndarray  # leg/head collision geoms that can land a hit
    target_geoms: np.ndarray  # geoms on head/thorax/abdomen that can be hit


def _build_mjcf() -> mjcf.RootElement:
    root = mjcf.RootElement(model="fly_muaythai")
    root.compiler.angle = "radian"
    root.compiler.autolimits = True
    root.option.timestep = 1e-4
    root.option.gravity = (0, 0, -981)
    root.option.density = 0.00128
    root.option.viscosity = 0.000185
    root.option.cone = "elliptic"
    root.option.noslip_iterations = 3
    root.size.njmax = 2000
    root.size.nconmax = 600

    root.asset.add("texture", name="grid", type="2d", builtin="checker",
                   rgb1=".22 .24 .28", rgb2=".18 .2 .23", width=300, height=300)
    root.asset.add("material", name="canvas", texture="grid", texrepeat="12 12")
    root.asset.add("texture", name="sky", type="skybox", builtin="gradient",
                   rgb1=".32 .42 .55", rgb2=".08 .1 .14", width=256, height=256)
    root.worldbody.add("light", pos=(0, 0, 3), dir=(0, 0, -1), directional=True)
    root.worldbody.add("geom", name="floor", type="plane", size=(2, 2, 0.1),
                       material="canvas", friction=(1, 0.005, 0.0001))
    # Ring ropes: short boxes arranged on a circle.
    seg_len = 2 * np.pi * RING_RADIUS / WALL_SEGMENTS
    for i in range(WALL_SEGMENTS):
        a = 2 * np.pi * i / WALL_SEGMENTS
        root.worldbody.add(
            "geom", name=f"rope_{i}", type="box",
            size=(0.01, seg_len / 2 + 0.005, 0.08),
            pos=(RING_RADIUS * np.cos(a), RING_RADIUS * np.sin(a), 0.08),
            euler=(0, 0, a), rgba=(0.9, 0.85, 0.7, 0.35))
    root.worldbody.add("camera", name="ringside", pos=(0, -1.6, 1.1),
                       xyaxes=(1, 0, 0, 0, 0.57, 0.82))
    root.worldbody.add("camera", name="overhead", pos=(0, 0, 2.2),
                       xyaxes=(1, 0, 0, 0, 1, 0))

    colors = {"red": (0.85, 0.25, 0.2, 1), "blue": (0.2, 0.4, 0.9, 1)}
    for k, name in enumerate(FIGHTERS):
        fly = FruitFly(name=name, use_wings=False, use_mouth=False,
                       use_antennae=False, joint_filter=0.0)
        model = fly.mjcf_model
        # First-person camera on the thorax, at about eye height just above the
        # head, looking forward and slightly down so the forelegs are in view.
        model.find("body", "thorax").add(
            "camera", name="fpv", pos=(0.07, 0, 0.075), xyaxes=(0, -1, 0, 0.3, 0, 1), fovy=95)
        # Tint the fighter so the corners are obvious.
        for mat in model.find_all("material"):
            if mat.name in ("body", "blue", "red"):
                mat.rgba = colors[name]
        side = -1 if k == 0 else 1
        yaw = 0.0 if k == 0 else np.pi
        frame = root.attach(model)
        frame.pos = (side * START_GAP / 2, 0, 0.14)
        frame.quat = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        frame.add("freejoint", name="root")
    return root


def build() -> tuple[mujoco.MjModel, dict[str, FighterIndex]]:
    root = _build_mjcf()
    xml = root.to_xml_string()
    model = mujoco.MjModel.from_xml_string(xml, root.get_assets())
    return model, _index(model)


def root_joint_name(fighter: str) -> str:
    # mjcf names a freejoint on an attachment frame "<model>/<joint>/".
    return f"{fighter}/root/"


def _index(model: mujoco.MjModel) -> dict[str, FighterIndex]:
    out = {}
    for name in FIGHTERS:
        prefix = f"{name}/"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name(name))
        assert jid >= 0, f"missing freejoint for {name}"
        ctrl = {}
        for i in range(model.nu):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n.startswith(prefix):
                ctrl[n[len(prefix):]] = i
        body_ids = np.array([
            b for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            .startswith(prefix)])
        strike, target = [], []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in body_ids or model.geom_contype[g] == 0:
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     model.geom_bodyid[g])[len(prefix):]
            if body.startswith(("tibia", "tarsus", "claw", "head")):
                strike.append(g)
            if body.startswith(("head", "thorax", "abdomen")):
                target.append(g)
        out[name] = FighterIndex(
            name=name,
            qpos_adr=model.jnt_qposadr[jid],
            qvel_adr=model.jnt_dofadr[jid],
            thorax_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"{prefix}thorax"),
            ctrl=ctrl,
            body_ids=body_ids,
            strike_geoms=np.array(strike),
            target_geoms=np.array(target),
        )
    return out

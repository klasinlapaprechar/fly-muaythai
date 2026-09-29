"""Render every move in the fly's repertoire on its own, slowed down and labeled.

Usage:
    python -m flymt.moves_demo        # out/videos/moves.mp4 + docs/img/moves.png
"""

from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flymt import motor
from flymt.fight_env import FightEnv

C = {k: i for i, k in enumerate(motor.COMMANDS)}
FRAME_EVERY = 2  # motor ticks (1 ms each) per frame -> 500 fps of sim time
FPS = 30  # playback, so strikes play ~17x slower than real time

# (label, which legs, command held for the segment, strike to trigger, seconds, distance)
MOVES = [
    ("LEFT JAB", "left foreleg (T1) punches straight ahead", {}, "jab_l", 0.3, 0.26),
    ("RIGHT JAB", "right foreleg (T1) punches straight ahead", {}, "jab_r", 0.3, 0.26),
    ("LEFT KICK", "left middle leg (T2) sweeps forward, a roundhouse", {}, "kick_l", 0.35, 0.26),
    ("RIGHT KICK", "right middle leg (T2) sweeps forward, a roundhouse", {}, "kick_r", 0.35, 0.26),
    ("LUNGE (knee)", "rears up on the hind legs and slams forward", {}, "lunge", 0.4, 0.28),
    ("GUARD", "both forelegs raised in front of the head", {"guard": 1}, None, 0.35, 0.3),
    ("BOXING STANCE", "rears up on middle + hind legs, forelegs up", {"box": 1}, None, 0.45, 0.3),
    ("CLINCH", "forelegs lock on and grip the opponent", {"clinch": 1}, None, 0.45, 0.2),
    ("WALK", "tripod gait: three legs step at a time", {"forward": 1}, None, 0.5, 0.5),
    ("TURN LEFT", "legs on each side step in opposite directions", {"turn": 1}, None, 0.5, 0.4),
]
# Legs that light up yellow during each move.
HIGHLIGHT = {"LEFT JAB": ("T1_left",), "RIGHT JAB": ("T1_right",),
             "LEFT KICK": ("T2_left",), "RIGHT KICK": ("T2_right",),
             "LUNGE (knee)": ("T1_left", "T1_right", "T3_left", "T3_right"),
             "GUARD": ("T1_left", "T1_right"), "CLINCH": ("T1_left", "T1_right"),
             "BOXING STANCE": ("T1_left", "T1_right", "T2_left", "T2_right",
                               "T3_left", "T3_right")}
LEG_SEGMENTS = ("coxa", "femur", "tibia", "tarsus", "tarsus2", "tarsus3", "tarsus4", "claw")


def _leg_geoms(model, legs) -> list[int]:
    bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"red/{seg}_{leg}")
              for leg in legs for seg in LEG_SEGMENTS}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]


def _place(env: FightEnv, dist: float):
    """Put red at the origin facing +x and blue `dist` cm ahead, facing red."""
    d = env.data
    for name, x, yaw in (("red", -dist / 2, 0.0), ("blue", dist / 2, np.pi)):
        f = env.idx[name]
        d.qpos[f.qpos_adr:f.qpos_adr + 3] = [x, 0, 0.105]
        d.qpos[f.qpos_adr + 3:f.qpos_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)


def _label(img: np.ndarray, title: str, sub: str, t: float) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    big = ImageFont.load_default(size=34)
    small = ImageFont.load_default(size=18)
    draw.rectangle([0, 0, im.width, 78], fill=(13, 17, 23))
    draw.text((16, 8), title, fill=(255, 213, 79), font=big)
    draw.text((16, 50), sub, fill=(230, 230, 230), font=small)
    draw.text((16, im.height - 28), f"t = {t * 1000:4.0f} ms   (~17x slow motion)",
              fill=(230, 230, 230), font=small)
    return np.asarray(im)


def _peak_score(env: FightEnv, move: str) -> float:
    m, d = env.model, env.data
    th = env.idx["red"].thorax_body
    R = d.xmat[th].reshape(3, 3)
    if move in ("lunge", "box"):
        return float(R[2, 0])  # body pitch, nose up
    legs = {"jab_l": "claw_T1_left", "jab_r": "claw_T1_right",
            "kick_l": "claw_T2_left", "kick_r": "claw_T2_right"}
    if move in legs:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"red/{legs[move]}")
        return float((R.T @ (d.xpos[b] - d.xpos[th]))[0])  # reach forward
    return float(d.time)  # holds and walking: latest frame


def _contact_sheet(frames, path: Path, cols: int = 2):
    h, w = frames[0].shape[:2]
    small = [np.asarray(Image.fromarray(f).resize((w // 2, h // 2))) for f in frames]
    rows = -(-len(small) // cols)
    sheet = np.full((rows * (h // 2), cols * (w // 2), 3), 13, np.uint8)
    for i, f in enumerate(small):
        r, c = divmod(i, cols)
        sheet[r * (h // 2):(r + 1) * (h // 2), c * (w // 2):(c + 1) * (w // 2)] = f
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, sheet)


def main():
    env = FightEnv(seed=0)
    env.reset()
    behind = mujoco.Renderer(env.model, 420, 560)
    top = mujoco.Renderer(env.model, 420, 560)
    mat0, rgba0 = env.model.geom_matid.copy(), env.model.geom_rgba.copy()
    out = Path("out/videos/moves.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    peaks = []
    zero = np.zeros(len(motor.COMMANDS))
    for title, sub, hold, strike, seconds, dist in MOVES:
        env.reset()
        _place(env, dist)
        for m in env.motor.values():
            m.reset()
        for _ in range(80):  # settle standing
            env._motor_tick({"red": zero, "blue": zero})
        base = zero.copy()
        for k, v in hold.items():
            base[C[k]] = v
        env.model.geom_matid[:], env.model.geom_rgba[:] = mat0, rgba0
        for g in _leg_geoms(env.model, HIGHLIGHT.get(title, ())):
            env.model.geom_matid[g] = -1  # use plain color instead of the body material
            env.model.geom_rgba[g] = (1.0, 0.85, 0.1, 1.0)
        best, best_score, frame = None, -np.inf, None
        for tick in range(int(seconds / 1e-3)):
            cmd = base.copy()
            if strike and tick < 5:
                cmd[C[strike]] = 1.0
            env._motor_tick({"red": cmd, "blue": zero})
            if tick % FRAME_EVERY:
                continue
            p = env.data.xpos[env.idx["red"].thorax_body].copy()
            cam_b = mujoco.MjvCamera()  # behind red, looking at blue: red's left is your left
            cam_b.lookat[:] = p + [0.12, 0, 0.0]
            cam_b.distance, cam_b.azimuth, cam_b.elevation = 0.55, 0, -28
            cam_t = mujoco.MjvCamera()  # straight down, red at the bottom
            cam_t.lookat[:] = p + [dist / 2, 0, 0]
            cam_t.distance, cam_t.azimuth, cam_t.elevation = 0.7, 0, -89
            behind.update_scene(env.data, cam_b)
            top.update_scene(env.data, cam_t)
            frame = _label(np.concatenate([behind.render(), top.render()], 1),
                           title, sub, tick * 1e-3)
            writer.append_data(frame)
            score = _peak_score(env, strike or next(iter(hold), ""))
            if score > best_score:
                best, best_score = frame, score
        for _ in range(12):  # brief hold on the last frame between moves
            writer.append_data(frame)
        peaks.append(best)
    writer.close()
    _contact_sheet(peaks, Path("docs/img/moves.png"))
    print(f"saved {out} and docs/img/moves.png ({len(MOVES)} moves)")


if __name__ == "__main__":
    main()

"""Watch a bout next to the fly's brain: every real neuron glows with its activity.

Usage:
    python -m flymt.visualize --ckpt checkpoints/latest.pt --stage 2
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import mujoco
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flymt import motor, opponents  # noqa: E402
from flymt.brain import CHANNELS, Circuit, TorchBrain  # noqa: E402
from flymt.connectome import ROLE_COLORS, display_positions  # noqa: E402
from flymt.fight_env import FightEnv  # noqa: E402

LABELS = {"vis_target_left": "LC10 vision (L)", "vis_target_right": "LC10 vision (R)",
          "looming": "LPLC2/LC4 looming", "touch_head": "head bristles",
          "touch_body": "body touch", "touch_legs": "leg touch", "balance": "JO balance",
          "aggression": "pC1 aggression", "dopamine": "PAM dopamine",
          "inter": "interneurons", "dn": "descending"}
STRIKE_NAMES = {"jab_l": "LEFT JAB", "jab_r": "RIGHT JAB", "kick_l": "LEFT KICK",
                "kick_r": "RIGHT KICK", "lunge": "LUNGE"}


def record_bout(circuit: Circuit, ckpt: str | None, stage: int, seconds: float,
                seed: int, sample: bool):
    """Run one bout; return per-tick frames, neuron rates, commands and events."""
    env = FightEnv(seed=seed)
    brain = TorchBrain(circuit)
    if ckpt:
        brain.load_state_dict(torch.load(ckpt, map_location="cpu")["brain"])
    opp = opponents.make(min(stage, 2), np.random.default_rng(seed))
    renderer = mujoco.Renderer(env.model, 360, 480)
    cam = mujoco.MjvCamera()
    obs = env.reset()
    h = brain.init_hidden(1)
    frames, rates, cmds, events = [], [], [], []
    while env.t < seconds:
        cmd, _, h = brain.act(torch.from_numpy(obs["red"])[None], h,
                              deterministic=not sample)
        c = cmd[0].numpy()
        obs, _, done, ev = env.step({"red": c, "blue": opp.act(obs["blue"])})
        a, b = (env.data.xpos[env.idx[n].thorax_body] for n in ("red", "blue"))
        cam.lookat[:] = (a + b) / 2
        cam.distance = 0.9 + 0.8 * np.linalg.norm(a - b)
        # Side-on to the line between the fighters.
        cam.azimuth = 90 + np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
        cam.elevation = -25
        renderer.update_scene(env.data, cam)
        frames.append(renderer.render())
        rates.append(h[0].numpy().astype(np.float16))
        cmds.append(c)
        events.append(ev)
        if done:
            break
    return frames, np.array(rates), np.array(cmds), events, env.score


def render_video(circuit, frames, rates, cmds, events, score, out: Path, stills: Path,
                 fps: int = 30):
    xy = display_positions(circuit)
    order = np.argsort(circuit.role != "inter", kind="stable")  # interneurons underneath
    base = np.array([matplotlib.colors.to_rgb(ROLE_COLORS[r]) for r in circuit.role])
    groups = ["inter", *CHANNELS, "dn"]
    masks = [circuit.role == g for g in groups]
    fig = plt.figure(figsize=(13, 6.2), facecolor="#0d1117")
    ax_f = fig.add_axes([0.0, 0.18, 0.47, 0.74])
    ax_b = fig.add_axes([0.48, 0.02, 0.36, 0.93])
    ax_g = fig.add_axes([0.87, 0.18, 0.12, 0.74])
    ax_c = fig.add_axes([0.03, 0.03, 0.42, 0.12])
    for ax in (ax_f, ax_b):
        ax.axis("off")
    im = ax_f.imshow(frames[0])
    sc = ax_b.scatter(xy[order, 0], xy[order, 1], s=3, c="k", linewidths=0)
    ax_b.set_aspect("equal")
    ax_b.set_title(f"{circuit.n} real neurons from {circuit.dataset}", color="w", fontsize=9)
    banner = ax_f.text(0.5, 0.95, "", transform=ax_f.transAxes, ha="center", va="top",
                       fontsize=20, weight="bold", color="#ffd54f")
    clock = ax_f.text(0.01, 0.01, "", transform=ax_f.transAxes, color="w", fontsize=9)
    ax_g.set_facecolor("#0d1117")
    bars = ax_g.barh(range(len(groups)), np.zeros(len(groups)),
                     color=[ROLE_COLORS[g] for g in groups])
    ax_g.set_yticks(range(len(groups)), [LABELS[g] for g in groups], color="w", fontsize=7)
    ax_g.set_xlim(0, 0.6)
    ax_g.tick_params(axis="x", colors="#8b949e", labelsize=6)
    ax_g.set_title("mean activity", color="w", fontsize=8)
    ax_c.set_facecolor("#0d1117")
    cbars = ax_c.bar(range(len(motor.COMMANDS)), np.zeros(len(motor.COMMANDS)),
                     color="#f06292")
    ax_c.set_xticks(range(len(motor.COMMANDS)), motor.COMMANDS, color="w", fontsize=7)
    ax_c.set_ylim(-1, 1)
    ax_c.tick_params(axis="y", colors="#8b949e", labelsize=6)
    ax_c.set_title("descending-neuron commands to the body", color="w", fontsize=8)

    flash, flash_text = 0, ""
    red_pts = 0.0
    writer = iio.get_writer(out, fps=fps, codec="libx264", quality=7)
    stills.mkdir(parents=True, exist_ok=True)
    saved = 0
    for t in range(len(frames)):
        new_hit = False
        for e in events[t]:
            if e[0] == "throw" and e[1] == "red":
                flash, flash_text = 12, STRIKE_NAMES[e[2]]
            if e[0] == "hit" and e[1] == "red":
                flash, flash_text, new_hit = 20, STRIKE_NAMES[e[2]] + " LANDS!", True
                red_pts += 1
            if e[0] == "knockdown" and e[1] == "red":
                flash, flash_text = 40, "KNOCKDOWN!"
        r = rates[t].astype(np.float32)
        glow = np.clip(r / 0.5, 0, 1) ** 0.7
        sc.set_facecolors((base * (0.12 + 0.88 * glow[:, None]))[order])
        sc.set_sizes((2 + 30 * glow ** 2)[order])
        im.set_data(frames[t])
        banner.set_text(flash_text if flash > 0 else "")
        clock.set_text(f"t = {t * 0.01:.2f} s  (shown at {fps / 100:.1f}x speed)  "
                       f"red hits landed: {int(red_pts)}")
        for bar, m in zip(bars, masks):
            bar.set_width(float(r[m].mean()) if m.any() else 0.0)
        for bar, v in zip(cbars, cmds[t]):
            bar.set_height(v)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
        if new_hit and saved < 4:
            iio.imwrite(stills / f"bout_hit_{saved}.png", img)
            saved += 1
        flash = max(0, flash - 1)
    writer.close()
    iio.imwrite(stills / "bout_last.png", img)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="data/connectome/circuit.npz")
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (omit = untrained brain)")
    ap.add_argument("--stage", type=int, default=2,
                    help="0 bag, 1 mover, 2 sparring; -1 = the checkpoint's training stage")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of greedy")
    ap.add_argument("--out", default="out/bout.mp4")
    ap.add_argument("--stills", default="docs/img")
    args = ap.parse_args()
    circuit = Circuit.load(args.circuit)
    if args.stage < 0:  # fight whatever the checkpoint is training against
        args.stage = int(torch.load(args.ckpt, map_location="cpu")["stage"]) if args.ckpt else 0
    frames, rates, cmds, events, score = record_bout(
        circuit, args.ckpt, args.stage, args.seconds, args.seed, args.sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_video(circuit, frames, rates, cmds, events, score, out, Path(args.stills))
    throws = sum(e[0] == "throw" and e[1] == "red" for ev in events for e in ev)
    print(f"saved {out} ({len(frames)} ticks) | red threw {throws} strikes | "
          f"score red {score['red'].points:.1f} : blue {score['blue'].points:.1f}")


if __name__ == "__main__":
    main()

"""Fight the trained fly yourself, in first person.

You are the blue fly. The red fly is driven live by the trained connectome brain,
and the panel on the right shows its 6,170 real neurons firing as it fights you.

Usage:
    python -m flymt.play                  # hard: the final trained brain
    python -m flymt.play --level easy     # easy: update 100, medium: update 500
    python -m flymt.play --speed 0.2      # "fly time": fraction of real speed
    python -m flymt.play --demo           # no window: a scripted player records out/videos/play_demo.mp4

Controls:
    W / S       walk forward / back          A / D     turn left / right
    J / K       left / right jab             U / I     left / right kick
    L           lunge (knee)
    Space       guard (hold)                 B         boxing stance (hold)
    C           clinch (hold)
    Tab         first person / over the shoulder
    + / -       faster / slower              G  shadows (slower)
    P  pause    R  new round    Esc  quit
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from flymt import motor, opponents
from flymt.brain import Circuit, TorchBrain, to_command
from flymt.connectome import ROLE_COLORS, display_positions
from flymt.fight_env import FightEnv

# Shipped fighters (models/, written by flymt.export_models); every level is the
# same connectome brain at a different point in training.
LEVELS = {"easy": "models/fighter_easy.pt", "medium": "models/fighter_medium.pt",
          "hard": "models/fighter_hard.pt"}
CIRCUIT = "models/circuit.npz"
STRIKE_NAMES = {"jab_l": "LEFT JAB", "jab_r": "RIGHT JAB", "kick_l": "LEFT KICK",
                "kick_r": "RIGHT KICK", "lunge": "LUNGE"}
WIN_W, WIN_H = 1280, 720
VIEW_W = 900  # game view on the left, brain panel on the right
FPS = 30
_C = {k: i for i, k in enumerate(motor.COMMANDS)}


class Game:
    def __init__(self, level: str, ckpt: str | None, speed: float, round_s: float,
                 view: str, seed: int, headless: bool):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("fly-muaythai: you are blue")
        self.font = pygame.font.SysFont("helvetica", 18)
        self.big = pygame.font.SysFont("helvetica", 40, bold=True)
        self.small = pygame.font.SysFont("helvetica", 14)
        self.tiny = pygame.font.SysFont("helvetica", 12)

        torch.set_num_threads(2)
        self.env = FightEnv(seed=seed)
        self.env.round_seconds = round_s
        m = self.env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = VIEW_W, WIN_H
        self.renderer = mujoco.Renderer(m, WIN_H, VIEW_W)
        circuit = Circuit.load(CIRCUIT)
        self.brain = TorchBrain(circuit)
        path = ckpt or LEVELS[level]
        state = torch.load(path, map_location="cpu")
        # Shipped fighters hold only learned parameters; the fixed wiring (buffers)
        # is rebuilt from the circuit. Anything else missing is an error.
        missing, unexpected = self.brain.load_state_dict(state["brain"], strict=False)
        buffers = {n for n, _ in self.brain.named_buffers()}
        assert not unexpected and set(missing) <= buffers, (missing, unexpected)
        self.level = f"{level} (update {state['update']})" if not ckpt else Path(ckpt).name
        self.speed, self.view, self.paused = speed, view, False
        self.shadows = False  # shadows + reflections cost ~70 ms a frame; G toggles
        self.frame = 0
        self._brain_setup(circuit)
        self.reset()

    # ------------------------------------------------------------- set-up
    def _brain_setup(self, circuit: Circuit):
        """Precompute where each neuron is drawn and its role color."""
        xy = display_positions(circuit)
        ok = np.isfinite(xy).all(1)
        lo, hi = np.nanmin(xy[ok], 0), np.nanmax(xy[ok], 0)
        pw, ph, pad = WIN_W - VIEW_W, 470, 20
        s = min((pw - 2 * pad) / (hi[0] - lo[0]), (ph - 2 * pad) / (hi[1] - lo[1]))
        px = VIEW_W + pad + (xy[:, 0] - lo[0]) * s
        py = 60 + pad + (hi[1] - xy[:, 1]) * s
        self.n_xy = np.stack([px, py], 1)
        self.n_rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                               for c in (ROLE_COLORS[r] for r in circuit.role)], float)
        order = np.argsort(circuit.role != "inter", kind="stable")  # interneurons underneath
        self.n_order = order[ok[order]]

    def reset(self):
        self.obs = self.env.reset()
        self.h = self.brain.init_hidden(1)
        self.red_cmd = np.zeros(len(motor.COMMANDS))
        self.callouts: list[list] = []  # [text, color, frames left]
        self.over = False
        self.tick_acc = 0.0

    # --------------------------------------------------------------- input
    def player_command(self, keys) -> np.ndarray:
        pg = self.pg
        c = np.zeros(len(motor.COMMANDS))
        c[_C["forward"]] = 1.0 * keys[pg.K_w] - 0.5 * keys[pg.K_s]
        c[_C["turn"]] = 1.0 * keys[pg.K_a] - 1.0 * keys[pg.K_d]
        for key, name in ((pg.K_j, "jab_l"), (pg.K_k, "jab_r"), (pg.K_u, "kick_l"),
                          (pg.K_i, "kick_r"), (pg.K_l, "lunge"), (pg.K_SPACE, "guard"),
                          (pg.K_b, "box"), (pg.K_c, "clinch")):
            c[_C[name]] = float(keys[key])
        return c

    # ---------------------------------------------------------- simulation
    def tick(self, blue_cmd: np.ndarray):
        with torch.no_grad():
            action, _, self.h = self.brain.act(torch.from_numpy(self.obs["red"])[None], self.h)
        self.red_cmd = to_command(action.numpy())[0]
        self.obs, _, done, events = self.env.step({"red": self.red_cmd, "blue": blue_cmd})
        for e in events:
            who = "YOU" if e[1] == "blue" else "FLY"
            color = (79, 195, 247) if e[1] == "blue" else (239, 83, 80)
            if e[0] == "hit":
                self.callouts.append([f"{who}: {STRIKE_NAMES[e[2]]} LANDS", color, 45])
            elif e[0] == "knockdown":
                self.callouts.append([f"KNOCKDOWN! ({who})", (255, 213, 79), 90])
            elif e[0] == "block":
                self.callouts.append([f"{who}: BLOCKED", color, 30])
            elif e[0] == "clinch":
                self.callouts.append([f"{who}: CLINCH", color, 40])
        if done:
            self.over = True

    # ------------------------------------------------------------- drawing
    def _camera(self):
        d, blue = self.env.data, self.env.idx["blue"].thorax_body
        if self.view == "fpv":
            return "blue/fpv"
        cam = mujoco.MjvCamera()
        R = d.xmat[blue].reshape(3, 3)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        cam.lookat[:] = d.xpos[blue] + R[:, 0] * 0.12
        cam.distance, cam.azimuth, cam.elevation = 0.55, yaw, -22
        return cam

    def draw(self):
        pg = self.pg
        self.renderer.update_scene(self.env.data, self._camera())
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            self.renderer.scene.flags[flag] = self.shadows
        img = self.renderer.render()
        self.screen.blit(pg.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        if self.frame % 2 == 0:  # the brain panel refreshes at 15 fps
            self._draw_brain()
        self.frame += 1
        self._draw_hud()
        pg.display.flip()

    def _draw_brain(self):
        pg = self.pg
        pg.draw.rect(self.screen, (13, 17, 23), (VIEW_W, 0, WIN_W - VIEW_W, WIN_H))
        self.screen.blit(self.font.render("The fly's brain: 6,170 real neurons", True,
                                          (230, 230, 230)), (VIEW_W + 16, 14))
        self.screen.blit(self.small.render("from the male Drosophila connectome", True,
                                           (140, 150, 160)), (VIEW_W + 16, 38))
        rates = self.h[0].numpy()
        glow = np.clip(rates / 0.5, 0, 1) ** 0.7
        col = self.n_rgb * (0.15 + 0.85 * glow[:, None])
        for i in self.n_order:
            g = glow[i]
            pg.draw.circle(self.screen, col[i], self.n_xy[i], 1 + 3.2 * g * g)
        # The fly's current commands to its body.
        y0 = 560
        self.screen.blit(self.font.render("Its descending-neuron commands", True,
                                          (230, 230, 230)), (VIEW_W + 16, y0))
        names = ("walk", "turn", "jab L", "jab R", "kick L", "kick R", "lunge", "clinch",
                 "guard", "box")
        bw = (WIN_W - VIEW_W - 32) / len(names)
        for k, (nm, v) in enumerate(zip(names, self.red_cmd)):
            x = VIEW_W + 16 + k * bw
            hgt = float(np.clip(abs(v), 0, 1)) * 60
            pg.draw.rect(self.screen, (240, 98, 146), (x + 3, y0 + 100 - hgt, bw - 6, hgt))
            lab = self.tiny.render(nm, True, (170, 180, 190))
            self.screen.blit(lab, (x + (bw - lab.get_width()) / 2, y0 + 106 + (k % 2) * 14))

    def _draw_hud(self):
        pg = self.pg
        sc = self.env.score
        bar = pg.Surface((VIEW_W, 58), pg.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        self.screen.blit(bar, (0, 0))
        self.screen.blit(self.font.render(
            f"YOU (blue) {sc['blue'].points:5.1f}   :   {sc['red'].points:5.1f} FLY (red)",
            True, (255, 255, 255)), (16, 12))
        self.screen.blit(self.small.render(
            f"{self.env.t:4.1f} / {self.env.round_seconds:.0f} s   |   speed {self.speed:.2f}x   |   "
            f"{'first person' if self.view == 'fpv' else 'over the shoulder'}   |   "
            f"opponent: {self.level}", True, (200, 210, 220)), (16, 36))
        for k, c in enumerate(self.callouts[-3:]):
            txt = self.big.render(c[0], True, c[1])
            self.screen.blit(txt, ((VIEW_W - txt.get_width()) // 2, 80 + k * 48))
            c[2] -= 1
        self.callouts = [c for c in self.callouts if c[2] > 0]
        help_ = ("W/S walk  A/D turn  J/K jab  U/I kick  L lunge  Space guard  B box  "
                 "C clinch  Tab view  +/- speed  G shadows  P pause  R restart  Esc quit")
        self.screen.blit(self.small.render(help_, True, (230, 230, 230)), (12, WIN_H - 22))
        if self.over or self.paused:
            msg = "PAUSED" if self.paused else (
                "YOU WIN!" if sc["blue"].points > sc["red"].points else
                "THE FLY WINS" if sc["red"].points > sc["blue"].points else "DRAW")
            txt = self.big.render(msg + ("" if self.paused else "   (R = rematch)"), True,
                                  (255, 213, 79))
            self.screen.blit(txt, ((VIEW_W - txt.get_width()) // 2, WIN_H // 2 - 20))

    # ---------------------------------------------------------------- loop
    def advance(self, blue_cmd: np.ndarray):
        """Run the simulation forward for one frame of wall time at the current speed."""
        if self.over or self.paused:
            return
        self.tick_acc += self.speed * 100 / FPS  # 100 ticks = 1 s of fly time
        for _ in range(min(int(self.tick_acc), 5)):
            self.tick(blue_cmd)
            if self.over:
                break
        self.tick_acc -= int(self.tick_acc)

    def run(self):
        pg = self.pg
        clock = pg.time.Clock()
        while True:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                    pg.quit()
                    return
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_TAB:
                        self.view = "third" if self.view == "fpv" else "fpv"
                    elif ev.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                        self.speed = min(1.0, round(self.speed + 0.05, 2))
                    elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        self.speed = max(0.05, round(self.speed - 0.05, 2))
                    elif ev.key == pg.K_p:
                        self.paused = not self.paused
                    elif ev.key == pg.K_g:
                        self.shadows = not self.shadows
                    elif ev.key == pg.K_r:
                        self.reset()
            self.advance(self.player_command(pg.key.get_pressed()))
            self.draw()
            clock.tick(FPS)

    def demo(self, out: Path, seconds: float, seed: int):
        """Headless self-test: a scripted Veteran plays blue; record what a player sees."""
        import imageio.v2 as iio
        player = opponents.Veteran(np.random.default_rng(seed))
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=1)
        frames = int(seconds / self.speed * FPS)
        for f in range(frames):
            if f == frames // 2:
                self.view = "third" if self.view == "fpv" else "fpv"
            self.advance(player.act(self.obs["blue"]))
            self.draw()
            writer.append_data(self.pg.surfarray.array3d(self.screen).swapaxes(0, 1))
            if self.over:
                break
        writer.close()
        sc = self.env.score
        print(f"saved {out} ({f + 1} frames) | scripted player {sc['blue'].points:.1f} : "
              f"{sc['red'].points:.1f} trained fly")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=LEVELS, default="hard")
    ap.add_argument("--ckpt", default=None, help="any checkpoint instead of a level")
    ap.add_argument("--speed", type=float, default=0.25, help="fraction of real fly speed")
    ap.add_argument("--round", type=float, default=20.0, help="round length, fly seconds")
    ap.add_argument("--view", choices=("fpv", "third"), default="fpv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="headless recorded self-test")
    ap.add_argument("--demo-seconds", type=float, default=4.0)
    ap.add_argument("--out", default="out/videos/play_demo.mp4")
    args = ap.parse_args()
    game = Game(args.level, args.ckpt, args.speed, args.round, args.view, args.seed,
                headless=args.demo)
    if args.demo:
        game.demo(Path(args.out), args.demo_seconds, args.seed)
    else:
        game.run()


if __name__ == "__main__":
    main()

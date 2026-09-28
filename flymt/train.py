"""Train the connectome brain to fight with evolution strategies (OpenAI-ES).

The brain always fights in the red corner. Opponents follow a curriculum:
    0 bag       stands still (heavy bag)
    1 mover     wanders and turns
    2 sparring  scripted fighter that closes in and strikes back
    3 selfplay  earlier versions of the brain (hall of fame)

Usage:
    python -m flymt.train --circuit data/connectome/circuit.npz
"""

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from flymt import motor
from flymt.brain import Brain, Circuit, Layout
from flymt.fight_env import OBS_FIELDS, FightEnv

STAGES = ("bag", "mover", "sparring", "selfplay")
# Promote to the next stage when the mean fitness reaches this.
PROMOTE_AT = {0: 4.0, 1: 4.0, 2: 3.0}
EPISODE_SECONDS = {0: 3.0, 1: 3.0, 2: 4.0, 3: 6.0}
CKPT_DIR = Path("checkpoints")

_F = {k: i for i, k in enumerate(OBS_FIELDS)}
_C = {k: i for i, k in enumerate(motor.COMMANDS)}


class Opponent:
    """Heavy bag: does nothing."""

    def reset(self, rng: np.random.Generator):
        pass

    def act(self, obs: np.ndarray, t: float) -> np.ndarray:
        return np.zeros(len(motor.COMMANDS))


class Mover(Opponent):
    def reset(self, rng):
        self.rng = rng
        self.cmd = np.zeros(len(motor.COMMANDS))

    def act(self, obs, t):
        if self.rng.random() < 0.02:  # new heading every ~0.5 s
            self.cmd[_C["forward"]] = self.rng.uniform(-0.3, 0.8)
            self.cmd[_C["turn"]] = self.rng.uniform(-0.8, 0.8)
        if obs[_F["rope_dist"]] < 0.15:  # steer off the ropes
            self.cmd[_C["turn"]] = 1.0
        return self.cmd


class Sparring(Opponent):
    """Closes the distance, then throws a random strike every ~0.25 s."""

    def reset(self, rng):
        self.rng = rng

    def act(self, obs, t):
        c = np.zeros(len(motor.COMMANDS))
        bearing = np.arctan2(obs[_F["opp_bearing_sin"]], obs[_F["opp_bearing_cos"]])
        c[_C["turn"]] = np.clip(2 * bearing, -1, 1)
        if obs[_F["opp_dist"]] > 0.2:
            c[_C["forward"]] = 0.8 if abs(bearing) < 0.5 else 0.2
        elif self.rng.random() < 0.04:
            c[_C[self.rng.choice(motor.STRIKES)]] = 1.0
        if any(obs[_F[f"opp_{s}"]] for s in motor.STRIKES) and self.rng.random() < 0.3:
            c[_C["guard"]] = 1.0
        return c


class BrainOpponent(Opponent):
    def __init__(self, brain: Brain):
        self.brain = brain

    def reset(self, rng):
        self.brain.reset()

    def act(self, obs, t):
        return self.brain.step(obs)


# ------------------------------------------------------------------ workers
_W: dict = {}


def _init_worker(circuit_path: str):
    circuit = Circuit.load(circuit_path)
    _W["circuit"] = circuit
    _W["layout"] = Layout(circuit)
    _W["env"] = FightEnv()


def evaluate(params: np.ndarray, seed: int, stage: int,
             opp_params: np.ndarray | None = None) -> tuple[float, float, float]:
    """One bout. Returns (fitness, points landed, points taken) for red."""
    env: FightEnv = _W["env"]
    env.rng = np.random.default_rng(seed)
    brain = Brain(_W["circuit"], _W["layout"], params)
    if stage == 0:
        opp = Opponent()
    elif stage == 1:
        opp = Mover()
    elif stage == 2:
        opp = Sparring()
    else:
        opp = BrainOpponent(Brain(_W["circuit"], _W["layout"], opp_params))
    opp.reset(np.random.default_rng(seed + 1))
    obs = env.reset()
    fitness, done = 0.0, False
    while not done and env.t < EPISODE_SECONDS[stage]:
        cmds = {"red": brain.step(obs["red"]), "blue": opp.act(obs["blue"], env.t)}
        obs, rewards, done, _ = env.step(cmds)
        fitness += rewards["red"]
    return fitness, env.score["red"].points, env.score["blue"].points


def _eval_task(args):
    return evaluate(*args)


# ----------------------------------------------------------------------- ES
def centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[x.argsort()] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


class Adam:
    def __init__(self, size, lr, b1=0.9, b2=0.999):
        self.m, self.v, self.t = np.zeros(size), np.zeros(size), 0
        self.lr, self.b1, self.b2 = lr, b1, b2

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(args):
    circuit = Circuit.load(args.circuit)
    layout = Layout(circuit)
    rng = np.random.default_rng(args.seed)
    ckpt = Path(args.out)
    ckpt.mkdir(parents=True, exist_ok=True)
    latest = ckpt / "latest.npz"
    if args.resume and latest.exists():
        ck = np.load(latest)
        theta, gen0, stage = ck["params"], int(ck["gen"]) + 1, int(ck["stage"])
        print(f"resumed gen {gen0} stage {STAGES[stage]}")
    else:
        theta, gen0, stage = layout.init(rng), 0, args.stage
    print(f"circuit: {circuit.n} neurons, {circuit.W.nnz} connections, "
          f"{len(layout.types)} cell types, {layout.n_dn} DN groups -> {layout.size} params")
    hall = [theta.copy()]
    opt = Adam(layout.size, args.lr)
    log_path = ckpt / "log.csv"
    new_log = not (args.resume and log_path.exists())
    log = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["gen", "stage", "mean_fitness", "best_fitness",
                         "mean_hits", "mean_taken", "wall_s"])

    with mp.get_context("spawn").Pool(args.workers, _init_worker, (args.circuit,)) as pool:
        for gen in range(gen0, gen0 + args.generations):
            t0 = time.time()
            eps = rng.standard_normal((args.pairs, layout.size))
            seeds = rng.integers(1 << 30, size=(args.pairs, args.bouts))
            tasks = []
            for i in range(args.pairs):
                for sign in (1, -1):
                    p = theta + sign * args.sigma * eps[i]
                    for b in range(args.bouts):
                        # Both halves of a pair see the same bout and opponent.
                        opp = hall[seeds[i, b] % len(hall)] if stage == 3 else None
                        tasks.append((p, int(seeds[i, b]), stage, opp))
            res = np.array(pool.map(_eval_task, tasks)).reshape(args.pairs, 2, args.bouts, 3)
            fit = res[..., 0].mean(-1)  # (pairs, 2)
            ranks = centered_ranks(fit.ravel()).reshape(fit.shape)
            grad = ((ranks[:, 0] - ranks[:, 1]) @ eps) / (2 * args.pairs * args.sigma)
            theta = theta + opt.step(grad - args.decay * theta)

            mean_fit = float(fit.mean())
            row = [gen, stage, round(mean_fit, 3), round(float(fit.max()), 3),
                   round(float(res[..., 1].mean()), 3), round(float(res[..., 2].mean()), 3),
                   round(time.time() - t0, 1)]
            writer.writerow(row)
            log.flush()
            print(f"gen {gen:4d} [{STAGES[stage]:8s}] fitness {mean_fit:+7.2f} "
                  f"(best {row[3]:+.2f}) landed {row[4]:.2f} taken {row[5]:.2f} "
                  f"{row[6]}s", flush=True)

            ck = dict(params=theta, gen=gen, stage=stage, fitness=mean_fit)
            np.savez(latest, **ck)
            if gen % args.save_every == 0:
                np.savez(ckpt / f"gen_{gen:04d}.npz", **ck)
            if stage in PROMOTE_AT and mean_fit >= PROMOTE_AT[stage]:
                np.savez(ckpt / f"graduated_{STAGES[stage]}.npz", **ck)
                stage += 1
                print(f"==> promoted to stage {STAGES[stage]}")
            if stage == 3 and gen % 10 == 0:
                hall.append(theta.copy())
    log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="data/connectome/circuit.npz")
    ap.add_argument("--out", default=str(CKPT_DIR))
    ap.add_argument("--generations", type=int, default=500)
    ap.add_argument("--pairs", type=int, default=12, help="antithetic pairs per generation")
    ap.add_argument("--bouts", type=int, default=1, help="bouts per candidate")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--decay", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--stage", type=int, default=0, choices=range(len(STAGES)))
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()

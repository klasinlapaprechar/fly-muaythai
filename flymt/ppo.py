"""Recurrent PPO: the connectome brain learns to fight from reward alone.

Fights run in parallel CPU worker processes (MuJoCo physics). The brain picks
actions on CPU during rollouts; the PPO update backpropagates through the
brain's recurrent dynamics on the Mac GPU (MPS) when available.

Usage:
    python -m flymt.ppo                       # train from scratch
    python -m flymt.ppo --resume              # continue from checkpoints/latest.pt
"""

import argparse
import csv
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flymt import motor, opponents
from flymt.brain import Circuit, TorchBrain
from flymt.fight_env import OBS_FIELDS, FightEnv

EPISODE_SECONDS = {0: 3.0, 1: 3.0, 2: 4.0, 3: 6.0}
# Promote when the mean net score (points landed - taken) per bout reaches this.
PROMOTE_AT = {0: 3.0, 1: 3.0, 2: 2.0}


# --------------------------------------------------------------- fight workers
def _worker(conn, n_envs: int, seed: int):
    envs = [FightEnv(seed=seed + k) for k in range(n_envs)]
    rng = np.random.default_rng(seed)
    stage = 0
    opps = [None] * n_envs
    stats = [None] * n_envs
    obs = [None] * n_envs

    def reset(k):
        opps[k] = opponents.make(stage, rng)
        stats[k] = {"ret": 0.0, "throws": 0}
        return envs[k].reset()

    while True:
        msg, payload = conn.recv()
        if msg == "stage":
            stage = payload
            obs = [reset(k) for k in range(n_envs)]
            conn.send((np.stack([o["red"] for o in obs]), np.stack([o["blue"] for o in obs])))
        elif msg == "step":
            red_cmds, blue_cmds = payload
            out_red, out_blue, rews, dones, finished = [], [], [], [], []
            for k, env in enumerate(envs):
                blue = opps[k].act(obs[k]["blue"]) if opps[k] else blue_cmds[k]
                o, r, done, ev = env.step({"red": red_cmds[k], "blue": blue})
                stats[k]["ret"] += r["red"]
                stats[k]["throws"] += sum(e[0] == "throw" and e[1] == "red" for e in ev)
                done = done or env.t >= EPISODE_SECONDS[stage]
                if done:
                    finished.append({**stats[k], "landed": env.score["red"].points,
                                     "taken": env.score["blue"].points})
                    o = reset(k)
                obs[k] = o
                out_red.append(o["red"])
                out_blue.append(o["blue"])
                rews.append(r["red"])
                dones.append(done)
            conn.send((np.stack(out_red), np.stack(out_blue), np.array(rews, np.float32),
                       np.array(dones), finished))
        elif msg == "close":
            return


class VecFights:
    """Parallel bouts: `n_workers` processes, each running `envs_per_worker` fights."""

    def __init__(self, n_workers: int, envs_per_worker: int, seed: int):
        ctx = mp.get_context("spawn")
        self.conns = []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            ctx.Process(target=_worker, args=(b, envs_per_worker, seed + 1000 * w),
                        daemon=True).start()
            self.conns.append(a)
        self.k = envs_per_worker
        self.n = n_workers * envs_per_worker

    def set_stage(self, stage: int):
        for c in self.conns:
            c.send(("stage", stage))
        res = [c.recv() for c in self.conns]
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])

    def step(self, red: np.ndarray, blue: np.ndarray | None):
        for i, c in enumerate(self.conns):
            sl = slice(i * self.k, (i + 1) * self.k)
            c.send(("step", (red[sl], None if blue is None else blue[sl])))
        res = [c.recv() for c in self.conns]
        cat = lambda j: np.concatenate([r[j] for r in res])  # noqa: E731
        return cat(0), cat(1), cat(2), cat(3), [f for r in res for f in r[4]]

    def close(self):
        for c in self.conns:
            c.send(("close", None))


# ---------------------------------------------------------------------- critic
class Critic(nn.Module):
    """Value function for PPO. Not part of the fly; it only helps training."""

    def __init__(self, obs_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def _cpu_state(m: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


# ------------------------------------------------------------------------- PPO
def train(args):
    dev = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    circuit = Circuit.load(args.circuit)
    brain = TorchBrain(circuit).to(dev)  # learner
    actor = TorchBrain(circuit)  # rollout copy on CPU
    obs_size = len(OBS_FIELDS)
    critic = Critic(obs_size).to(dev)
    params = list(brain.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage, update0, steps = args.stage, 0, 0
    if args.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev)
        brain.load_state_dict(ck["brain"])
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        stage, update0, steps = ck["stage"], ck["update"] + 1, ck["steps"]
        print(f"resumed at update {update0}, stage {opponents.STAGES[stage]}")
    print(f"device {dev} | circuit: {circuit.n} real neurons, {circuit.W.nnz} connections, "
          f"{len(brain.types)} cell types, {len(brain.dn_keys)} DN groups | "
          f"{sum(p.numel() for p in brain.parameters())} trainable brain parameters "
          f"(wiring fixed)", flush=True)

    vec = VecFights(args.workers, args.envs_per_worker, args.seed)
    N, T, L = vec.n, args.rollout, args.chunk
    assert T % L == 0
    hall: list[dict] = []  # past selves for self-play
    opp_brain = TorchBrain(circuit)

    def start_stage():
        nonlocal opp_h
        if stage == 3 and not hall:
            hall.append(_cpu_state(brain))
        if stage == 3:
            opp_brain.load_state_dict(hall[np.random.randint(len(hall))])
        opp_h = opp_brain.init_hidden(N)
        return vec.set_stage(stage)

    opp_h = None
    obs, blue_obs = start_stage()
    h = actor.init_hidden(N)
    actor.load_state_dict(_cpu_state(brain))
    recent = deque(maxlen=args.promote_window)
    log_path = out / "log.csv"
    new_log = not (args.resume and log_path.exists())
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["update", "stage", "steps", "ep_return", "landed", "taken",
                      "throws", "entropy", "value_loss", "sps"])

    for update in range(update0, update0 + args.updates):
        t0 = time.time()
        buf_obs = np.zeros((T, N, obs_size), np.float32)
        buf_act = np.zeros((T, N, len(motor.COMMANDS)), np.float32)
        buf_logp = np.zeros((T, N), np.float32)
        buf_rew = np.zeros((T, N), np.float32)
        buf_done = np.zeros((T, N), np.float32)  # bout ended after step t
        buf_h0 = np.zeros((T // L, N, circuit.n), np.float32)  # brain state at chunk starts
        finished = []
        for t in range(T):
            if t % L == 0:
                buf_h0[t // L] = h.numpy()
            act, logp, h = actor.act(torch.from_numpy(obs), h)
            blue = None
            if stage == 3:
                blue_t, _, opp_h = opp_brain.act(torch.from_numpy(blue_obs), opp_h)
                blue = blue_t.numpy()
            nobs, nblue, rew, done, fin = vec.step(act.numpy(), blue)
            buf_obs[t], buf_act[t], buf_logp[t] = obs, act.numpy(), logp.numpy()
            buf_rew[t], buf_done[t] = rew, done
            fresh = 1 - torch.from_numpy(done.astype(np.float32))[:, None]
            h, opp_h = h * fresh, opp_h * fresh  # new bout, fresh brain state
            obs, blue_obs = nobs, nblue
            finished += fin
        steps += T * N

        # Advantages (GAE). Time-limit ends are treated as terminal.
        with torch.no_grad():
            v = critic(torch.from_numpy(buf_obs).to(dev)).cpu().numpy()
            v_last = critic(torch.from_numpy(obs).to(dev)).cpu().numpy()
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else v[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * nv * nonterm - v[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + v

        # PPO epochs over (chunk x env) sequences, backprop through the brain.
        n_seq = (T // L) * N

        def seq(x):  # (T, N, ...) -> (chunks*N, L, ...)
            x = x.reshape(T // L, L, N, *x.shape[2:]).swapaxes(1, 2)
            return torch.from_numpy(np.ascontiguousarray(x.reshape(n_seq, L, *x.shape[3:]))).to(dev)

        S_obs, S_act, S_logp = seq(buf_obs), seq(buf_act), seq(buf_logp)
        S_adv, S_ret, S_done = seq(adv), seq(ret), seq(buf_done)
        S_h0 = torch.from_numpy(buf_h0.reshape(n_seq, circuit.n)).to(dev)
        ent_sum = vl_sum = 0.0
        n_mb = 0
        for _ in range(args.epochs):
            for mb in torch.randperm(n_seq, device=dev).split(args.minibatch):
                hh = S_h0[mb]
                logps, ents = [], []
                for t in range(L):
                    out_t, hh = brain(S_obs[mb, t], hh)
                    logps.append(brain.log_prob(out_t, S_act[mb, t]))
                    ents.append(brain.entropy(out_t))
                    hh = hh * (1 - S_done[mb, t])[:, None]
                logp_new = torch.stack(logps, 1)
                ent = torch.stack(ents, 1).mean()
                a = S_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp_new - S_logp[mb])
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = ((critic(S_obs[mb]) - S_ret[mb]) ** 2).mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad)
                opt.step()
                ent_sum += ent.item()
                vl_sum += vl.item()
                n_mb += 1
        actor.load_state_dict(_cpu_state(brain))

        # Logging, curriculum, checkpoints.
        recent.extend(finished)
        mean = lambda k: float(np.mean([f[k] for f in finished])) if finished else float("nan")  # noqa: E731
        sps = T * N / (time.time() - t0)
        row = [update, stage, steps, mean("ret"), mean("landed"), mean("taken"),
               mean("throws"), ent_sum / n_mb, vl_sum / n_mb, round(sps)]
        log.writerow([round(x, 3) if isinstance(x, float) else x for x in row])
        logf.flush()
        print(f"upd {update:4d} [{opponents.STAGES[stage]:8s}] steps {steps:8d} | "
              f"return {row[3]:+6.2f} landed {row[4]:4.2f} taken {row[5]:4.2f} "
              f"throws {row[6]:4.1f} | ent {row[7]:.2f} vloss {row[8]:.3f} | {sps:.0f} steps/s",
              flush=True)
        ck = dict(brain=brain.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(),
                  stage=stage, update=update, steps=steps)
        torch.save(ck, out / "latest.pt")
        if update % args.save_every == 0:
            torch.save(ck, out / f"update_{update:04d}.pt")

        net = [f["landed"] - f["taken"] for f in recent]
        if (stage in PROMOTE_AT and len(recent) == recent.maxlen
                and np.mean(net) >= PROMOTE_AT[stage]):
            torch.save(ck, out / f"graduated_{opponents.STAGES[stage]}.pt")
            stage += 1
            recent.clear()
            print(f"==> promoted to {opponents.STAGES[stage]}", flush=True)
            obs, blue_obs = start_stage()
            h = actor.init_hidden(N)
        elif stage == 3 and update % 20 == 0:
            hall.append(_cpu_state(brain))
            opp_brain.load_state_dict(hall[np.random.randint(len(hall))])
    vec.close()
    logf.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="data/connectome/circuit.npz")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=256, help="ticks per env per update")
    ap.add_argument("--chunk", type=int, default=32, help="BPTT length (ticks)")
    ap.add_argument("--minibatch", type=int, default=48, help="sequences per minibatch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--promote-window", type=int, default=48)
    ap.add_argument("--stage", type=int, default=0, choices=range(4))
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="train on CPU instead of MPS")
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()

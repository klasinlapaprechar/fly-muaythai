# fly-muaythai

A fruit fly brain, wired from the male *Drosophila* connectome, learns Muay Thai
inside a MuJoCo physics simulation, and you can watch its neurons light up with
every jab, kick and lunge.

Male flies already fight: they lunge, box with their forelegs, grab and hold
(the clinch), and threaten with their wings. This project gives that behavior a
ring, a referee, and a brain that learns to win.

## How it fits together

```
 sensory neurons         connectome-wired brain            descending neurons       "nerve cord"
 (vision, touch,   --->  (weights & signs fixed by  --->   (command vector)   --->  motor system
  balance)                the male CNS connectome;                                 (gait + strikes)
                          RL tunes per-cell-type gains)                                   |
        ^                                                                                 v
        +------------------------- MuJoCo two-fly ring + referee <------------------------+
```

| Module | What it does |
|---|---|
| `flymt/arena.py` | Ring with ropes, two flybody fruit flies (red and blue corners). |
| `flymt/ik.py` | Per-leg inverse kinematics in the thorax frame. |
| `flymt/motor.py` | The "nerve cord": tripod walking gait plus Muay Thai strikes (jab, roundhouse kick, lunge, clinch, guard) built from IK keyframes. |
| `flymt/fight_env.py` | Bout logic: physics stepping, a referee that scores strikes by contact force, knockdowns, egocentric observations, rewards. |

### Fly Muay Thai

| Muay Thai | Fly version | Points |
|---|---|---|
| Jab / cross | Foreleg (T1) punch | 1 |
| Roundhouse kick | Mid-leg (T2) sweep | 1.5 |
| Knee / elbow | Lunge: rear up and slam down | 2 |
| Clinch | Forelegs grab with claw adhesion | – |
| Knockdown | Opponent on its side/back for 0.15 s | 5 |

## Setup

Requires Python 3.11–3.12 (tested on macOS arm64).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
```

## Status

- [x] Two-fly ring, fly bodies, faster-than-real-time physics
- [x] Motor system: walking, turning, jab, kick, lunge, clinch, guard
- [x] Referee: contact-force strike scoring, knockdowns
- [x] Real fighting circuit from male-cns:v1.0: 6,170 neurons, 370k connections ([image](docs/img/circuit.png))
- [x] Recurrent PPO: CPU physics workers + backprop through the brain on Apple GPU (MPS)
- [ ] Training run: bag -> mover -> sparring -> self-play
- [ ] Neural activity visualization synced to strikes

## Credits

- Fly body: [flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al.)
- Physics: [MuJoCo](https://mujoco.org)
- Connectome: Janelia FlyEM male CNS dataset via [neuPrint](https://neuprint.janelia.org)

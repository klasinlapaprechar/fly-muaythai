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
| `flymt/connectome.py` | Pulls the fighting circuit from the male CNS connectome (neuPrint). |
| `flymt/brain.py` | PyTorch rate model of that circuit; it is the PPO policy. |
| `flymt/opponents.py` | Curriculum sparring partners: heavy bag, mover, sparring fighter. |
| `flymt/ppo.py` | Recurrent PPO: parallel physics workers on CPU, backprop through the brain on the Apple GPU. |
| `flymt/visualize.py` | Video of a bout next to the brain, every neuron glowing with its activity. |

![Fighting circuit](docs/img/circuit.png)

### What is real and what is modeled

| | Source |
|---|---|
| Neurons, cell types, who connects to whom, synapse counts | Real: `male-cns:v1.0` (6,170 neurons, 370k connections, 6.5M synapses) |
| Excitatory vs inhibitory | Real: predicted neurotransmitter (ACh +; GABA, Glu, His −; monoamines modulatory) |
| Synaptic strength scale per cell type, excitability | **Learned** by PPO (12k parameters); the wiring never changes |
| Neuron dynamics | Simplified firing-rate model (τ = 20 ms) |
| Eyes, bristles, antennae | Abstracted into drives for the real sensory neurons (LC10, LPLC2/LC4, BM, JO) |
| Leg control below the descending neurons | Scripted motor system standing in for the nerve cord |

Learning is standard RL: the brain samples actions, the referee's points are the
reward, and PPO updates the brain from that reward alone. PAM dopamine neurons
receive the reward signal so you can watch them flash on every scoring strike;
they do not drive learning.

## Train and watch

```bash
python -m flymt.connectome      # once; needs NEUPRINT_APPLICATION_CREDENTIALS in .env
python -m flymt.ppo             # train (resume with --resume)
python -m flymt.visualize --ckpt checkpoints/latest.pt --stage 2
```

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
- [x] Neural activity visualization synced to strikes

## Credits

- Fly body: [flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al.)
- Physics: [MuJoCo](https://mujoco.org)
- Connectome: Janelia FlyEM male CNS dataset via [neuPrint](https://neuprint.janelia.org)

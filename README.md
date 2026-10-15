# Broaden Your Horizons

Coherence across a bank of discount horizons identifies reusable motor transitions without prior knowledge of
timescale.

The discounted future of a movement is computed at 40 log-spaced horizons (a sampled Laplace transform of the future).
Differences between neighbouring horizons give 39 bands, peaking at future lags from 10 ms to 1.8 s. A transition is a
moment when at least three adjacent bands change direction together (above their 90th percentile). The rule and its
parameters (`horizons.py`) are fixed and used unchanged for every agent, condition and human participant.

Agents are soft actor-critic policies driving MotorNet's two-joint, six-muscle arm through an 8-target sequence.
Transitions found in agent 1's practised movement (subgoals A-E, hand state and muscle activations) become options for a
semi-Markov double DQN selector, which is tested zero-shot on unseen sequences. Human data are from Cuevas Rivera,
Szpilska & Kiebel (see `human_data/README.md`).

## Contents

| Path | |
|---|---|
| `horizons.py` | the multi-timescale bank, bands, coherence rule, single-horizon scores |
| `arm.py`, `control.py` | the arm task; controllers, options, selector network, rollouts and recordings |
| `human.py` | loading and segmenting the human data |
| `train_agent.py`, `train_reacher.py`, `train_selector.py` | training of agents, option reacher and DDQN selectors |
| `record.py`, `subgoals.py`, `evaluate.py` | movement recordings, subgoal discovery and single-horizon tuning, zero-shot evaluation |
| `analyze_*.py`, `make_figures.py` | statistics (`results/*.json`) and figures (`figures/`) |
| `verify.py` | quick reproducibility checks |
| `data/` | panels of unseen orders, subgoals, recorded movements, fixed feature scaling |
| `checkpoints/` | agents 1-3, option reacher, selectors |
| `results/transfer/` | every evaluation trial |

## Setup

```bash
mamba create -n hrl python=3.10
mamba activate hrl
uv pip install -r requirements.txt
```

## Reproduce

```bash
python verify.py                      # ~3 min
python analyze_transitions.py         # ~10 min
python analyze_humans.py              # ~5 min
python analyze_transfer.py
python analyze_muscles.py
python make_figures.py
```

From scratch (slow; saved outputs are included):

```bash
python record.py isolated | continuous | conditions
python subgoals.py agent1 && python subgoals.py tune && python subgoals.py agent3
python evaluate.py --all --workers 6  # ~40 min
python train_agent.py --seed 7 --envs 1 --out runs/agent1           # agent 1 (agents 2, 3: --seed 101 / 202 --envs 4)
python train_reacher.py --seed 7 --out runs/reacher
python train_selector.py --agent 1 --subgoals data/subgoals/agent1.json --seed 1 --steps 1000000 --out runs/agent1_seed1
python train_selector.py --agent 3 --subgoals data/subgoals/agent3_multiscale.json --seed 1 --steps 500000 --random-orders --out runs/agent3_multiscale_seed1
```

Agent 1 performs the practised sequence and provides subgoals A-E; agents 2 and 3 were trained the same way with other
seeds. Agent 3 is the independently trained agent of the transfer test. Agent 1's noise-free isolated-reach recording was
used to develop the method and is not part of the transition statistics. Evaluation runs on one CPU thread (outcomes
depend on floating-point summation order).

## Results

| Finding | Value [95% CI] | Statistics | Figure |
|---|---|---|---|
| Agents: events at movement onset (first third of a reach) | 64% vs chance 31%, 17/17 recordings | `results/transitions.json` | `horizon_across_systems.png` |
| Reach-specific muscle patterns at events | reach decoded 52% from muscles vs 19% from kinematics (chance 19%), p = 0.001 | `results/muscles.json` | `muscle_patterns.png` |
| Zero-shot transfer, 8 targets (96 unseen orders) | 51% -> 82%, +30.9 points [19.3, 45.6] | `results/transfer.json` | `transfer.png` |
| Zero-shot transfer, 16 targets (48 unseen sequences) | 32% -> 72%, +40.0 points [21.0, 62.3] | `results/transfer.json` | `transfer.png` |
| Independently trained agent (96 held-out orders) | frozen 28%, single tuned horizon 32%, multi-timescale 91%; +59.5 points [50.0, 68.3] | `results/transfer.json` | `independent_agent.png` |
| Heavier limbs, rescaled workspace: rule - tuned horizon | +5.5 points [1.6, 9.5] | `results/transitions.json` | `robustness.png` |
| Worst case over 15 conditions (points above chance) | rule +3.9, best of 40 single horizons -1.7 | `results/transitions.json` | `robustness.png` |
| Tempo x0.5 / x1 / x1.8 / x3 | rule 70% / 67% / 63% / 59%, tuned horizon 39% / 64% / 66% / 54% | `results/transitions.json` | `tempo.png` |
| Humans: segment onsets vs best single discount | 54% vs 45% (chance 32%), +8.9 points [4.1, 13.8] | `results/humans.json` | `humans_onsets.png` |
| Humans: events at via-point passages | 61% within 50 ms vs 20% of frames | `results/humans.json` | `humans_transitions.png` |
| Humans: hierarchical vs flat sequences | AUC 0.74 [0.52, 0.90] | `results/humans.json` | `humans_transitions.png` |
| Humans: horizon tuned on the agent | 17% vs chance 32%; rule 54% | `results/transitions.json` | `robustness.png` |

## Figures

| File | |
|---|---|
| `transfer.png` | zero-shot transfer to unseen 8- and 16-target sequences |
| `transfer_detail.png` | frozen agent vs hierarchical controller by start set |
| `completion_time.png` | failure-capped completion time |
| `completion_time_independent.png` | failure-capped completion time, independently trained agent only |
| `subgoal_torque.png` | muscle torques at shoulder and elbow when options reach subgoals A-E |
| `independent_agent.png` | transfer to an independently trained agent |
| `horizon_across_systems.png` | a single horizon tuned in agents or humans fails in the other system |
| `tempo.png` | time-stretched movements |
| `robustness.png` | heavier limbs, rescaled workspace, other agents, tempi, humans |
| `humans_onsets.png` | held-out humans: rule vs best single discount |
| `humans_transitions.png` | events at via-point passages; hierarchical vs flat sequences |
| `muscle_patterns.png` | reach identity decoded from muscle activations at events |

"""Frozen SAC controllers, subgoal options and the DDQN selector; batched rollouts and recordings."""
import json
import zipfile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

import arm

START_SEEDS = (42, 101, 202, 303, 404)   # five jitters of each panel posture
JITTER_DEG = 0.5
JITTER_SALT = 270019
SUBGOAL_RADIUS = 0.02                    # m, an option ends when the hand is this close to its subgoal
OPTION_STEPS = 30                        # an option runs at most 300 ms
STALL_STEPS, STALL_DISTANCE, STALL_PROGRESS = 10, 0.04, 0.001   # re-aim: no 1 mm progress in 100 ms, > 4 cm away
NOISE = 0.1                              # excitation noise of the noisy recordings
NOISE_SEEDS = (42, 101, 202, 303, 404)
ELEMENT_STEPS = 1530                     # recorded future of each movement element (15.3 s)


def load_policy(path, observation_size=30):
    """Load a saved SB3 SAC policy with plain spaces (no unpickling)."""
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    rate = float(data['learning_rate'])
    replacements = {key: None for key, value in data.items() if isinstance(value, dict) and ':serialized:' in value}
    replacements.update({'policy_class': SACPolicy, 'train_freq': (1, 'step'), 'lr_schedule': lambda _: rate,
                         'observation_space': gym.spaces.Box(-np.inf, np.inf, (observation_size,), dtype=np.float32),
                         'action_space': gym.spaces.Box(-1., 1., (6,), dtype=np.float32)})
    return SAC.load(str(path), device='cpu', custom_objects=replacements)


def excitation(policy, obs):
    action, _ = policy.predict(obs, deterministic=True)
    return (np.clip(np.asarray(action, dtype=np.float32), -1, 1) + 1) / 2


class QNetwork(nn.Module):
    def __init__(self, state_dim=30, action_count=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, action_count))

    def forward(self, state):
        return self.net(state)


def load_selector(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    network = QNetwork(int(checkpoint['config']['state_dim']), int(checkpoint['config']['action_count']))
    network.load_state_dict(checkpoint['model'])
    network.eval()
    network.config = checkpoint['config']
    return network


def load_subgoals(path):
    return json.loads(open(path).read())['subgoals']


def start_postures(postures, order, per_order=2):
    """Ten starts per order: two panel postures x five seeded 0.5 deg jitters."""
    rows = []
    for seed in START_SEEDS:
        for posture in range(per_order):
            rng = np.random.default_rng(np.random.SeedSequence([seed, posture, JITTER_SALT]))
            origin = np.asarray(postures[order * per_order + posture])
            rows.append(dict(start_seed=seed, posture=posture,
                             q=(origin + rng.uniform(-JITTER_DEG, JITTER_DEG, 2)).tolist()))
    return rows


def transfer(task, obs, agent, reacher, method, subgoals=(), selector=None, steps=arm.STEPS):
    """Run every row to sequence completion or the step budget.

    frozen    the agent alone
    reaim     on a stall the reacher heads for the current target for 300 ms (once per target)
    selector  at every free step the selector picks the agent or an option to one of the subgoals
    Returns per-row outcomes and, for `selector`, the arm state at every subgoal arrival.
    """
    n = len(obs)
    active, remaining = np.full(n, -1, dtype=int), np.zeros(n, dtype=int)
    acted, histories = set(), []
    option_steps, stalls = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    decisions, option_calls = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    first_completion, completed = np.full(n, -1, dtype=int), np.zeros(n, dtype=int)
    points = np.asarray([s['xy_m'] for s in subgoals]) if len(subgoals) else np.empty((0, 2))
    arrivals = []

    for step in range(steps):
        xy = arm.features(task)[:, :2]
        goal, visit = arm.as_array(task.goal), arm.as_array(task.sequence_index)
        finished = arm.as_array(task.finished).astype(bool)
        histories = (histories + [(xy.copy(), visit.copy())])[-(STALL_STEPS + 1):]

        if method == 'selector':
            ready = (remaining == 0) & ~finished
            if ready.any():
                with torch.no_grad():
                    q = selector(torch.from_numpy(obs[ready])).numpy()
                for values, row in zip(q, np.flatnonzero(ready)):
                    mask = np.r_[True, np.linalg.norm(points - xy[row], axis=1) > SUBGOAL_RADIUS]
                    choice = int(np.argmax(np.where(mask, values, -np.inf)))
                    decisions[row] += 1
                    if choice:
                        option_calls[row] += 1
                        active[row] = choice - 1
                        remaining[row] = OPTION_STEPS
        elif method == 'reaim' and len(histories) == STALL_STEPS + 1:
            first_xy, first_visit = histories[0]
            for row in range(n):
                if finished[row] or remaining[row] or first_visit[row] != visit[row] or (row, int(visit[row])) in acted:
                    continue
                distance = np.linalg.norm(goal[row] - xy[row])
                if distance > STALL_DISTANCE and np.linalg.norm(goal[row] - first_xy[row]) - distance < STALL_PROGRESS:
                    acted.add((row, int(visit[row])))
                    stalls[row] += 1
                    remaining[row] = OPTION_STEPS

        commands = excitation(agent, obs)
        use = np.flatnonzero((remaining > 0) & ~finished)
        if len(use):
            destinations = goal[use] if method == 'reaim' else points[active[use]]
            action, _ = reacher.predict(np.column_stack((destinations, obs[use, 10:])), deterministic=True)
            commands[use] = (np.clip(action, -1, 1) + 1) / 2
            option_steps[use] += 1
        obs = np.asarray(task.step(commands, deterministic=True)[0], dtype=np.float32)
        remaining = np.maximum(0, remaining - 1)

        for row in use:
            arrived = method == 'selector' and np.linalg.norm(arm.features(task)[row, :2] - points[active[row]]) <= SUBGOAL_RADIUS
            target_hit = int(task.sequence_index[row]) != int(visit[row]) or bool(task.finished[row])
            if arrived:
                arrivals.append(arrival(task, row, active[row], step + 1))
            if arrived or (method == 'reaim' and target_hit):
                remaining[row] = 0

        newly = arm.as_array(task.finished).astype(bool) & (first_completion < 0)
        first_completion[newly] = step + 1
        completed = np.maximum(completed, arm.as_array(task.sequence_index).astype(int) + arm.as_array(task.finished).astype(int))
        if np.all(first_completion >= 0):
            break

    outcomes = [dict(success=bool(k >= 0), completion_s=round(k * .01, 2) if k >= 0 else None, targets_completed=int(c),
                     option_steps=int(o), stalls=int(s), decisions=int(d), option_calls=int(m))
                for k, c, o, s, d, m in zip(first_completion, completed, option_steps, stalls, decisions, option_calls)]
    return outcomes, arrivals


def arrival(task, row, option, step):
    """Arm state when an option reaches its subgoal: joints, muscle forces, moment arms, activations."""
    states = task.states
    return dict(row=row, option=int(option), step=step, joint=arm.as_array(states['joint'][row]),
                force=arm.as_array(states['muscle'][row, 6]), moment=arm.as_array(states['geometry'][row, 2:]),
                activation=arm.as_array(states['muscle'][row, 0]))


def perform_cycles(agent, count, noise=0., seed=None, start=arm.START, ring=arm.RING, upper=1., fore=1.):
    """The practised order performed without interruption for `count` cycles.

    Returns phi (T + 1, 10) and, per acquired element, the step its target appeared and the step it was acquired.
    Noise comes from one generator for the whole run.
    """
    task, obs = arm.make_task(arm.PRACTISED * count, [list(start)], max_steps=10 ** 6, ring=ring, upper=upper, fore=fore)
    rng = np.random.default_rng(seed)
    phi, onsets, hits = [arm.features(task)[0]], [0], []
    for k in range(400 * 8 * count):
        commands = excitation(agent, obs)
        if noise:
            commands = np.clip(commands + rng.normal(0, noise, commands.shape), 0, 1).astype(np.float32)
        obs, _, done, _, info = task.step(commands, deterministic=True)
        obs = np.asarray(obs, dtype=np.float32)
        phi.append(arm.features(task)[0])
        if bool(np.asarray(info['completed']).reshape(-1)[0]):
            hits.append(k + 1)
            onsets.append(k + 1)
        if done or (k + 1 - onsets[-1]) >= 400:
            break
    return np.asarray(phi), np.asarray(onsets[:len(hits)]), np.asarray(hits)


def element_windows(phi, onsets, hits, cycles=7, steps=ELEMENT_STEPS):
    """Each element of the first `cycles` cycles with its recorded future; the hit step is relative to the onset."""
    rows = []
    for e, (onset, hit) in enumerate(zip(onsets, hits)):
        repeat, index = divmod(e, len(arm.PRACTISED))
        if repeat < cycles and onset + steps < len(phi):
            rows.append(dict(phi=phi[onset:onset + steps + 1], hit=hit - onset, repeat=repeat, target=arm.PRACTISED[index]))
    return rows


def perform_element(agent, start, repeat, index, steps=ELEMENT_STEPS, noise=0., rng=None):
    """One element in isolation: the sequence up to its target onset, then the goal is held for `steps` steps.

    Returns phi (steps + 1, 10) and the step its target was acquired (-1 if never; phi NaN if it never began).
    """
    target = arm.PRACTISED[index]
    prefix = arm.PRACTISED * repeat + arm.PRACTISED[:index + 1]
    task, obs = arm.make_task(prefix + (target,) * (steps + 2), [start], max_steps=10000)
    for _ in range(5000):
        if int(task.sequence_index[0]) == len(prefix) - 1:
            break
        obs, _, done, truncated, _ = task.step(excitation(agent, obs), deterministic=True)
        obs = np.asarray(obs, dtype=np.float32)
    else:
        return np.full((steps + 1, 10), np.nan), -1
    phi, hit = [arm.features(task)[0]], -1
    for k in range(steps):
        commands = excitation(agent, obs)
        if noise:
            commands = np.clip(commands + rng.normal(0, noise, commands.shape), 0, 1).astype(np.float32)
        obs, _, _, _, info = task.step(commands, deterministic=True)
        obs = np.asarray(obs, dtype=np.float32)
        if hit < 0 and bool(np.asarray(info['completed']).reshape(-1)[0]):
            hit = k + 1
        phi.append(arm.features(task)[0])
    return np.asarray(phi), hit


def available_options(info, subgoals):
    """Mask over [agent, option 1, ...]: an option is unavailable while the hand is at its subgoal."""
    hand = np.asarray(info['states']['fingertip'], dtype=float).reshape(-1)
    return np.r_[True, [np.linalg.norm(hand - np.asarray(s['xy_m'], dtype=np.float32)) > SUBGOAL_RADIUS for s in subgoals]]


def run_choice(choice, observation, info, env, agent, reacher, subgoals, max_steps=None):
    """Execute one selector choice in a single-row environment: one agent step, or an option to a subgoal."""
    if choice == 0:
        action, _ = agent.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        return dict(duration=1, rewards=[float(reward)], observation=observation, info=info,
                    terminated=terminated, truncated=truncated)
    xy = np.asarray(subgoals[choice - 1]['xy_m'], dtype=np.float32)
    timeout = OPTION_STEPS if max_steps is None else min(OPTION_STEPS, int(max_steps))
    rewards, terminated, truncated = [], False, False
    for _ in range(timeout):
        action, _ = reacher.predict(np.concatenate((xy, observation[10:30])), deterministic=True)
        observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        observation = np.asarray(observation, dtype=np.float32)
        rewards.append(float(reward))
        hand = np.asarray(info['states']['fingertip'], dtype=float).reshape(-1)
        if terminated or truncated or np.linalg.norm(hand - xy) <= SUBGOAL_RADIUS:
            break
    return dict(duration=len(rewards), rewards=rewards, observation=observation, info=info,
                terminated=terminated, truncated=truncated)

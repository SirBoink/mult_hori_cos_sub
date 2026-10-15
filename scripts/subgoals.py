"""Subgoals: multi-timescale events away from the targets, and the single-horizon comparison libraries.

    python subgoals.py agent1   events in agent 1's practised elements (isolated reaches, cycles 0-2) -> subgoals A-E
    python subgoals.py tune     the single discount whose subgoals give agent 1 the most reach progress from stalls
    python subgoals.py agent3   agent 3: multi-timescale subgoals and subgoals of the tuned single discount
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.progress import track

import arm
import control
import horizons as H

ROOT = Path(__file__).resolve().parent
TUNING_SEEDS = (270001, 270002)   # two jittered starts per target
TUNING_JITTER = 3.0               # deg
TUNING_STEPS = 1530
PROGRESS_STEP = 50                # progress toward the target is measured 500 ms after the stall
BRANCH_STEPS = 100
BATCH = 8                         # discounts branched together


def write(name, description, found, **extra):
    library = dict(description=description, termination_radius_m=control.SUBGOAL_RADIUS,
                   timeout_steps=control.OPTION_STEPS, **extra,
                   subgoals=[dict(id=chr(65 + i), xy_m=[float(v) for v in e['xy_m']], element=int(e['element']),
                                  step=int(e['step']), **({'bands': [min(e['bands']), max(e['bands'])]} if 'bands' in e else {}))
                             for i, e in enumerate(found)])
    (ROOT / f'data/subgoals/{name}.json').write_text(json.dumps(library, indent=1) + '\n')
    print(name, [s['xy_m'] for s in library['subgoals']])


def practised_elements(path, repeats):
    with np.load(path) as data:
        data = {k: data[k] for k in data.files}
    ids = np.flatnonzero(np.isin(data['repeat'], repeats) & (data['hit'] > 0))
    return data, ids


def find(path, repeats):
    data, ids = practised_elements(path, repeats)
    found, _, singles = H.detect(data['phi'][ids], data['hit'][ids], data['target'][ids], ids, arm.targets())
    return data, ids, found, singles


def agent1():
    _, _, found, _ = find(ROOT / 'data/recordings/isolated/agent1.npz', [0, 1, 2])
    write('agent1', 'multi-timescale events of agent 1, isolated reaches of cycles 0-2', H.strongest(H.merge(found)))


def single_libraries():
    """Subgoals of each of the 40 single discounts in agent 1's practised elements (all features)."""
    data, ids, _, singles = find(ROOT / 'data/recordings/isolated/agent1.npz', [0, 1, 2])
    goals = arm.targets()
    return [H.single_horizon_points([s[g] for s in singles], data['phi'][ids], data['hit'][ids], goals) for g in range(40)]


def capture(task, obs):
    frame = {f'state_{k}': arm.as_array(v) for k, v in task.states.items()}
    frame.update(observation=obs.copy(), goal=arm.as_array(task.goal), visit=arm.as_array(task.sequence_index),
                 finished=arm.as_array(task.finished), hold=arm.as_array(task.hold_count), phi=arm.features(task))
    frame.update({f'buffer_{k}': np.stack([arm.as_array(v) for v in values], axis=1)
                  for k, values in task.obs_buffer.items() if len(values)})
    return frame


def record_reaches(agent, sequence, postures):
    """Every frame of a batch of reaches (state before each step)."""
    task, obs = arm.make_task(sequence, postures, max_steps=20000)
    frames = []
    for step in range(TUNING_STEPS + 1):
        frames.append(capture(task, obs))
        if step == TUNING_STEPS:
            break
        obs = np.asarray(task.step(control.excitation(agent, obs), deterministic=True)[0], dtype=np.float32)
    return {key: np.stack([f[key] for f in frames]) for key in frames[0]}


def restore(data, k, sequence):
    task, _ = arm.make_task(sequence, [arm.START], max_steps=20000)
    task.effector.states = {key[6:]: torch.tensor(data[key][k:k + 1], dtype=torch.float32)
                            for key in data if key.startswith('state_')}
    task.sequence_index = torch.tensor(data['visit'][k:k + 1], dtype=torch.long)
    task.finished = torch.tensor(data['finished'][k:k + 1], dtype=torch.bool)
    task.hold_count = torch.tensor(data['hold'][k:k + 1], dtype=torch.long)
    task.goal = torch.tensor(data['goal'][k:k + 1], dtype=torch.float32)
    task.elapsed_steps, task.elapsed = k, k * .01
    for key in task.obs_buffer:
        if 'buffer_' + key in data:
            task.obs_buffer[key] = [torch.tensor(data['buffer_' + key][k:k + 1, j], dtype=torch.float32)
                                    for j in range(data['buffer_' + key].shape[1])]
    return task, data['observation'][k:k + 1].copy()


def first_stall(data, last=800):
    """First step with the hand > 4 cm from its target and < 1 mm closer than 100 ms earlier (30 if none)."""
    distance = np.linalg.norm(data['phi'][:, :2] - data['goal'], axis=1)
    target = np.argmax(data['observation'][:, 2:10], axis=1)
    for k in range(10, min(len(distance), last + 1)):
        if (not bool(data['finished'][k]) and np.all(target[k - 10:k + 1] == target[k]) and distance[k] > .04
                and distance[k - 10] - distance[k] < .001):
            return k
    return 30


def nearest_subgoal(xy, library):
    available = [(np.linalg.norm(xy - np.asarray(s['xy_m'])), i) for i, s in enumerate(library)
                 if np.linalg.norm(xy - np.asarray(s['xy_m'])) > control.SUBGOAL_RADIUS]
    return min(available)[1] if available else None


def branch(data, k, sequence, libraries, agent, reacher):
    """From frame k, one row per library: an option to its nearest subgoal, then the agent; progress at 500 ms."""
    task, obs = restore(data, k, sequence)
    goal = arm.as_array(task.goal)[0]
    visit = int(task.sequence_index[0])
    start = np.linalg.norm(arm.features(task)[0, :2] - goal)
    n = len(libraries)
    chosen = [nearest_subgoal(arm.features(task)[0, :2], library) for library in libraries]
    destinations = np.asarray([library[c]['xy_m'] if c is not None else goal for c, library in zip(chosen, libraries)])
    available = np.array([c is not None for c in chosen])
    task.effector.states = {key: value.repeat((n,) + (1,) * (value.ndim - 1)) for key, value in task.states.items()}
    for key in ('sequence_index', 'finished', 'hold_count'):
        setattr(task, key, getattr(task, key).repeat(n))
    task.goal = task.goal.repeat(n, 1)
    task.obs_buffer = {key: [v.repeat(n, 1) for v in values] for key, values in task.obs_buffer.items()}
    obs = np.repeat(obs, n, axis=0)
    acquired, progress = np.full(n, np.nan), None
    for offset in range(BRANCH_STEPS):
        available &= np.linalg.norm(arm.features(task)[:, :2] - destinations, axis=1) > control.SUBGOAL_RADIUS
        use = np.flatnonzero(np.isnan(acquired) & available & (offset < control.OPTION_STEPS))
        commands = control.excitation(agent, obs)
        if len(use):
            action, _ = reacher.predict(np.column_stack((destinations[use], obs[use, 10:])), deterministic=True)
            commands[use] = (np.clip(action, -1, 1) + 1) / 2
        obs, _, _, truncated, _ = task.step(commands, deterministic=True)
        obs = np.asarray(obs, dtype=np.float32)
        hit = arm.as_array(task.finished).astype(bool) | (arm.as_array(task.sequence_index) != visit)
        acquired[np.isnan(acquired) & hit] = (offset + 1) * .01
        if offset == PROGRESS_STEP - 1:   # acquisition counts as reaching the target
            progress = np.where(np.isfinite(acquired), start, start - np.linalg.norm(arm.features(task)[:, :2] - goal, axis=1))
        if truncated:
            break
    return progress


def tune():
    """Mean 500 ms progress toward the target when each discount's subgoals move agent 1 out of a stall."""
    torch.set_num_threads(1)
    agent = control.load_policy(ROOT / 'checkpoints/agent1.zip')
    reacher = control.load_policy(ROOT / 'checkpoints/reacher.zip', 22)
    libraries = single_libraries()
    progress = []
    for trial in track(range(len(arm.PRACTISED)), description='tuning the single horizon'):
        sequence = (arm.PRACTISED[trial],) * 2000
        postures = [(np.array(arm.START) + np.random.default_rng(np.random.SeedSequence([seed, trial, 610]))
                     .uniform(-TUNING_JITTER, TUNING_JITTER, 2)).tolist() for seed in TUNING_SEEDS]
        frames = record_reaches(agent, sequence, postures)
        for row in range(len(postures)):
            data = {key: value[:, row] for key, value in frames.items()}
            k = first_stall(data)
            progress.append(np.concatenate([branch(data, k, sequence, libraries[g:g + BATCH], agent, reacher)
                                            for g in range(0, 40, BATCH)]))
    mean = np.mean(progress, axis=0)
    best = max(range(40), key=lambda g: (mean[g], -g))
    out = dict(selected=best, discount=float(H.discounts()[best]), horizon_s=float(H.mean_horizons()[best]),
               progress_m=mean.tolist(), reaches=len(progress))
    (ROOT / 'results/single_horizon.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'discount {best} (horizon {out["horizon_s"]:.3f} s), progress {mean[best] * 1000:.1f} mm')


def agent3():
    data, ids, found, singles = find(ROOT / 'data/recordings/continuous/agent3.npz', [1, 2, 3])
    multiscale = H.strongest(H.merge(found))
    write('agent3_multiscale', 'multi-timescale events of agent 3, continuous cycles 1-3', multiscale)
    g = json.loads((ROOT / 'results/single_horizon.json').read_text())['selected']
    points = H.single_horizon_points([s[g] for s in singles], data['phi'][ids], data['hit'][ids], arm.targets())
    points = [dict(p, element=int(ids[p['element']])) for p in points][:len(multiscale)]
    write('agent3_single', f'peaks of single discount {g} of agent 3, continuous cycles 1-3', points,
          discount=g, horizon_s=float(H.mean_horizons()[g]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['agent1', 'tune', 'agent3'])
    dict(agent1=agent1, tune=tune, agent3=agent3)[parser.parse_args().stage]()

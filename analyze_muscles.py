"""Muscle patterns at transition events, and joint torques when options reach their subgoals.

    python analyze_muscles.py   -> results/muscles.json

decoding  events in agent 1's continuous recordings (noise-free and five noise seeds, cycles 1-6): which reach an event
          belongs to, decoded from the six muscle activations or from position-free kinematics (phase, speed / peak
          speed, acceleration / largest acceleration, time from peak speed); nearest centroid, leave one recording out
torque    per-muscle joint torque (-force x moment arm) at every subgoal arrival of agent 1's selectors
"""
import json
from pathlib import Path

import numpy as np

import arm
import horizons as H

ROOT = Path(__file__).resolve().parent
RECORDINGS = ['agent1'] + [f'agent1_noise{k}' for k in range(1, 6)]
PERMUTATIONS = 1000


def describe(name):
    with np.load(ROOT / f'data/recordings/continuous/{name}.npz') as d:
        keep = (d['repeat'] >= 1) & (d['hit'] > 0)
        phis, stops, targets = d['phi'][keep], d['hit'][keep], d['target'][keep]
    found, _, _ = H.detect(phis, stops, targets, np.arange(len(stops)), arm.targets())
    rows = []
    for phi, stop, target, events in zip(phis, stops, targets, found):
        k = np.arange(1, stop + 1)
        speed = np.linalg.norm(phi[k, 2:4], axis=1)
        accel = np.gradient(speed)
        peak = int(np.argmax(speed))
        for e in events:
            j = e['step'] - 1
            if 0 <= j < len(k):
                rows.append(dict(recording=name, target=int(target), muscles=phi[k[j], 4:10].tolist(),
                                 kinematics=[e['step'] / stop, float(speed[j] / speed[peak]),
                                             float(accel[j] / np.abs(accel).max()), 10 * (j - peak)]))
    return rows


def nearest_centroid(x, y, groups):
    hits = []
    for group in np.unique(groups):
        train, held = groups != group, groups == group
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
        z = (x - mu) / sd
        classes = np.unique(y[train])
        centres = np.array([z[train][y[train] == c].mean(0) for c in classes])
        guess = classes[np.argmin(((z[held][:, None] - centres[None]) ** 2).sum(-1), axis=1)]
        hits += (guess == y[held]).tolist()
    return float(np.mean(hits))


def decoding():
    events = [row for name in RECORDINGS for row in describe(name)]
    y, groups = np.array([e['target'] for e in events]), np.array([e['recording'] for e in events])
    muscles, kinematics = np.array([e['muscles'] for e in events]), np.array([e['kinematics'] for e in events])
    accuracy = nearest_centroid(muscles, y, groups)
    rng = np.random.default_rng(0)
    null = [nearest_centroid(muscles, rng.permutation(y), groups) for _ in range(PERMUTATIONS)]
    return dict(events=len(events), recordings=len(RECORDINGS), muscles=accuracy, kinematics=nearest_centroid(kinematics, y, groups),
                chance=float(np.bincount(y).max() / len(y)), null=null,
                p=float((1 + sum(n >= accuracy for n in null)) / (1 + PERMUTATIONS)))


def torque():
    arrivals = [np.load(ROOT / f'results/transfer/agent1_seed{k}_arrivals.npz') for k in (1, 2, 3)]
    out = {}
    for option in range(5):
        trials = []
        for seed, a in enumerate(arrivals):
            mine = a['option'] == option
            per_muscle = -a['moment'][mine] * a['force'][mine][:, None, :]     # arrivals x joint x muscle
            keys = np.stack([a['order'][mine], a['start_seed'][mine], a['posture'][mine]], 1)
            for key in np.unique(keys, axis=0):
                t = per_muscle[(keys == key).all(1)]
                trials.append([np.maximum(t, 0).sum(2).mean(0), np.minimum(t, 0).sum(2).mean(0)])
        trials = np.array(trials)                                               # trials x (positive, negative) x joint
        out[chr(65 + option)] = dict(trials=len(trials), arrivals=int(sum((a['option'] == option).sum() for a in arrivals)),
                                     positive=trials[:, 0].mean(0).tolist(), negative=trials[:, 1].mean(0).tolist(),
                                     net=trials.sum(1).mean(0).tolist())
    return out


if __name__ == '__main__':
    out = dict(decoding=decoding(), torque=torque())
    (ROOT / 'results/muscles.json').write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out['decoding'].items() if k != 'null'})
    print(json.dumps(out['torque'], indent=1))

"""Record the agents' movements (outputs in data/recordings/).

    python record.py isolated     each element of cycles 0-2 in isolation (goal held afterwards); agents 1-3,
                                  noise-free and with 5 motor-noise seeds
    python record.py continuous   20 uninterrupted cycles; elements of cycles 0-6 with their 15.3 s future
    python record.py conditions   agent 1 with heavier limbs or a rescaled workspace; a condition is kept only if the
                                  agent completes cycles 1-3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.progress import track

import arm
import control

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/recordings'
CONDITIONS = ([('mass', f) for f in (.5, .7, .85, 1.15, 1.3, 1.8)]
              + [('forearm', f) for f in (.5, .7, .85, 1.3, 1.5, 1.8, 2.2)]
              + [('workspace', f) for f in (.6, .8, .9, 1.1, 1.2, 1.4)])


def runs():
    """(agent, noise index or None) for every recording."""
    return [(a, k) for a in (1, 2, 3) for k in [None, 1, 2, 3, 4, 5]]


def name(agent, k):
    return f'agent{agent}' if k is None else f'agent{agent}_noise{k}'


def save(path, rows):
    np.savez_compressed(path, **{key: np.stack([r[key] for r in rows]) for key in ('phi', 'hit', 'repeat', 'target')})


def isolated():
    plan = [(r, i) for r in (0, 1, 2) for i in range(len(arm.PRACTISED))]
    for agent, k in runs():
        policy = control.load_policy(ROOT / f'checkpoints/agent{agent}.zip')
        seed = None if k is None else control.NOISE_SEEDS[k - 1]
        rows = []
        for r, i in track(plan, description=name(agent, k)):
            rng = None if seed is None else np.random.default_rng([seed, r, i])
            phi, hit = control.perform_element(policy, list(arm.START), r, i, noise=0. if seed is None else control.NOISE, rng=rng)
            rows.append(dict(phi=phi, hit=hit, repeat=r, target=arm.PRACTISED[i]))
        save(OUT / f'isolated/{name(agent, k)}.npz', rows)


def continuous():
    for agent, k in track(runs(), description='continuous recordings'):
        policy = control.load_policy(ROOT / f'checkpoints/agent{agent}.zip')
        seed = None if k is None else control.NOISE_SEEDS[k - 1]
        phi, onsets, hits = control.perform_cycles(policy, 20, 0. if seed is None else control.NOISE, seed)
        rows = control.element_windows(phi, onsets, hits)
        if rows:   # an agent that stops progressing early leaves no element with its full future
            save(OUT / f'continuous/{name(agent, k)}.npz', rows)
        print(name(agent, k), f'{len(rows)} elements ({len(hits)} targets acquired)')


def conditions():
    policy = control.load_policy(ROOT / 'checkpoints/agent1.zip')
    summary = {}
    for body, f in track(CONDITIONS, description='conditions'):
        change = dict(ring=arm.RING * f) if body == 'workspace' else dict(upper=f) if body == 'mass' else dict(fore=f)
        phi, onsets, hits = control.perform_cycles(policy, 20, **change)
        rows = control.element_windows(phi, onsets, hits)
        usable = [r for r in rows if r['repeat'] >= 1]
        label = f'{body}_x{f:g}'
        summary[label] = dict(body=body, factor=f, acquisitions=len(hits),
                              median_reach_ms=float(np.median([r['hit'] for r in usable]) * 10) if usable else None,
                              kept=sum(1 <= r['repeat'] <= 3 for r in rows) == 24)
        if summary[label]['kept']:
            save(OUT / f'conditions/{label}.npz', rows)
    (OUT / 'conditions/conditions.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['isolated', 'continuous', 'conditions'])
    torch.set_num_threads(1)
    dict(isolated=isolated, continuous=continuous, conditions=conditions)[parser.parse_args().kind]()

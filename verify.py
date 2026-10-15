"""Quick reproducibility checks (~3 min): subgoals, recordings, evaluations and panels match the saved files.

    python verify.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

import arm
import control
import evaluate
import horizons as H
import subgoals

ROOT = Path(__file__).resolve().parent


def check_subgoals():
    _, _, found, _ = subgoals.find(ROOT / 'data/recordings/isolated/agent1.npz', [0, 1, 2])
    saved = json.loads((ROOT / 'data/subgoals/agent1.json').read_text())['subgoals']
    assert [e['xy_m'] for e in H.strongest(H.merge(found))] == [s['xy_m'] for s in saved]
    print('subgoals A-E rediscovered')


def check_recordings():
    agent = control.load_policy(ROOT / 'checkpoints/agent1.zip')
    isolated = np.load(ROOT / 'data/recordings/isolated/agent1.npz')
    for e in range(3):
        phi, hit = control.perform_element(agent, list(arm.START), int(isolated['repeat'][e]), arm.PRACTISED.index(int(isolated['target'][e])))
        assert np.array_equal(phi, isolated['phi'][e]) and hit == isolated['hit'][e]
    phi, onsets, hits = control.perform_cycles(agent, 20)
    continuous = np.load(ROOT / 'data/recordings/continuous/agent1.npz')
    rows = control.element_windows(phi, onsets, hits)
    assert all(np.array_equal(r['phi'], p) for r, p in zip(rows, continuous['phi'])) and len(rows) == len(continuous['phi'])
    print('recordings reproduced')


def check_evaluations():
    for job in ('agent1_seed1', 'agent1_reaim', 'agent3_multiscale_seed1'):
        new = evaluate.run(job, orders=2)
        with (ROOT / f'results/transfer/{job}.csv').open() as stream:
            saved = [r for r in csv.DictReader(stream) if int(r['order']) < 2]
        for a, b in zip(new, saved):
            assert str(a['success']) == b['success'] and a['targets_completed'] == int(b['targets_completed'])
            assert a['option_steps'] == int(b['option_steps']) and str(a['completion_s'] or '') == b['completion_s']
    print('evaluations reproduced')


def check_panels():
    load = lambda name: [tuple(o) for o in json.loads((ROOT / f'data/panels/{name}.json').read_text())['orders']]
    orders, validation, heldout = load('orders96'), load('orders16_validation'), load('orders96_heldout')
    for panel in (orders, validation, heldout, load('sequences48')):
        assert len(set(panel)) == len(panel) and all(o[0] == 0 for o in panel) and arm.PRACTISED not in panel
    assert not set(heldout) & (set(orders) | set(validation))
    print('panels: unique, unseen, held-out orders disjoint')


if __name__ == '__main__':
    torch.set_num_threads(1)
    check_panels()
    check_subgoals()
    check_recordings()
    check_evaluations()

"""Zero-shot evaluation on unseen orders: one CSV row per trial (10 starts per order).

    python evaluate.py --all --workers 6      every evaluation behind the figures (~40 min)
    python evaluate.py --job agent1_seed1     one job
    python evaluate.py --job agent1_seed1 --orders 2
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

import arm
import control

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/transfer'
FIELDS = ['order', 'targets', 'start_seed', 'posture', 'success', 'targets_completed', 'completion_s', 'option_steps',
          'stalls', 'decisions', 'option_calls']


def jobs():
    """name -> (agent, panel, method, selector, subgoals)"""
    table = {}
    for prefix, panel in (('agent1', 'orders96'), ('agent1_long', 'sequences48')):
        table[f'{prefix}_frozen'] = (1, panel, 'frozen', None, None)
        table[f'{prefix}_reaim'] = (1, panel, 'reaim', None, None)
        for k in (1, 2, 3):
            table[f'{prefix}_seed{k}'] = (1, panel, 'selector', f'agent1_seed{k}', 'agent1')
    table['agent3_frozen'] = (3, 'orders96_heldout', 'frozen', None, None)
    for library in ('multiscale', 'single'):
        for k in range(1, 6):
            table[f'agent3_{library}_seed{k}'] = (3, 'orders96_heldout', 'selector', f'agent3_{library}_seed{k}', f'agent3_{library}')
    return table


def run(name, orders=None, queue=None):
    torch.set_num_threads(1)
    agent_id, panel_name, method, selector_name, subgoals_name = jobs()[name]
    panel = json.loads((ROOT / f'data/panels/{panel_name}.json').read_text())
    agent = control.load_policy(ROOT / f'checkpoints/agent{agent_id}.zip')
    reacher = control.load_policy(ROOT / 'checkpoints/reacher.zip', 22)
    selector = control.load_selector(ROOT / f'checkpoints/selectors/{selector_name}.pt') if selector_name else None
    subgoals = control.load_subgoals(ROOT / f'data/subgoals/{subgoals_name}.json') if subgoals_name else ()
    rows, arrivals = [], []
    for o, order in enumerate(panel['orders'][:orders]):
        starts = control.start_postures(panel['postures_deg'], o)
        task, obs = arm.make_task(order, [s['q'] for s in starts], panel['max_steps'])
        outcomes, arrived = control.transfer(task, obs, agent, reacher, method, subgoals, selector, panel['max_steps'])
        rows += [dict(order=o, targets=' '.join(map(str, order)), start_seed=s['start_seed'], posture=s['posture'], **r)
                 for s, r in zip(starts, outcomes)]
        arrivals += [dict(a, order=o, start_seed=starts[a['row']]['start_seed'], posture=starts[a['row']]['posture'])
                     for a in arrived]
        if queue is not None:
            queue.put(1)
    if orders is not None:
        return rows
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if name.startswith('agent1_seed'):
        np.savez_compressed(OUT / f'{name}_arrivals.npz',
                            **{key: np.array([a[key] for a in arrivals]) for key in
                               ('order', 'start_seed', 'posture', 'option', 'step', 'joint', 'force', 'moment', 'activation')})
    return rows


def run_all(workers):
    names = list(jobs())
    total = sum(len(json.loads((ROOT / f'data/panels/{jobs()[n][1]}.json').read_text())['orders']) for n in names)
    with Manager() as manager, ProcessPoolExecutor(workers) as pool, \
            Progress(TextColumn('evaluations'), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        queue = manager.Queue()
        bar = progress.add_task('', total=total)
        futures = [pool.submit(run, name, None, queue) for name in names]
        done = 0
        while done < total:
            try:
                queue.get(timeout=5)
                done += 1
                progress.update(bar, completed=done)
            except Empty:
                for future in futures:
                    if future.done() and future.exception():
                        raise future.exception()
        for future in futures:
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--job', choices=list(jobs()))
    parser.add_argument('--orders', type=int, help='first N orders only, nothing written')
    args = parser.parse_args()
    if args.all:
        run_all(args.workers)
    else:
        rows = run(args.job, args.orders)
        print(f"{args.job}: {sum(r['success'] for r in rows)}/{len(rows)} sequences completed")

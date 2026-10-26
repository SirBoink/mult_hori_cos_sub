"""The unchanged rule on trajectories simulated by the published flat (vpSOC) and hierarchical (HiSeq) optimal-control
models of Cuevas Rivera & Kiebel, built exactly as in their simulations.py, noise-free as in their published sets.

    python model_comparison.py   -> results/model_comparison.json

For each model, SETS parameter sets are drawn from the published grid (paper_conf). Each set runs the six sequences
once. Per (set, sequence) we score the rule's events as for humans: rate inside the second target (analyze_humans.chunking)
and share within 50 ms of a via-point passage (analyze_humans.via_points).
"""
import json
import os
import tempfile
import warnings
import zipfile

import numpy as np
from rich.progress import track

import analyze_humans as A   # also puts the published code on sys.path
import human
from src import paper_conf as pc
from src import simulations as sims
from src.model import hierarchical as hie

SETS = 40
QUIET = {'motor_noise': False, 'obs_noise': False, 'add_noise': False}


def flat(r, h):
    """vpSOC as in sims.vpsoc3_reference."""
    agent = hie.vpSOC(t_ends=1.5, h_scaling=h, soc_pars={'r': r, 'add_noise': 10000})
    return sims.all_seqs(agent=agent, num_runs=1, t_end=1.5, dyn_kwargs=QUIET)[0]


def hierarchical(h, r, mondis):
    """HiSeq as in sims.hiseq_reference."""
    agent = hie.HiSeq(t_ends=1.7, h_scaling=h, dt=.01, mondis_abs=False,
                      soc_pars={'sigma_c': .3, 'add_noise': 5000, 'r': r, 'omega_nu': .6, 'omega_f': .6})
    agent.delta_t_monitor, agent.monitor_distance, agent.monitor_sd, agent.tau23 = 0., mondis, 3, 5
    return sims.all_seqs(agent, num_runs=1, t_end=3.4, background_noise=0, dyn_kwargs=QUIET)[0]


def score(data, mean, std):
    """Rule events of one parameter set's six trials -> per-sequence rate in the second target and via-point share."""
    trials, skipped = [], 0
    for _, trial in data.groupby('unique_trial_id'):
        xy, id_move = trial[['pos_x', 'pos_y']].to_numpy(float), int(trial['id_move'].iloc[0])
        try:
            trials.append(dict(id_move=id_move, xy=xy, bounds=human.movement_bounds(xy, human.SEQUENCES[id_move])))
        except (StopIteration, IndexError, ValueError):   # never moved, or missed a via target
            skipped += 1
    events = A.whole_movement_events(trials, mean, std) if trials else []
    units = []
    for i, t in enumerate(trials):
        mine = [e for e in events if e['trial'] == i]
        second = human.SEQUENCES[t['id_move']][1]
        frames = int(human.inside(t['xy'][t['bounds'][0]:len(t['xy']) - 1], second).sum())
        hits = sum(int(human.inside(np.asarray(e['xy_m']) * 1000, second)) for e in mine)
        near = [min(abs(t['bounds'][0] + e['step'] - q) for q in t['bounds'][1:]) for e in mine]
        units.append(dict(rate=hits / max(frames, 1), near=int(sum(n <= 5 for n in near)), events=len(mine)))
    return units, skipped


def run(name, build, grid, mean, std):
    rng = np.random.default_rng(0)
    picks = rng.choice(len(grid), min(SETS, len(grid)), replace=False)
    units, skipped = [], 0
    for k in track(picks, description=name):
        u, s = score(build(*grid[k]), mean, std)
        units += u
        skipped += s
    events = sum(u['events'] for u in units)
    return dict(sets=len(picks), trials=len(units), skipped=skipped, rates=[u['rate'] for u in units],
                within_50ms=sum(u['near'] for u in units) / max(events, 1))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    mean, std = human.development_scale()
    with tempfile.TemporaryDirectory() as work:   # the published models read ./experiment_sheet.svg
        with zipfile.ZipFile(human.ARCHIVE) as archive, open(os.path.join(work, 'experiment_sheet.svg'), 'wb') as f:
            f.write(archive.read('bids/code/experiment_sheet.svg'))
        here = os.getcwd()
        os.chdir(work)
        try:
            flat_grid = [(r, h) for r in pc.vpsoc3_pars['r'] for h in pc.vpsoc3_pars['h']]
            hierarchical_grid = [(h, r, m) for h in pc.all_h for r in pc.all_r for m in pc.all_mondis]
            out = dict(flat=run('flat model', flat, flat_grid, mean, std),
                       hierarchical=run('hierarchical model', hierarchical, hierarchical_grid, mean, std))
        finally:
            os.chdir(here)
    humans = json.loads((A.ROOT / 'results/humans.json').read_text())
    out['humans'] = dict(rates=humans['chunking']['hierarchical'] + humans['chunking']['flat'],
                         within_50ms=humans['via_points']['within_50ms']['events'])
    for k in ('flat', 'hierarchical', 'humans'):
        print(k, 'median rate', np.median(out[k]['rates']), 'within 50 ms', out[k]['within_50ms'],
              {key: out[k][key] for key in ('sets', 'trials', 'skipped') if key in out[k]})
    (A.ROOT / 'results/model_comparison.json').write_text(json.dumps(out, indent=1))

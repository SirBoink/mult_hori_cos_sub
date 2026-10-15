"""Zero-shot transfer statistics from results/transfer/*.csv.

    python analyze_transfer.py   -> results/transfer.json

agent1   selectors trained on the practised order only, on 96 unseen 8-target orders and 48 unseen 16-target sequences;
         bootstrap over selector seeds, then orders
detail   the 96 orders by start set: completion, failure-capped duration, targets completed, option use
agent3   the independently trained agent: selectors trained on random orders with multi-timescale or single-horizon
         subgoals, on 96 held-out orders; bootstrap over seeds within each library, then orders
"""
import csv
import json
from pathlib import Path

import numpy as np

import control

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / 'results/transfer'
BOOT = 10_000


def rows(name):
    with (TRIALS / f'{name}.csv').open() as stream:
        return sorted(csv.DictReader(stream), key=lambda r: (int(r['order']), int(r['start_seed']), int(r['posture'])))


def success(name):
    """orders x 10 starts."""
    return np.array([r['success'] == 'True' for r in rows(name)], float).reshape(-1, 10)


def duration(name, cap):
    return np.array([float(r['completion_s']) if r['success'] == 'True' else cap for r in rows(name)])


def seeds_then_orders(a, b, rng):
    """a: seeds x orders, b: orders; mean over seeds of a - b, resampling seeds then orders."""
    s = rng.integers(0, a.shape[0], (BOOT, a.shape[0]))
    o = rng.integers(0, a.shape[1], (BOOT, a.shape[1]))
    d = [a[si][:, oi].mean() - b[oi].mean() for si, oi in zip(s, o)]
    return [float(a.mean() - b.mean()), *np.percentile(d, [2.5, 97.5]).tolist()]


def agent1():
    out = {}
    for panel, prefix in (('orders96', 'agent1'), ('sequences48', 'agent1_long')):
        frozen, reaim = success(f'{prefix}_frozen').mean(1), success(f'{prefix}_reaim').mean(1)
        selectors = np.array([success(f'{prefix}_seed{k}').mean(1) for k in (1, 2, 3)])
        out[panel] = dict(orders=len(frozen), frozen=float(frozen.mean()), reaim=float(reaim.mean()),
                          seeds=selectors.mean(1).tolist(), selectors=float(selectors.mean()),
                          selectors_minus_frozen=seeds_then_orders(selectors, frozen, np.random.default_rng(0)))
    return out


def detail():
    frozen = rows('agent1_frozen')
    learned = [r for k in (1, 2, 3) for r in rows(f'agent1_seed{k}')]
    cap = lambda r: float(r['completion_s']) if r['success'] == 'True' else 8.0
    summary = lambda rs: dict(
        completion=float(np.mean([r['success'] == 'True' for r in rs])),
        duration=float(np.mean([cap(r) for r in rs])), duration_sem=float(np.std([cap(r) for r in rs], ddof=1) / np.sqrt(len(rs))),
        option_share=float(sum(int(r['option_calls']) for r in rs) / max(sum(int(r['decisions']) for r in rs), 1)),
        targets=np.bincount([int(r['targets_completed']) for r in rs], minlength=9).tolist())
    by_start = {str(s): dict(frozen=summary([r for r in frozen if int(r['start_seed']) == s]),
                             hierarchical=summary([r for r in learned if int(r['start_seed']) == s])) for s in control.START_SEEDS}
    return dict(start_sets=by_start, pooled=dict(frozen=summary(frozen), hierarchical=summary(learned)))


def agent3():
    rng = np.random.default_rng(42)
    multiscale = np.array([success(f'agent3_multiscale_seed{k}') for k in range(1, 6)])
    single = np.array([success(f'agent3_single_seed{k}') for k in range(1, 6)])
    frozen = success('agent3_frozen')
    n = multiscale.shape[1]
    minus_single, minus_frozen = [], []
    for _ in range(BOOT):
        o = rng.integers(0, n, n)
        m = multiscale[rng.integers(0, len(multiscale), len(multiscale))][:, o].mean()
        minus_single.append(m - single[rng.integers(0, len(single), len(single))][:, o].mean())
        minus_frozen.append(m - frozen[o].mean())
    seeds_m, seeds_s = multiscale.mean((1, 2)), single.mean((1, 2))
    time = lambda names: float(np.mean(np.concatenate([duration(x, 8.0) for x in names])))
    # per-order completion (over starts and seeds) with a 95% order-bootstrap interval of the mean
    per_order = dict(frozen=frozen.mean(1), single=single.mean((0, 2)), multiscale=multiscale.mean((0, 2)))
    boot = rng.integers(0, n, (BOOT, n))
    orders = {k: dict(means=v.tolist(), ci=np.percentile(v[boot].mean(1), [2.5, 97.5]).tolist(),
                      trials=int(t.size), successes=int(t.sum()))
              for (k, v), t in zip(per_order.items(), (frozen, single, multiscale))}
    return dict(frozen=float(frozen.mean()), multiscale=float(multiscale.mean()), single=float(single.mean()),
                multiscale_seeds=seeds_m.tolist(), single_seeds=seeds_s.tolist(),
                multiscale_minus_single=[float(multiscale.mean() - single.mean()), *np.percentile(minus_single, [2.5, 97.5]).tolist()],
                multiscale_minus_frozen=[float(multiscale.mean() - frozen.mean()), *np.percentile(minus_frozen, [2.5, 97.5]).tolist()],
                multiscale_seeds_above_every_single=int((seeds_m > seeds_s.max()).sum()), orders=orders,
                duration=dict(frozen=time(['agent3_frozen']), single=time([f'agent3_single_seed{k}' for k in range(1, 6)]),
                              multiscale=time([f'agent3_multiscale_seed{k}' for k in range(1, 6)])))


if __name__ == '__main__':
    out = dict(agent1=agent1(), detail=detail(), agent3=agent3())
    (ROOT / 'results/transfer.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['agent1'], indent=1))
    print(json.dumps({k: v for k, v in out['agent3'].items()}, indent=1))

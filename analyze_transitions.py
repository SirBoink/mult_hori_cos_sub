"""Where in a movement the coherence rule fires, and whether a single horizon could do the same.

    python analyze_transitions.py   -> results/transitions.json

onsets      17 isolated-reach recordings (agents 1-3, noise-free and noisy; agent 1's noise-free recording was used to
            develop the method and is left out): share of events in the first third of each reach vs chance
bands       per-band onset share at the rule's event count (kinematics), for the cross-system comparison
tempo       the same recordings stretched in time (x0.5 to x3): rule vs single discounts
robustness  rule vs the single horizon tuned once on agent 1, across bodies, workspaces, agents, tempi and humans
"""
import json
from pathlib import Path

import numpy as np
from rich.progress import track
from scipy.stats import binomtest

import arm
import horizons as H
import human

ROOT = Path(__file__).resolve().parent
REC = ROOT / 'data/recordings'
KIN, ALL = slice(0, 4), slice(None)
GOALS = arm.targets()
WARPS = (0.5, 1.0, 1.8, 3.0)
BOOT = 10_000
CONDITIONS = ['mass_x1.15', 'mass_x1.3', 'forearm_x1.3', 'forearm_x1.5', 'forearm_x1.8', 'workspace_x0.8', 'workspace_x0.9',
              'workspace_x1.1']


def isolated_recordings():
    return [p for a in (1, 2, 3) for p in ([] if a == 1 else [REC / f'isolated/agent{a}.npz'])
            + [REC / f'isolated/agent{a}_noise{k}.npz' for k in range(1, 6)]]


def load(path, continuous=False):
    with np.load(path) as d:
        keep = (d['hit'] > 0) & ((d['repeat'] >= 1) if continuous else True)
        return d['phi'][keep], [int(s) for s in d['hit'][keep]]


def changes(phis, stops, columns):
    mean, std = H.load_scale(columns)
    out = [H.change(phi[:, columns], mean, std) for phi in phis]
    return [m[:, :s] for (m, _), s in zip(out, stops)], [d[:, :s] for (_, d), s in zip(out, stops)]


def warp(phis, stops, w):
    """Stretch time by w (positions and velocities resampled, not rescaled); keep the 15.3 s future."""
    out = []
    for p in phis:
        p = p[:, KIN].astype(float)
        t = np.arange(0, len(p) - 1 + 1e-9, 1 / w)
        p = np.column_stack([np.interp(t, np.arange(len(p)), p[:, j]) for j in range(4)])
        out.append(np.vstack([p, np.tile(p[-1], (max(0, 1531 - len(p)), 1))]))
    return out, [int(round(s * w)) for s in stops]


def onsets():
    rows = {}
    for path in track(isolated_recordings(), description='onsets'):
        phis, stops = load(path)
        frames = np.concatenate([np.arange(1, s + 1) / s for s in stops])
        row = dict(elements=len(stops), chance=float((frames < H.ONSET).mean()))
        for name, columns in (('all', ALL), ('kinematics', KIN)):
            bands, _ = changes(phis, stops, columns)
            t = H.tally(H.rule_phases(bands, phis, stops, 3, GOALS))
            row[name] = dict(events=int(t[1]), early=int(t[0]), share=float(t[0] / t[1]))
        row['p'] = float(binomtest(row['all']['early'], row['all']['events'], row['chance'], alternative='greater').pvalue)
        rows[path.stem] = row
    pooled = {name: sum(r[name]['early'] for r in rows.values()) / sum(r[name]['events'] for r in rows.values())
              for name in ('all', 'kinematics')}
    return dict(recordings=rows, pooled=pooled, mean_chance=float(np.mean([r['chance'] for r in rows.values()])),
                above_chance=sum(r['all']['share'] > r['chance'] for r in rows.values()))


def bands():
    units, rule = [], np.zeros(4)
    for path in track(isolated_recordings(), description='bands'):
        phis, stops = load(path)
        b, _ = changes(phis, stops, KIN)
        phases = H.rule_phases(b, phis, stops, 3, GOALS)
        rule += H.tally(phases)
        k = sum(map(len, phases))
        units.append([H.top(H.peaks([x[i] for x in b], stops), k, len(stops)).tolist() for i in range(H.BANDS)])
    total = np.sum(units, 0)
    return dict(rule=float(rule[0] / rule[1]), band_share=(total[:, 0] / total[:, 1]).tolist(), units=units)


def tempo():
    data = [load(p) for p in isolated_recordings()]
    out = {}
    for w in track(WARPS, description='tempo'):
        rule, heads = np.zeros(4), np.zeros((40, 4))
        for phis, stops in data:
            p, s = warp(phis, stops, w)
            mean, std = H.load_scale(KIN)
            scored = [H.change(x, mean, std) for x in p]
            b, d = [m[:, :n] for (m, _), n in zip(scored, s)], [q[:, :n] for (_, q), n in zip(scored, s)]
            rule += H.tally(H.rule_phases(b, p, s, 3, GOALS))
            heads += np.array([H.above(H.peaks([x[g] for x in d], s), len(s)) for g in range(40)])
        profile = heads[:, 0] / heads[:, 1]
        stops = np.array([x for _, st in data for x in st])
        stretched = np.round(stops * w).astype(int)
        out[str(w)] = dict(rule=float(rule[0] / rule[1]), profile=profile.tolist(), best=int(np.argsort(-profile)[0]),
                           chance=float(sum(np.sum(np.arange(1, x + 1) / x < H.ONSET) for x in stretched) / stretched.sum()))
    tuned = out['1.0']['best']
    horizons = H.mean_horizons()
    for v in out.values():
        v.update(tuned=v['profile'][tuned], best_share=v['profile'][v['best']], best_horizon_s=float(horizons[v['best']]))
    return dict(tempi=out, tuned=tuned, tuned_horizon_s=float(horizons[tuned]))


def per_element(phases):
    early = np.array([sum(p < H.ONSET for p in e) for e in phases])
    return np.column_stack([early, [len(e) for e in phases]])


def condition(b, d, phis, stops):
    ruled = H.rule_phases(b, phis, stops, 3, GOALS)
    k = sum(map(len, ruled))
    singles = []
    for g in range(40):
        pk = H.peaks([x[g] for x in d], stops)
        chosen = pk[np.argsort(-pk[:, 1], kind='stable')[:k]]
        singles.append(per_element([[p for i, _, p in chosen if i == e] for e in range(len(stops))]))
    chance = np.array([[int(np.ceil(s / 3)) - 1, s] for s in stops])
    return dict(rule=per_element(ruled), singles=np.array(singles), chance=chance, median_ms=float(np.median(stops) * 10),
                events=k, elements=len(stops))


def agent_condition(paths, w=1.0):
    phis, stops = [], []
    for path in paths:
        p, s = load(path, continuous=True)
        p = [x[:, KIN].astype(float) for x in p]
        if w != 1:
            p, s = warp(p, s, w)
        phis += p
        stops += s
    mean, std = H.load_scale(KIN)
    scored = [H.change(x, mean, std) for x in phis]
    return condition([m[:, :n] for (m, _), n in zip(scored, stops)], [q[:, :n] for (_, q), n in zip(scored, stops)], phis, stops)


def human_condition():
    mean, std = human.development_scale()
    b, d, phis, stops = [], [], [], []
    for p in human.HELDOUT:
        for collected, new in zip((b, d, phis, stops), human.segment_changes(p, mean, std)):
            collected += new
    return condition(b, d, phis, stops)


def share(t, axis=-1):
    return t[..., 0].sum(axis) / np.maximum(t[..., 1].sum(axis), 1)


def robustness():
    cont = REC / 'continuous'
    sources = {'trained': [cont / 'agent1.npz'], **{n: [REC / f'conditions/{n}.npz'] for n in CONDITIONS},
               'agent3': [cont / f'agent3_noise{k}.npz' for k in (2, 3, 4, 5, 1)] + [cont / 'agent3.npz'],
               'agent2': [cont / f'agent2_noise{k}.npz' for k in (3, 4, 5)]}
    res = {}
    for name in track(list(sources) + ['tempo_x0.5', 'tempo_x1.8', 'tempo_x3', 'humans'], description='robustness'):
        res[name] = (human_condition() if name == 'humans' else agent_condition([cont / 'agent1.npz'], float(name[7:]))
                     if name.startswith('tempo') else agent_condition(sources[name]))
    names = list(res)
    rule = np.array([share(res[n]['rule'], 0) for n in names])
    single = np.array([share(res[n]['singles']) for n in names])
    chance = np.array([share(res[n]['chance'], 0) for n in names])
    tuned = int(np.argmax(single[0]))
    rng = np.random.default_rng(42)
    diff = []
    for _ in range(BOOT):
        d = []
        for n in CONDITIONS:
            i = rng.integers(0, res[n]['elements'], res[n]['elements'])
            d.append(share(res[n]['rule'][i], 0) - share(res[n]['singles'][tuned][i], 0))
        diff.append(np.mean(d))
    point = float(np.mean([rule[names.index(n)] - single[names.index(n), tuned] for n in CONDITIONS]))
    wins = 0
    for _ in range(BOOT // 10):
        r, s = [], []
        for n, c in zip(names, chance):
            i = rng.integers(0, res[n]['elements'], res[n]['elements'])
            r.append(share(res[n]['rule'][i], 0) - c)
            s.append(share(res[n]['singles'][:, i]) - c)
        wins += min(r) > np.max(np.min(s, axis=0))
    worst_single = (single - chance[:, None]).min(0)
    horizons = H.mean_horizons()
    hum = res['humans']
    person = np.arange(hum['elements']) // 240
    draws = np.random.default_rng(0).integers(0, 10, (BOOT, 10))
    pick = lambda t, people: share(np.concatenate([t[person == p] for p in people]), 0)
    boot = np.array([[pick(hum['rule'], ps) - pick(hum['singles'][tuned], ps), pick(hum['singles'][tuned], ps) - pick(hum['chance'], ps)]
                     for ps in draws])
    return dict(
        tuned=tuned, tuned_horizon_s=float(horizons[tuned]),
        conditions={n: dict(median_ms=res[n]['median_ms'], elements=res[n]['elements'], events=res[n]['events'],
                            chance=float(c), rule=float(r), tuned=float(s[tuned]), best=float(s.max()), best_index=int(s.argmax()),
                            profile=s.tolist()) for n, r, s, c in zip(names, rule, single, chance)},
        rule_minus_tuned=dict(conditions=CONDITIONS, mean=point, ci95=np.percentile(diff, [2.5, 97.5]).tolist()),
        worst_case=dict(rule=float((rule - chance).min()), best_single=float(worst_single.max()),
                        best_single_index=int(worst_single.argmax()), singles=worst_single.tolist(),
                        rule_wins_in_bootstraps=wins / (BOOT // 10)),
        humans=dict(rule_minus_tuned_ci95=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                    tuned_minus_chance_ci95=np.percentile(boot[:, 1], [2.5, 97.5]).tolist()))


if __name__ == '__main__':
    out = dict(onsets=onsets(), bands=bands(), tempo=tempo(), robustness=robustness())
    (ROOT / 'results/transitions.json').write_text(json.dumps(out, indent=1))
    r = out['robustness']
    print(f"onsets: rule {out['onsets']['pooled']['all']:.3f} (kinematics {out['onsets']['pooled']['kinematics']:.3f}), "
          f"chance {out['onsets']['mean_chance']:.3f}, {out['onsets']['above_chance']}/17 recordings above chance")
    print(f"rule - tuned horizon on 8 bodies/workspaces {r['rule_minus_tuned']['mean']:+.4f} {r['rule_minus_tuned']['ci95']}")
    print(f"worst case above chance: rule {r['worst_case']['rule']:+.4f}, best single horizon {r['worst_case']['best_single']:+.4f}")

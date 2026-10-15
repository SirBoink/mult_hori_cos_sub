"""The unchanged rule on human sequential reaching.

    python analyze_humans.py   -> results/humans.json

onsets      held-out participants: share of events in the first third of each segment, vs the best single discount
            chosen on the same data at the same event count (participant bootstrap)
bands       per-band onset share at the rule's event count (cross-system comparison)
via_points  events relative to the nearest via-point passage
chunking    event rate inside the second target in sequences the published model analysis calls hierarchical vs flat
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

import arm
import horizons as H
import human

ROOT = Path(__file__).resolve().parent
PUBLISHED = ROOT / 'human_data/published_models'
sys.path.insert(0, str(PUBLISHED))
from src import analyses as published   # noqa: E402  (Cuevas Rivera & Kiebel's code, unmodified)

GOALS = arm.targets()


def participant(number, mean, std):
    bands, singles, phis, stops = human.segment_changes(number, mean, std)
    phases = H.rule_phases(bands, phis, stops, 3, GOALS)
    k = sum(map(len, phases))
    return dict(rule=H.tally(phases), chance=H.chance(stops),
                discounts=np.array([H.top(H.peaks([s[g] for s in singles], stops), k, len(stops)) for g in range(40)]),
                bands=np.array([H.top(H.peaks([b[i] for b in bands], stops), k, len(stops)) for i in range(H.BANDS)]),
                phases=phases, stops=stops)


def via_points(people_segments):
    """Signed frames from each event (and each frame) to the nearest via-point passage (start of segments 2 and 3)."""
    near_events, near_frames = [], []
    for phases, segments in people_segments:
        for events, (onset, length, passages) in zip(phases, segments):
            for phase in events:
                near_events.append(min((onset + round(phase * length) - q for q in passages), key=abs))
            near_frames += [min((onset + k - q for q in passages), key=abs) for k in range(1, length + 1)]
    e, f = np.abs(near_events), np.abs(near_frames)
    out = {f'within_{w}ms': dict(events=float(np.mean(e <= w // 10)), frames=float(np.mean(f <= w // 10))) for w in (50, 100, 150)}
    bins = np.arange(-40, 41, 5)
    out.update(bins_frames=bins.tolist(), events_histogram=np.histogram(near_events, bins)[0].tolist(),
               frames_histogram=np.histogram(near_frames, bins)[0].tolist())
    return out


def segment_passages(number):
    return [(onset, length, [o for o, _ in segments]) for _, segments in human.onset_segments(number) for onset, length in segments]


def labels(chalf, vpsoc, hiseq):
    """(participant, sequence) -> hierarchical (covered by the hierarchical model only), flat (covered by the flat
    via-point model) or neither; only combinations with negative mean halfway coarticulation, as published."""
    np.random.seed(0)
    means = chalf.groupby(['part', 'id_move'])['cross'].mean().reset_index().query('cross < 0')
    combos = set(zip(means['part'].astype(int), means['id_move'].astype(int)))
    data = chalf[[(int(p), int(m)) in combos for p, m in zip(chalf['part'], chalf['id_move'])]]
    hiseq_bad, vpsoc_bad = (set().union(*({(int(b[0]), int(b[1])) for b in t} for t in model))
                            for model in published.model_parts(data.copy(), vpsoc.copy(), hiseq.copy()))
    return {c: 'flat' if c not in vpsoc_bad else 'hierarchical' if c not in hiseq_bad else 'neither' for c in sorted(combos)}


def whole_movement_events(trials, mean, std):
    """Events from movement onset to trial end, percentiles pooled over one participant's trials."""
    bands = []
    for t in trials:
        multi, _ = H.change(human.pad(human.features(t['xy'] / 1000)), mean, std)
        bands.append(multi[:, t['bounds'][0]:len(t['xy']) - 1])
    refs = H.reference(np.concatenate(bands, axis=1))
    found = []
    for i, (t, b) in enumerate(zip(trials, bands)):
        a = t['bounds'][0]
        phi = human.pad(human.features(t['xy'] / 1000))[a:]
        found += [dict(e, trial=i) for e in H.events(H.percentiles(b, refs), phi, b.shape[1], human.CENTRES / 1000, -1, i)]
    return found


def auc(positive, negative):
    positive, negative = np.asarray(positive, float)[:, None], np.asarray(negative, float)[None, :]
    return float(((positive > negative) + .5 * (positive == negative)).mean())


def chunking():
    read = lambda name: pd.read_csv(PUBLISHED / f'{name}_chalf.csv', index_col=0)
    found = labels(read('data'), read('vpsoc'), read('hiseq'))
    trials = {}
    for p in track(list(human.DEVELOPMENT) + list(human.HELDOUT), description='chunking'):
        for id_move, xy, _ in human.read_trials(p):
            trials.setdefault(p, []).append(dict(id_move=id_move, xy=xy, bounds=human.movement_bounds(xy, human.SEQUENCES[id_move])))
    mean, std = human.development_scale()
    rates = {}
    for p, mine in trials.items():
        events = whole_movement_events(mine, mean, std)
        for m in range(6):
            if (p, m) not in found:
                continue
            ids = [i for i, t in enumerate(mine) if t['id_move'] == m]
            second = human.SEQUENCES[m][1]
            frames = sum(int(human.inside(mine[i]['xy'][mine[i]['bounds'][0]:len(mine[i]['xy']) - 1], second).sum()) for i in ids)
            hits = sum(int(human.inside(np.asarray(e['xy_m']) * 1000, second)) for e in events if e['trial'] in ids)
            rates[(p, m)] = dict(rate=hits / max(frames, 1), label=found[(p, m)])
    people = sorted({p for p, _ in rates})
    pick = lambda ps: ([rates[(p, m)]['rate'] for p in ps for m in range(6) if rates.get((p, m), {}).get('label') == 'hierarchical'],
                       [rates[(p, m)]['rate'] for p in ps for m in range(6) if rates.get((p, m), {}).get('label') == 'flat'])
    rng = np.random.default_rng(0)
    boot = [auc(*s) for s in (pick(rng.choice(people, len(people))) for _ in range(2000)) if len(s[0]) and len(s[1])]
    hierarchical, flat = pick(people)
    return dict(auc=auc(hierarchical, flat), ci95=np.percentile(boot, [2.5, 97.5]).tolist(), hierarchical=hierarchical, flat=flat)


if __name__ == '__main__':
    mean, std = human.development_scale()
    people = [participant(p, mean, std) for p in track(human.HELDOUT, description='held-out participants')]
    pool = lambda key: np.sum([p[key] for p in people], 0)
    rule, chance, discounts, band_t = pool('rule'), pool('chance'), pool('discounts'), pool('bands')
    best = int(np.nanargmax(discounts[:, 0] / discounts[:, 1]))
    t_rule = np.array([p['rule'] for p in people], float)
    t_best = np.array([p['discounts'][best] for p in people], float)
    draws = np.random.default_rng(0).integers(0, len(people), (10000, len(people)))
    diff = t_rule[draws][..., 0].sum(1) / t_rule[draws][..., 1].sum(1) - t_best[draws][..., 0].sum(1) / t_best[draws][..., 1].sum(1)
    onsets = dict(participants=len(people), rule=float(rule[0] / rule[1]), events_per_segment=float(rule[1] / rule[3]),
                  chance=float(chance[0] / chance[1]), best_discount=best, best_horizon_s=float(H.mean_horizons()[best]),
                  best_discount_share=float(discounts[best, 0] / discounts[best, 1]),
                  rule_minus_best=[float(rule[0] / rule[1] - discounts[best, 0] / discounts[best, 1]), *np.percentile(diff, [2.5, 97.5]).tolist()],
                  participants_above_chance=int(sum(p['rule'][0] / p['rule'][1] > p['chance'][0] / p['chance'][1] for p in people)))
    out = dict(onsets=onsets,
               bands=dict(rule=onsets['rule'], band_share=(band_t[:, 0] / band_t[:, 1]).tolist(), units=[p['bands'].tolist() for p in people]),
               via_points=via_points([(p['phases'], segment_passages(n)) for p, n in zip(people, human.HELDOUT)]),
               chunking=chunking())
    (ROOT / 'results/humans.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in onsets.items()}, indent=1))
    print('via points', out['via_points']['within_50ms'], 'chunking AUC', out['chunking']['auc'], out['chunking']['ci95'])

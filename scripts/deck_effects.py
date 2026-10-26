"""Forest plot of the main effects of the fixed rule (percentage points, 95% CI) for the review deck.

    python deck_effects.py   -> figures/deck_effects.png (600 dpi)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
read = lambda path: json.loads((ROOT / path).read_text())
transitions, transfer = read('results/transitions.json'), read('results/transfer.json')
via = read('submission/checks/sensitivity_results.json')['via_singles.json']['50ms']


def onset_excess():
    """Share of events in the first third minus event-weighted chance, bootstrapped over the 17 recordings."""
    rows = transitions['onsets']['recordings'].values()
    early, events, chance = (np.array([f(r) for r in rows], float) for f in
                             (lambda r: r['all']['early'], lambda r: r['all']['events'], lambda r: r['chance']))
    excess = lambda i: early[i].sum(-1) / events[i].sum(-1) - (chance[i] * events[i]).sum(-1) / events[i].sum(-1)
    draws = np.random.default_rng(0).integers(0, len(early), (10000, len(early)))
    return [excess(np.arange(len(early))), *np.percentile(excess(draws), [2.5, 97.5])]


robust, humans = transitions['robustness'], transitions['robustness']['conditions']['humans']
rows = [
    ('Agents', None, None),
    ('Onset vs chance, isolated reaches', onset_excess(), BLUE),
    ('Completion vs base, 8 targets', transfer['agent1']['orders96']['selectors_minus_frozen'], BLUE),
    ('Completion vs base, 16 targets', transfer['agent1']['sequences48']['selectors_minus_frozen'], BLUE),
    ('Completion vs base, agent 3', transfer['agent3']['multiscale_minus_frozen'], BLUE),
    ('Completion vs one horizon, agent 3', transfer['agent3']['multiscale_minus_single'], BLUE),
    ('Onset vs tuned horizon, new bodies', [robust['rule_minus_tuned']['mean'], *robust['rule_minus_tuned']['ci95']], BLUE),
    ('Humans (10 held out)', None, None),
    ('Onset vs horizon tuned on agents', [humans['rule'] - humans['tuned'], *robust['humans']['rule_minus_tuned_ci95']], ORANGE),
    ('Via-points vs best single timescale', via['rule_minus_best'], ORANGE),
]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'axes.linewidth': .7})
fig, ax = plt.subplots(figsize=(6.6, 4.2))
fig.subplots_adjust(left=.45, right=.77, top=.98, bottom=.13)
for y, (label, effect, color) in enumerate(rows):
    if effect is None:
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold', color=INK)
        continue
    mean, lo, hi = (100 * v for v in effect)
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle='round')
    ax.plot(mean, y, 'o', color=color, ms=6.5, mec='white', mew=1)
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', color=INK)
    ax.text(1.02, y, f'+{mean:.1f} [{lo:.1f}, {hi:.1f}]', transform=ax.get_yaxis_transform(), va='center', color=INK)
ax.axvline(0, color=MUTED, ls='--', lw=.8)
ax.set_ylim(len(rows) - .5, -.6)
ax.set_xlim(-5, 75)
ax.set_xticks([0, 20, 40, 60])
ax.set_yticks([])
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.set_xlabel('difference (percentage points), 95% CI', color=INK)
(ROOT / 'figures').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figures/deck_effects.png', dpi=600, facecolor='white')
for label, effect, _ in rows:
    if effect is not None:
        print(f'{label:40s} +{100 * effect[0]:.1f} [{100 * effect[1]:.1f}, {100 * effect[2]:.1f}]')

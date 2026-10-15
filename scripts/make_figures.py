"""All figures from results/*.json -> figures/*.png (600 dpi).

    python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import horizons as H

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
BLUE, ORANGE, INK, MUTED, LIGHT, GREY = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9d8d4', '#8f8e8a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'legend.frameon': False, 'figure.facecolor': 'white'})
read = lambda name: json.loads((ROOT / f'results/{name}.json').read_text())
transfer, transitions, humans, muscles = read('transfer'), read('transitions'), read('humans'), read('muscles')


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(OUT / f'{name}.png')


def pct_labels(ax, x, values, offset=.02, **kw):
    for xi, v in zip(x, values):
        ax.text(xi, v + offset, f'{v:.0%}', ha='center', fontsize=kw.get('fontsize', 8), color=kw.get('color', INK))


def fig_transfer():
    t = transfer['agent1']
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = [('frozen agent', LIGHT, 'frozen'), ('re-aim at goal', GREY, 'reaim'), ('multi-timescale subgoals', BLUE, 'selectors')]
    for gi, panel in enumerate(('orders96', 'sequences48')):
        for j, (label, color, key) in enumerate(bars):
            ax.bar(gi * 4 + j, t[panel][key], .8, color=color, label=label if gi == 0 else None)
            pct_labels(ax, [gi * 4 + j], [t[panel][key]])
        seeds = t[panel]['seeds']
        ax.plot(gi * 4 + 2 + np.linspace(-.2, .2, len(seeds)), seeds, 'o', ms=3.5, color=INK, mfc='white', mew=.7)
    ax.set_xticks([1, 5], [f"8 targets\n({t['orders96']['orders']} unseen orders)", f"16 targets\n({t['sequences48']['orders']} unseen sequences)"])
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel('sequences completed (zero-shot)')
    ax.legend(loc='upper left', fontsize=7.5, ncol=3, handlelength=1, columnspacing=.8)
    ax.set_title('Skill transfer to unseen sequences\n(dots: selector seeds)', fontsize=9, loc='left')
    save(fig, 'transfer')


def fig_transfer_detail():
    d = transfer['detail']
    keys = list(d['start_sets']) + ['pooled']
    groups = [d['start_sets'][k] for k in keys[:-1]] + [d['pooled']]
    labels = [f'starts {i + 1}' for i in range(len(keys) - 1)] + ['all']
    x, w = np.arange(len(keys)), .36
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    a, b, c, e = axes.flat
    for off, key, color, label in ((-w / 2, 'frozen', MUTED, 'frozen agent'), (w / 2, 'hierarchical', BLUE, 'agent + DDQN selector')):
        values = [g[key]['completion'] for g in groups]
        a.bar(x + off, values, w, color=color, label=label)
        pct_labels(a, x + off, values, fontsize=6.5, color=color)
        b.bar(x + off, [g[key]['duration'] for g in groups], w, yerr=[g[key]['duration_sem'] for g in groups], capsize=2,
              color=color, error_kw=dict(lw=.7))
    for xi, g in zip(x, groups):
        a.text(xi, max(g['frozen']['completion'], g['hierarchical']['completion']) + .13,
               f"+{100 * (g['hierarchical']['completion'] - g['frozen']['completion']):.0f}", ha='center', fontsize=6.5, color=BLUE)
        b.text(xi + w / 2, g['hierarchical']['duration'] + g['hierarchical']['duration_sem'] + .25,
               f"-{100 * (1 - g['hierarchical']['duration'] / g['frozen']['duration']):.0f}%", ha='center', fontsize=6.5, color=BLUE)
    a.set_ylim(0, 1.3)
    a.set_ylabel('sequences completed')
    a.legend(loc='upper left', fontsize=7, ncol=2)
    b.set_ylabel('failure-capped duration (s)')
    b.set_ylim(0, 8)
    pooled = d['pooled']
    for off, key, color in ((-w / 2, 'frozen', MUTED), (w / 2, 'hierarchical', BLUE)):
        share = np.array(pooled[key]['targets']) / sum(pooled[key]['targets'])
        c.bar(np.arange(9) + off, share, w, color=color)
    c.set_xticks(range(9))
    c.set_xlabel('targets completed (of 8)')
    c.set_ylabel('share of trials')
    shares = [g['hierarchical']['option_share'] for g in groups]
    e.bar(x, shares, .5, color=ORANGE)
    for xi, s in zip(x, shares):
        e.text(xi, s * 1.03, f'{s:.1%}', ha='center', fontsize=6.5, color=ORANGE)
    e.axhline(pooled['hierarchical']['option_share'], color=ORANGE, ls='--', lw=.8)
    e.set_ylim(0, max(shares) * 1.35)
    e.set_ylabel('selector decisions\nthat call an option')
    for ax in (a, b, e):
        ax.set_xticks(x, labels, fontsize=7)
    for ax, letter in zip(axes.flat, 'abcd'):
        ax.set_title(letter, loc='left', fontsize=9, fontweight='bold')
    fig.suptitle(f'Frozen agent vs hierarchical controller: 96 unseen orders x 10 starts, three selector seeds', fontsize=9, x=.02, ha='left')
    fig.tight_layout()
    save(fig, 'transfer_detail')


def fig_completion_time():
    pooled, a3 = transfer['detail']['pooled'], transfer['agent3']['duration']
    panels = [('Agent', [('Base SAC', pooled['frozen']['duration'], GREY), ('DDQN +\nmulti-timescale', pooled['hierarchical']['duration'], BLUE)]),
              ('Independently trained agent', [('Base SAC', a3['frozen'], GREY), ('DDQN +\nsingle horizon', a3['single'], LIGHT),
                                               ('DDQN +\nmulti-timescale', a3['multiscale'], BLUE)])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[2, 3]), sharey=True)
    for ax, (title, bars) in zip(axes, panels):
        base = bars[0][1]
        for i, (label, value, color) in enumerate(bars):
            ax.bar(i, value, .7, color=color, edgecolor=INK, lw=.5)
            ax.text(i, value + .15, f'{value:.2f} s\n' + ('reference' if i == 0 else f'{100 * (value / base - 1):+.1f}% vs base'),
                    ha='center', fontsize=7.5)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5)
        ax.tick_params(axis='x', length=0)
        ax.set_title(f'{title}: 96 unseen orders', fontsize=9, loc='left')
    axes[0].set_ylabel('failure-capped completion time (s)')
    axes[0].set_ylim(0, 8.5)
    fig.tight_layout()
    save(fig, 'completion_time')


def fig_subgoal_torque():
    torque = muscles['torque']
    names = list(torque)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for j, (ax, joint) in enumerate(zip(axes, ('Shoulder', 'Elbow'))):
        for i, n in enumerate(names):
            ax.bar(i - .17, torque[n]['positive'][j], .32, color=BLUE, label='positive muscle torque' if i == 0 else None)
            ax.bar(i + .17, torque[n]['negative'][j], .32, color=ORANGE, label='negative muscle torque' if i == 0 else None)
            ax.plot(i, torque[n]['net'][j], 'o', color=INK, ms=4.5, zorder=4, label='net muscle torque' if i == 0 else None)
        ax.axhline(0, color=INK, lw=.7)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel('option arrival at subgoal')
        ax.set_title(joint, fontsize=9)
    axes[0].set_ylabel('muscle torque (N m)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7.5, bbox_to_anchor=(.5, 1.06))
    fig.tight_layout()
    save(fig, 'subgoal_torque')


def fig_independent_agent():
    t = transfer['agent3']
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    bars = [('frozen agent', t['frozen'], LIGHT, None), ('single tuned\nhorizon', t['single'], GREY, t['single_seeds']),
            ('multi-timescale', t['multiscale'], BLUE, t['multiscale_seeds'])]
    for i, (label, value, color, seeds) in enumerate(bars):
        ax.bar(i, value, .7, color=color)
        pct_labels(ax, [i], [value], offset=.03)
        if seeds:
            ax.plot(i + np.linspace(-.2, .2, len(seeds)), seeds, 'o', ms=3.5, color=INK, mfc='white', mew=.7)
    d = t['multiscale_minus_single']
    ax.annotate(f'+{100 * d[0]:.1f} points\n[95% CI {100 * d[1]:.1f}, {100 * d[2]:.1f}]', (1.5, 1.04), ha='center', fontsize=7.5)
    ax.set_xticks(range(3), [b[0] for b in bars], fontsize=8)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('unseen orders completed')
    ax.set_title('Transfer to an independently trained agent\n(DDQN on random orders; dots: selector seeds)', fontsize=9, loc='left')
    save(fig, 'independent_agent')


def fig_horizon_across_systems():
    lags = H.band_lags()
    systems = [(transitions['bands'], BLUE, 'agents'), (humans['bands'], ORANGE, 'humans (held out)')]
    chance = transitions['onsets']['mean_chance']
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    rng = np.random.default_rng(0)
    for data, color, label in systems:
        u = np.array(data['units'], float)
        draws = rng.integers(0, len(u), (2000, len(u)))
        lo, hi = np.nanpercentile(u[draws][..., 0].sum(1) / u[draws][..., 1].sum(1), [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=color, alpha=.18, lw=0)
        ax.plot(lags, data['band_share'], color=color, lw=1.4, label=f'{label}: single horizon')
        ax.axhline(data['rule'], color=color, ls=':', lw=1.6, label=f'{label}: multi-timescale rule')
    for (src, color, _), (dst, _, _) in ((systems[0], systems[1]), (systems[1], systems[0])):
        g = int(np.nanargmax(src['band_share']))
        y0, y1 = src['band_share'][g], dst['band_share'][g]
        ax.plot(lags[g], y0, 'v', color=color, ms=6)
        ax.annotate('', (lags[g], y1), (lags[g], y0), arrowprops=dict(arrowstyle='->', color=color, lw=1))
        ax.plot(lags[g], y1, 'x', color=color, ms=7, mew=1.5)
    ax.axhline(chance, color=MUTED, ls='--', lw=.8)
    ax.text(.011, chance - .06, 'chance', color=MUTED, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('single predictive horizon (s)')
    ax.set_ylabel('transitions marked at movement onset\n(share of events in first third)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=7.5, ncol=2)
    ax.set_title('A horizon tuned in one system fails in the other (arrows);\nthe fixed rule works in both', fontsize=9, loc='left')
    save(fig, 'horizon_across_systems')


def fig_tempo():
    t = transitions['tempo']
    keys = list(t['tempi'])
    w = [float(k) for k in keys]
    get = lambda key: [t['tempi'][k][key] for k in keys]
    horizons = get('best_horizon_s')
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(w, get('rule'), 'o-', color=BLUE, ms=6, lw=2, label='multi-timescale rule (fixed)')
    ax.plot(w, get('tuned'), 's-', color=MUTED, ms=5, lw=1.4, label='single horizon tuned at 1x')
    ax.plot(w, get('chance'), '--', color=MUTED, lw=.8)
    ax.text(2.2, get('chance')[-1] - .07, 'chance', color=MUTED, fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(w)
    ax.minorticks_off()
    ax.set_xticklabels([f'{x:g}x\n{h:.2f} s' for x, h in zip(w, horizons)])
    ax.set_xlabel('movement duration (x normal); best single horizon')
    ax.set_ylabel('share of events in first third')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Change the tempo: the best horizon moves {max(horizons) / min(horizons):.0f}-fold,\n'
                 'the tuned horizon drops, the rule holds', fontsize=9, loc='left')
    save(fig, 'tempo')


def fig_robustness():
    r = transitions['robustness']
    c = r['conditions']
    order = ['trained', 'mass_x1.15', 'mass_x1.3', 'forearm_x1.3', 'forearm_x1.5', 'forearm_x1.8', 'workspace_x0.8',
             'workspace_x0.9', 'workspace_x1.1', 'agent2', 'agent3', 'tempo_x0.5', 'tempo_x1.8', 'tempo_x3', 'humans']
    label = lambda n: n.replace('_x', ' ×').replace('agent', 'agent ')
    x = np.arange(len(order))
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.4), gridspec_kw=dict(height_ratios=[1.1, 1]))
    for key, color, text in (('rule', BLUE, 'multi-timescale rule (fixed)'),
                             ('tuned', MUTED, f"single horizon tuned once on the trained agent ({r['tuned_horizon_s'] * 1000:.0f} ms)")):
        a.plot(x, [100 * (c[n][key] - c[n]['chance']) for n in order], 'o-', color=color, ms=4, lw=1.3, label=text)
    for i, n in enumerate(order):
        if n.split('_')[0] in ('mass', 'forearm', 'workspace'):
            a.axvspan(i - .4, i + .4, color='#f2f2f2', zorder=0)
    a.axhline(0, color=INK, lw=.6)
    a.set_xticks(x, [label(n) for n in order], rotation=45, ha='right', fontsize=7)
    a.set_ylabel('events at movement onset\n(points above chance)')
    a.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1.2), ncol=2)
    m = r['rule_minus_tuned']
    a.set_title(f"a  Heavier limbs and rescaled workspaces (shaded): rule - tuned horizon {100 * m['mean']:+.1f} points "
                f"[{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}]", fontsize=8, loc='left', pad=22)
    worst = 100 * np.array(r['worst_case']['singles'])
    b.bar(H.mean_horizons() * 1000, worst, width=H.mean_horizons() * 1000 * .1, color=GREY, label='each single horizon')
    b.axhline(100 * r['worst_case']['rule'], color=BLUE, lw=1.6, label='multi-timescale rule')
    b.axhline(0, color=INK, lw=.6)
    b.set_xscale('log')
    b.set_xlabel('single horizon (ms)')
    b.set_ylabel('worst case over all 15 conditions\n(points above chance)')
    b.legend(fontsize=7, loc='lower right')
    b.set_title('b  Worst case: the rule stays above every one of the 40 single horizons', fontsize=8, loc='left')
    fig.tight_layout()
    save(fig, 'robustness')


def fig_humans_onsets():
    o = humans['onsets']
    rule, best, diff = o['rule'], o['best_discount_share'], o['rule_minus_best']
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    names = ['chance', 'best single discount\n(tuned in hindsight)', 'multi-timescale rule\n(fixed in advance)']
    for x, (value, color) in enumerate([(o['chance'], LIGHT), (best, GREY), (rule, ORANGE)]):
        ax.bar(x, value, .7, color=color)
        ax.text(x, value + .015, f'{value:.0%}', ha='center', fontsize=9)
    ax.annotate(f'+{diff[0] * 100:.0f} points\n[95% CI {diff[1] * 100:.0f}, {diff[2] * 100:.0f}]', (2, rule + .07), ha='center', fontsize=8)
    ax.set_xticks(range(3), names, fontsize=7.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, .75)
    ax.set_ylabel('segment onsets marked\n(share of events in first third)')
    ax.set_title(f"{o['participants']} held-out humans, via-point task:\nthe fixed rule beats the best single discount",
                 fontsize=9, loc='left')
    save(fig, 'humans_onsets')


def fig_humans_transitions():
    v, ch = humans['via_points'], humans['chunking']
    bins = np.array(v['bins_frames']) * 10
    centres = (bins[:-1] + bins[1:]) / 2
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    events, frames = np.array(v['events_histogram'], float), np.array(v['frames_histogram'], float)
    a.bar(centres, frames / frames.sum(), 50, color=LIGHT, label='all frames')
    a.step(centres, events / events.sum(), where='mid', color=ORANGE, lw=1.6, label='rule events')
    a.axvspan(-50, 50, color=ORANGE, alpha=.08, lw=0)
    w = v['within_50ms']
    a.text(.03, .95, f"{w['events']:.0%} of events within 50 ms\n({w['frames']:.0%} of frames)", transform=a.transAxes, va='top', fontsize=7.5)
    a.set_xlabel('time from nearest via-point passage (ms)')
    a.set_ylabel('share')
    a.legend(fontsize=7, loc='upper right')
    a.set_title('a  Events mark segment transitions', fontsize=9, loc='left')
    rng = np.random.default_rng(1)
    for i, (key, color) in enumerate((('flat', GREY), ('hierarchical', ORANGE))):
        values = np.array(ch[key]) * 100
        b.plot(i + rng.uniform(-.15, .15, len(values)), values, 'o', ms=3, color=color, alpha=.7)
        b.plot([i - .25, i + .25], [np.median(values)] * 2, color=INK, lw=1.5)
    b.set_xticks([0, 1], [f"flat\n(n = {len(ch['flat'])})", f"hierarchical\n(n = {len(ch['hierarchical'])})"])
    b.set_xlim(-.6, 1.6)
    b.set_ylabel('events per 100 frames\ninside the second target')
    b.set_title(f"b  Chunked sequences: AUC {ch['auc']:.2f}\n[95% CI {ch['ci95'][0]:.2f}, {ch['ci95'][1]:.2f}]", fontsize=9, loc='left')
    fig.tight_layout()
    save(fig, 'humans_transitions')


def fig_muscle_patterns():
    d = muscles['decoding']
    fig, ax = plt.subplots(figsize=(4, 3.2))
    values = [d['muscles'], d['kinematics']]
    ax.bar([0, 1], values, .6, color=[BLUE, GREY])
    pct_labels(ax, [0, 1], values)
    ax.violinplot([d['null']], [0], widths=.5, showextrema=False)
    ax.axhline(d['chance'], color=MUTED, ls='--', lw=.8)
    ax.text(1.35, d['chance'] + .01, 'chance', color=MUTED, fontsize=7.5)
    ax.set_xticks([0, 1], ['six muscle\nactivations', 'position-free\nkinematics'])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-.6, 1.7)
    ax.set_ylim(0, .7)
    ax.set_ylabel('reach decoded at transition events')
    ax.set_title(f"Transition events carry reach-specific muscle patterns\n({d['events']} events; permutation p = {d['p']:.3f})",
                 fontsize=9, loc='left')
    save(fig, 'muscle_patterns')


if __name__ == '__main__':
    for make in (fig_transfer, fig_transfer_detail, fig_completion_time, fig_subgoal_torque, fig_independent_agent,
                 fig_horizon_across_systems, fig_tempo, fig_robustness, fig_humans_onsets, fig_humans_transitions,
                 fig_muscle_patterns):
        make()

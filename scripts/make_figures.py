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

ROOT = Path(__file__).resolve().parent.parent
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


def fig_completion_time_independent():
    a3 = transfer['agent3']['duration']
    bars = [('Base SAC', a3['frozen'], GREY), ('DDQN +\nsingle horizon', a3['single'], LIGHT), ('DDQN +\nmulti-timescale', a3['multiscale'], BLUE)]
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    for i, (label, value, color) in enumerate(bars):
        ax.bar(i, value, .7, color=color, edgecolor=INK, lw=.5)
        ax.text(i, value + .15, f'{value:.2f} s\n' + ('reference' if i == 0 else f'{100 * (value / a3["frozen"] - 1):+.1f}% vs base'),
                ha='center', fontsize=7.5)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('failure-capped completion time (s)')
    ax.set_ylim(0, 8.5)
    fig.tight_layout()
    save(fig, 'completion_time_independent')


def completion_panel(a):
    """Held-out agent on unseen orders: dots are orders, markers mean and 95% interval (Figure 1 of the draft)."""
    t = transfer['agent3']
    rng = np.random.default_rng(0)
    for i, (key, label, color) in enumerate((('frozen', 'Base', MUTED), ('single', 'Single', GREY), ('multiscale', 'DDQN', BLUE))):
        o = t['orders'][key]
        m, (lo, hi) = 100 * np.mean(o['means']), [100 * c for c in o['ci']]
        a.scatter(i + rng.uniform(-.2, .2, len(o['means'])), 100 * np.array(o['means']), s=8, color=color, alpha=.3, lw=0)
        a.errorbar(i + .3, m, [[m - lo], [hi - m]], fmt='o', ms=6, color=color, capsize=3, lw=1.2)
        n = 10 * len(o['means'])  # trials per run; selector controllers show the mean over their seeds
        a.text(i, 108, f"{m:.2f}%\n{o['successes'] * n / o['trials']:.0f}/{n}", ha='center', fontsize=7.5, color=color)
    a.set_xticks(range(3), ['Base', 'Single', 'DDQN'])
    a.tick_params(axis='x', length=0)
    a.set_ylim(-5, 122)
    a.set_yticks(range(0, 101, 20))
    a.set_ylabel('sequence completion (%)')


def fig_completion_orders():
    t = transfer['agent3']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3), gridspec_kw=dict(width_ratios=[1, 1.1]))
    completion_panel(a)
    for y, key, label in ((1, 'multiscale_minus_frozen', 'versus base'), (0, 'multiscale_minus_single', 'versus single')):
        d, lo, hi = [100 * v for v in t[key]]
        b.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='o', ms=6, color=BLUE, capsize=3, lw=1.2)
        b.text(d, y + .15, f'+{d:.1f} pp  [{lo:.1f}, {hi:.1f}]', ha='center', fontsize=7.5)
    b.axvline(0, color=MUTED, lw=.7, ls=':')
    b.set_yticks([1, 0], ['versus base', 'versus single'])
    b.tick_params(axis='y', length=0)
    b.set_ylim(-.5, 1.5)
    b.set_xlim(-5, 80)
    b.set_xlabel('paired completion gain (percentage points)')
    fig.tight_layout()
    save(fig, 'completion_orders')


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
        ax.set_xlabel('subgoal')
        ax.set_title(joint, fontsize=9)
    axes[0].set_ylabel('muscle torque (N m)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7.5, bbox_to_anchor=(.5, 1.06))
    fig.tight_layout()
    save(fig, 'subgoal_torque')


def fig_subgoal_net_torque():
    torque = muscles['torque']
    names = list(torque)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for j, (dx, hatch) in enumerate(((-.2, None), (.2, '////'))):
        get = lambda k: np.array([torque[n][k][j] for n in names])
        ax.bar(x + dx, get('positive'), .38, color=BLUE, hatch=hatch, edgecolor='white', lw=0)
        ax.bar(x + dx, get('negative'), .38, color=ORANGE, hatch=hatch, edgecolor='white', lw=0)
        ax.plot(x + dx, get('net'), 'o', color=INK, ms=3.5, zorder=4)
    ax.axhline(0, color=INK, lw=.7)
    ax.set_xticks(x, names)
    ax.set_xlabel('subgoal')
    ax.set_ylabel('muscle torque (N m)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Line2D([], [], marker='o', color=INK, ls='', ms=3.5),
               plt.Rectangle((0, 0), 1, 1, fc=GREY), plt.Rectangle((0, 0), 1, 1, fc=GREY, hatch='////', ec='white', lw=0)]
    ax.legend(handles, ['positive', 'negative', 'net', 'shoulder', 'elbow'], loc='upper center', bbox_to_anchor=(.5, 1.2),
              ncol=5, fontsize=6.5, handlelength=1, columnspacing=.7)
    save(fig, 'subgoal_net_torque')


def torque_panel(ax):
    """Light bars: total muscle torque at each joint; dark bars: |net|; labels: share cancelled by opposing muscles."""
    torque = muscles['torque']
    names = list(torque)
    x = np.arange(len(names))
    for j, (dx, color, joint) in enumerate(((-.2, BLUE, 'shoulder'), (.2, ORANGE, 'elbow'))):
        t = {k: np.array([torque[n][k][j] for n in names]) for k in ('positive', 'negative', 'net')}
        total, net = t['positive'] - t['negative'], np.abs(t['net'])
        ax.bar(x + dx, total, .38, color=color, alpha=.3)
        ax.bar(x + dx, net, .38, color=color, label=joint)
        for xi, tot, n in zip(x + dx, total, net):
            ax.text(xi, tot + 1, f'{1 - n / tot:.0%}', ha='center', fontsize=5, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 75)
    ax.set_xlabel('subgoal')
    ax.set_ylabel('muscle torque (N m)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=.3))
    ax.legend(handles, ['shoulder |net|', 'elbow |net|', 'cancelled (%)'], loc='upper center', bbox_to_anchor=(.5, 1.22),
              ncol=3, fontsize=6.5, handlelength=1, columnspacing=.8)


def fig_torque_cancellation():
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    torque_panel(ax)
    save(fig, 'torque_cancellation')


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
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    via_point_panel(a)
    a.set_title('a  Transition events cluster at via-points', fontsize=9, loc='left')
    hierarchy_panel(b)
    b.set_title(f"b  Hierarchical sequences: AUC {humans['chunking']['auc']:.2f}", fontsize=9, loc='left')
    fig.tight_layout()
    save(fig, 'humans_transitions')


def via_point_panel(a, fontsize=7.5):
    """Human transition events and all frames, relative to the nearest via-point passage."""
    v = humans['via_points']
    bins = np.array(v['bins_frames']) * 10
    centres = (bins[:-1] + bins[1:]) / 2
    events, frames = np.array(v['events_histogram'], float), np.array(v['frames_histogram'], float)
    a.bar(centres, frames / frames.sum(), 50, color=LIGHT, label='all frames')
    a.step(centres, events / events.sum(), where='mid', color=ORANGE, lw=1.6, label='transition events')
    a.axvspan(-50, 50, color=ORANGE, alpha=.08, lw=0)
    w = v['within_50ms']
    a.text(.03, .95, f"{w['events']:.0%} of events within 50 ms\n({w['frames']:.0%} of frames)", transform=a.transAxes, va='top', fontsize=fontsize)
    a.set_xlabel('time from nearest via-point passage (ms)')
    a.set_ylabel('share')
    a.legend(fontsize=fontsize - .5, loc='upper right')


def hierarchy_panel(b):
    """Our rule's events inside the second target, in human sequences fit by the authors' flat (vpSOC) or
    hierarchical (HiSeq) optimal-control model; violins: distribution over sequences, boxes: quartiles and median."""
    ch = humans['chunking']
    groups = [np.array(ch[key]) * 100 for key in ('flat', 'hierarchical')]
    violins = b.violinplot(groups, [0, 1], widths=.75, showextrema=False)
    for body, color in zip(violins['bodies'], (GREY, ORANGE)):
        body.set(facecolor=color, edgecolor=color, alpha=.35)
    b.boxplot(groups, positions=[0, 1], widths=.18, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor='white', color=INK, lw=.8), medianprops=dict(color=INK, lw=1.2),
              whiskerprops=dict(color=INK, lw=.8), capprops=dict(color=INK, lw=.8))
    b.set_xticks([0, 1], [f"flat model\n(vpSOC, n = {len(ch['flat'])})", f"hierarchical\nmodel (HiSeq,\nn = {len(ch['hierarchical'])})"], fontsize=6.5)
    b.set_xlim(-.6, 1.6)
    b.set_xlabel("human sequences fit by the\nauthors' optimal-control model", fontsize=6.5)
    b.set_ylabel("our rule's transition events\nper 100 frames in 2nd target")


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


def workflow_panel(ax):
    """The workflow diagram of the original draft, redrawn: four boxes, read clockwise from the top left."""
    boxes = {(0, 1): 'Fingertip position,\nvelocity & muscle\nactivations', (1, 1): 'Future predictions\nover 40\ntimescales',
             (1, 0): 'Changes shared by\nneighboring bands\nmark transitions', (0, 0): 'Selector chooses\noptions to subgoals\nor primitive actions'}
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#f2f2f2', ec=MUTED, lw=.6))
    w, h = .45, .37
    centre = lambda c, r: (.245 + .51 * c, .235 + .53 * r)
    for (c, r), text in boxes.items():
        x, y = centre(c, r)
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc='white', ec=INK, lw=.6))
        ax.text(x, y, text, ha='center', va='center', fontsize=5.3, color=INK, linespacing=1.05)
    arrow = dict(arrowstyle='->', color=MUTED, lw=.8)
    ax.annotate('', centre(1, 1) - np.array([w / 2, 0]), centre(0, 1) + np.array([w / 2, 0]), arrowprops=arrow)
    ax.annotate('', centre(1, 0) + np.array([0, h / 2]), centre(1, 1) - np.array([0, h / 2]), arrowprops=arrow)
    ax.annotate('', centre(0, 0) + np.array([w / 2, 0]), centre(1, 0) - np.array([w / 2, 0]), arrowprops=arrow)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def fig_submission():
    """COSYNE abstract: a, b results; c workflow; d torque at subgoal arrival (7.5 in wide, the text width)."""
    fig = plt.figure(figsize=(7.5, 2.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, 1.45], height_ratios=[1.35, 1.2], wspace=.42, hspace=.6)
    a, b = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    c, d = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    # a  held-out agent, 96 unseen orders x 10 start postures
    t = transfer['agent3']
    rng = np.random.default_rng(0)
    for i, (key, color) in enumerate((('frozen', MUTED), ('single', GREY), ('multiscale', BLUE))):
        o = t['orders'][key]
        m, (lo, hi) = 100 * np.mean(o['means']), [100 * v for v in o['ci']]
        a.scatter(i + rng.uniform(-.2, .2, len(o['means'])), 100 * np.array(o['means']), s=4, color=color, alpha=.3, lw=0)
        a.errorbar(i + .3, m, [[m - lo], [hi - m]], fmt='o', ms=4, color=color, capsize=2, lw=1)
        a.text(i, 106, f'{m:.0f}%', ha='center', fontsize=7.5, color=color)
    a.set_xticks(range(3), ['no\nsubgoals', 'single\ntimescale', 'multi-\nscale'], fontsize=6.5)
    a.tick_params(axis='x', length=0)
    a.set_ylim(-5, 118)
    a.set_yticks(range(0, 101, 25))
    a.set_ylabel('sequences completed (%)')
    a.set_title('a  Held-out agent', fontsize=8.5, loc='left')
    onset_panel(b)
    b.set_title(f"b  Humans ({humans['onsets']['participants']} held out)", fontsize=8.5, loc='left')
    workflow_panel(c)
    c.set_title('c  Workflow', fontsize=8.5, loc='left')
    torque_panel(d)
    handles, _ = d.get_legend_handles_labels()
    d.legend(handles + [plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=.3)], ['shoulder |net|', 'elbow |net|', 'cancelled (%)'],
             loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=3, fontsize=6, handlelength=.8, columnspacing=.5)
    d.set_title('d  Torque at subgoal arrival', fontsize=8.5, loc='left', pad=16)
    save(fig, 'submission')


def onset_panel(b):
    """Held-out humans: transitions in the onset window (first third) of a segment, against chance."""
    o, tuned = humans['onsets'], transitions['robustness']['conditions']['humans']['tuned']
    bars = [('best for\nagents', tuned, LIGHT), ('best single\n(hindsight)', o['best_discount_share'], GREY),
            ('multi-\nscale', o['rule'], ORANGE)]
    for x, (label, value, color) in enumerate(bars):
        b.bar(x, 100 * value, .65, color=color)
        b.text(x, 100 * value + 1.5, f'{value:.0%}', ha='center', fontsize=7.5)
    b.axhline(100 * o['chance'], color=MUTED, ls='--', lw=.8)
    b.text(-.45, 100 * o['chance'] + 1.5, 'chance', color=MUTED, fontsize=7)
    diff = [100 * v for v in o['rule_minus_best']]
    b.text(2, 100 * o['rule'] + 8, f'+{diff[0]:.0f} [{diff[1]:.0f}, {diff[2]:.0f}]', ha='center', fontsize=7)
    b.set_xticks(range(3), [x[0] for x in bars], fontsize=6.5)
    b.tick_params(axis='x', length=0)
    b.set_ylim(0, 75)
    b.set_ylabel('transitions in onset window (%)')


def fig_draft():
    """Supervisor's draft layout as one figure: a completion, b torque, c workflow, d-f human reaching."""
    fig = plt.figure(figsize=(7.5, 4.1))
    top, bottom = fig.add_gridspec(2, 1, hspace=.75)
    a, b, c = (fig.add_subplot(s) for s in top.subgridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=.45))
    d, e, f = (fig.add_subplot(s) for s in bottom.subgridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=.6))
    completion_panel(a)
    for label in a.texts:
        label.set_fontsize(6)  # percentage labels must fit above the narrow columns
    torque_panel(b)
    workflow_panel(c)
    via_point_panel(d, fontsize=7)
    d.legend(fontsize=6.5, loc='center right')
    d.set_xlabel('time from via-point passage (ms)')
    hierarchy_panel(e)
    e.set_xticks([0, 1], ['flat\nmodel', 'hierarchical\nmodel'], fontsize=6.5)
    e.set_xlabel('model explaining the sequence', fontsize=7)
    e.set_ylabel("our rule's boundaries in the\n2nd target (per 100 frames)", fontsize=7.5)
    onset_panel(f)
    f.set_xticks(range(3), ['agent-\ntuned', 'best\nsingle', 'multi-\nscale'], fontsize=6.5)
    f.set_ylabel('onset window (%)')
    titles = ['a  Held-out agent', 'b  Torque at subgoal arrival', 'c  Workflow', 'd  Humans: via-points',
              f"e  Authors' models: AUC {humans['chunking']['auc']:.2f}", 'f  Humans: onsets']
    for ax, title in zip((a, b, c, d, e, f), titles):
        ax.set_title(title, fontsize=8.5, loc='left', pad=18 if ax in (a, b) else 6)
    save(fig, 'draft_submission')


if __name__ == '__main__':
    for make in (fig_transfer, fig_transfer_detail, fig_completion_time, fig_completion_time_independent, fig_completion_orders, fig_subgoal_torque, fig_subgoal_net_torque, fig_torque_cancellation, fig_independent_agent,
                 fig_horizon_across_systems, fig_tempo, fig_robustness, fig_humans_onsets, fig_humans_transitions,
                 fig_muscle_patterns, fig_submission, fig_draft):
        make()

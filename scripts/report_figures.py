"""Figures for the Phase 1 report -> report_doc/latex/figures/*.png (600 dpi, text in Adobe Garamond Pro).

    python scripts/report_figures.py              all figures
    python scripts/report_figures.py fig_tempo    one figure

Every number comes from results/*.json, submission/checks/sensitivity_results.json or the saved recordings. Nothing is
recomputed here except the multi-timescale map of one practised cycle and one human trial, which use horizons.py and
the published analysis code unchanged.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import horizons as H

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'report_doc/latex/figures'
for _font in sorted((ROOT / 'report_doc/adobe-garamond-pro-2').glob('*.otf')):
    font_manager.fontManager.addfont(str(_font))
BLUE, ORANGE = '#2a78d6', '#eb6834'                    # agents / humans (and the multi-timescale rule in humans)
INK, SOFT, MUTED, HAIR = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
PALE, MID, DARK = '#d9d8d4', '#a9a8a3', '#6d6c68'      # baselines, light to dark
WIDE, HALF = 5.9, 3.7                                  # inches; the report's text width is 15 cm
ORDER = (0, 3, 7, 5, 6, 1, 2, 4)                       # practised target order
CENTRE = np.array([-0.017, 0.454])                     # fingertip at the start posture (m)
TARGETS = CENTRE + 0.1 * np.column_stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)])
L1, L2 = 0.309, 0.333                                  # upper arm and forearm (m)

S = 1.15                                               # Adobe Garamond Pro is small on the body, so every size is scaled up
plt.rcParams.update({
    'font.family': ['Adobe Garamond Pro', 'DejaVu Serif'], 'font.size': 8 * S, 'axes.labelsize': 8 * S, 'axes.titlesize': 8.5 * S,
    'xtick.labelsize': 7.5 * S, 'ytick.labelsize': 7.5 * S, 'legend.fontsize': 7.5 * S, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': SOFT, 'ytick.color': SOFT, 'axes.linewidth': .6,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.unicode_minus': True, 'lines.solid_capstyle': 'round'})
read = lambda name: json.loads((ROOT / f'results/{name}.json').read_text())
transfer, transitions, humans, muscles = read('transfer'), read('transitions'), read('humans'), read('muscles')
checks = json.loads((ROOT / 'submission/checks/sensitivity_results.json').read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(OUT / f'{name}.png')


def letter(ax, text, x=-.14, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9.5 * S, fontweight='bold', va='bottom', ha='left')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=HAIR, lw=.5)
    ax.set_axisbelow(True)


def pct(v, digits=0):
    return f'{100 * v:.{digits}f}%'


def percent_axis(ax, ticks, axis='y'):
    labels = [f'{100 * t:.0f}%' for t in ticks]
    (ax.set_yticks if axis == 'y' else ax.set_xticks)(ticks, labels)


# ---------------------------------------------------------------- shared drawing helpers

def draw_arm(ax, hand=CENTRE, color=MID, lw=4.5):
    """Upper arm and forearm (elbow-flexed solution) with the shoulder at the origin; returns the elbow."""
    r2 = (hand ** 2).sum()
    q2 = np.arccos(np.clip((r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
    q1 = np.arctan2(hand[1], hand[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    elbow = L1 * np.array([np.cos(q1), np.sin(q1)])
    ax.plot([0, elbow[0], hand[0]], [0, elbow[1], hand[1]], color=color, lw=lw, solid_capstyle='round', zorder=1)
    ax.plot([0, elbow[0]], [0, elbow[1]], 'o', color='white', mec=color, mew=1.2, ms=5, zorder=2)
    return elbow


def draw_targets(ax, labels=True, size=.02):
    for k, (x, y) in enumerate(TARGETS):
        ax.add_patch(Circle((x, y), size, fc='white', ec=MUTED, lw=.7, zorder=2))
        if labels:
            ax.text(*(TARGETS[k] + .38 * (TARGETS[k] - CENTRE)), f'T{k}', ha='center', va='center', fontsize=6.5 * S, color=SOFT)


def one_cycle():
    """One practised cycle of agent 1 (motor-noise seed 1): features, change percentiles, events, arrivals."""
    with np.load(ROOT / 'data/recordings/continuous/agent1_noise1.npz') as d:
        keep = (d['hit'] > 0) & (d['repeat'] >= 1)
        phis, stops = d['phi'][keep], d['hit'][keep]
    mean, std = H.load_scale()
    multi = [H.change(p, mean, std)[0] for p in phis]
    refs = H.reference(np.concatenate([m[:, :s] for m, s in zip(multi, stops)], axis=1))
    arrivals = np.cumsum(stops[:8])
    steps = int(arrivals[-1]) + 5
    pct_map = H.percentiles(multi[0][:, :steps], refs)
    events = H.events(pct_map, phis[0], steps, TARGETS, -1, 0)
    return phis[0][:steps + 1], pct_map, events, arrivals


def draw_map(ax, pct_map, events):
    lags = H.band_lags()
    r = (lags[1] / lags[0]) ** .5
    edges = np.geomspace(lags[0] / r, lags[-1] * r, len(lags) + 1)
    shown = np.where(np.isfinite(pct_map), pct_map, 0)
    mesh = ax.pcolormesh(np.arange(pct_map.shape[1] + 1) * H.DT, edges, shown, cmap='Blues', vmin=.5, vmax=1,
                         shading='flat', rasterized=True)
    for e in events:
        ax.plot(e['step'] * H.DT, lags[e['peak_band']], 'o', ms=4, color=ORANGE, mec='white', mew=.6, clip_on=False, zorder=5)
    ax.set_yscale('log')
    ax.set_ylim(edges[0], edges[-1])
    ax.set_yticks([.01, .1, 1], ['10 ms', '100 ms', '1 s'])
    ax.set_xlim(0, pct_map.shape[1] * H.DT)
    return mesh


def human_example(participant=411, trial=0):
    """One held-out human trial (fingertip, mm) and the rule's events in it, percentiles pooled over the participant."""
    import analyze_humans as A
    import human
    mean, std = human.development_scale()
    trials = [dict(id_move=m, xy=xy, bounds=human.movement_bounds(xy, human.SEQUENCES[m])) for m, xy, _ in human.read_trials(participant)]
    events = [e for e in A.whole_movement_events(trials, mean, std) if e['trial'] == trial]
    return trials[trial], np.array([e['xy_m'] for e in events]) * 1000, human


def draw_human_layout(ax, human, labels=True):
    names = {'pt_start': 'start', 'pt_left': 'first', 'pt_right': 'first', 'st_left': 'second', 'st_top': 'second',
             'st_right': 'second'}
    for name, (x, y), r in zip(human.NAMES, human.CENTRES, human.RADII):
        ax.add_patch(Circle((x, y), r, fc='white', ec=MUTED, lw=.7, zorder=1))
        if labels:
            ax.text(x, y, names[name], ha='center', va='center', fontsize=6.2 * S, color=SOFT, zorder=2)


def photo(source, name, width_cm):
    """A photograph resampled to 600 dpi at the width it is printed (no other change)."""
    from PIL import Image
    image = Image.open(ROOT / source).convert('RGB')
    width = round(width_cm / 2.54 * 600)
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / f'{name}.png', dpi=(600, 600))
    print(OUT / f'{name}.png')


# ---------------------------------------------------------------- Chapter 1

def fig_manus():
    photo('slide_deck/3_introduction.png', 'fig1_1_manus', 11)


def panel_title(fig, x, y, text):
    fig.text(x, y, text, fontsize=8.5 * S, ha='left', va='top')


def fig_overview():
    phi, pct_map, events, arrivals = one_cycle()
    fig = plt.figure(figsize=(WIDE, 4.6))
    a = fig.add_axes([0, .53, .3, .41])
    b = fig.add_axes([.42, .6, .56, .34])
    c = fig.add_axes([0, .02, .27, .38])
    d = fig.add_axes([.39, .09, .25, .31])
    e = fig.add_axes([.68, .02, .32, .38])
    for ax in (a, c, e):
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
    draw_arm(a, lw=3.5)
    draw_targets(a, labels=False, size=.012)
    a.plot(*phi[:, :2].T, color=BLUE, lw=.7, alpha=.85, zorder=3)
    for ev in events:
        a.plot(*ev['xy_m'], 'o', ms=3, color=ORANGE, mec='white', mew=.4, zorder=4)
    a.text(0, -.02, 'shoulder', fontsize=6.2 * S, color=SOFT, ha='center', va='top')
    a.set_xlim(-.16, .26)
    a.set_ylim(-.06, .58)
    draw_map(b, pct_map, events)
    for s in arrivals:
        b.axvline(s * H.DT, color=MUTED, lw=.4)
    b.set_xlabel('time (s)')
    b.set_ylabel('how far ahead')
    draw_targets(c, labels=False, size=.012)
    place = {'A': (-.009, 0, 'right', 'center'), 'B': (0, -.01, 'center', 'top'), 'C': (.01, 0, 'left', 'center'),
             'D': (.009, -.001, 'left', 'center'), 'E': (.01, 0, 'left', 'center')}
    for s in json.loads((ROOT / 'data/subgoals/agent1.json').read_text())['subgoals']:
        dx, dy, ha, va = place[s['id']]
        c.plot(*s['xy_m'], 'D', ms=4, color=ORANGE, mec='white', mew=.4, zorder=4)
        c.text(s['xy_m'][0] + dx, s['xy_m'][1] + dy, s['id'], fontsize=6.5 * S, color=INK, ha=ha, va=va, zorder=5)
    c.set_xlim(CENTRE[0] - .12, CENTRE[0] + .12)
    c.set_ylim(CENTRE[1] - .12, CENTRE[1] + .12)
    t = transfer['agent3']
    rows = [('agent\nalone', t['frozen'], MID), ('single\ntimescale', t['single'], DARK), ('many\ntimescales', t['multiscale'], BLUE)]
    for x, (name, value, color) in enumerate(rows):
        d.bar(x, value, .62, color=color)
        d.text(x, value + .03, pct(value), ha='center', fontsize=7 * S)
    d.set_xticks(range(3), [r[0] for r in rows], fontsize=6.2 * S)
    d.tick_params(axis='x', length=0)
    d.set_ylim(0, 1.08)
    percent_axis(d, [0, .5, 1])
    d.set_ylabel('unseen orders completed')
    grid(d)
    trial, points, human = human_example()
    draw_human_layout(e, human, labels=False)
    xy = trial['xy'][trial['bounds'][0]:]
    e.plot(*xy.T, color=ORANGE, lw=.8, alpha=.9, zorder=3)
    e.plot(*points.T, 'o', ms=3, color=INK, mec='white', mew=.4, zorder=4)
    e.set_xlim(-520, -5)
    e.set_ylim(-390, -5)
    panel_title(fig, 0, 1, 'a  Practised reaching')
    panel_title(fig, .33, 1, 'b  Change in the predicted future at 39 timescales')
    panel_title(fig, 0, .47, 'c  Subgoals')
    panel_title(fig, .33, .47, 'd  Reuse by a new agent')
    panel_title(fig, .68, .47, 'e  A human reach')
    save(fig, 'fig1_2_overview')


# ---------------------------------------------------------------- Chapter 3: method

def fig_tasks():
    import human
    fig = plt.figure(figsize=(WIDE, 2.45))
    a, b, c = fig.add_axes([0, 0, .28, .86]), fig.add_axes([.32, 0, .3, .86]), fig.add_axes([.66, 0, .34, .86])
    for ax in (a, b, c):
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
    elbow = draw_arm(a, color=MID, lw=3.5)
    draw_targets(a, labels=False, size=.012)
    a.plot(*CENTRE, 'o', ms=3, color=INK, zorder=4)
    a.text(0, -.025, 'shoulder', fontsize=6.2 * S, color=SOFT, ha='center', va='top')
    a.text(elbow[0] + .02, elbow[1], 'elbow', fontsize=6.2 * S, color=SOFT, ha='left', va='center')
    a.text(CENTRE[0] + .14, CENTRE[1], 'targets on a\n10 cm ring', fontsize=6.2 * S, color=SOFT, ha='left', va='center')
    a.set_xlim(-.14, .34)
    a.set_ylim(-.06, .58)
    for k, (x, y) in enumerate(TARGETS):
        b.add_patch(Circle((x, y), .02, fc='white', ec=MUTED, lw=.7, zorder=2))
        b.text(*(TARGETS[k] + .37 * (TARGETS[k] - CENTRE)), f'T{k}', ha='center', va='center', fontsize=6.2 * S, color=SOFT)
        b.text(x, y, str(ORDER.index(k) + 1), ha='center', va='center', fontsize=6.8 * S, color=BLUE, fontweight='bold', zorder=4)
    points = np.vstack([CENTRE] + [TARGETS[k] for k in ORDER])
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        direction = (end - start) / np.linalg.norm(end - start)
        b.add_patch(FancyArrowPatch(start + .022 * direction * (i > 0), end - .022 * direction, arrowstyle='-|>',
                                    mutation_scale=5, color=BLUE, lw=.6, alpha=.6, zorder=3))
    b.plot(*CENTRE, 's', ms=3.2, color=INK, zorder=4)
    b.plot([CENTRE[0] - .17, CENTRE[0] - .07], [CENTRE[1] - .15] * 2, color=INK, lw=.8)
    b.text(CENTRE[0] - .12, CENTRE[1] - .158, '10 cm', fontsize=6.2 * S, color=SOFT, ha='center', va='top')
    b.set_xlim(CENTRE[0] - .17, CENTRE[0] + .17)
    b.set_ylim(CENTRE[1] - .18, CENTRE[1] + .16)
    names = {'pt_start': 'start', 'pt_left': 'first', 'pt_right': 'first', 'st_left': 'second', 'st_top': 'second',
             'st_right': 'second'}
    for name, (x, y), r in zip(human.NAMES, human.CENTRES, human.RADII):
        c.add_patch(Circle((x, y), r, fc='white', ec=MUTED, lw=.7, zorder=1))
        above = name == 'st_top'
        c.text(x, y + r + 9 if above else y - r - 9, names[name], ha='center', va='bottom' if above else 'top', fontsize=6.2 * S,
               color=SOFT)
    path = [human.CENTRES[0], human.CENTRES[1], human.CENTRES[4], human.CENTRES[0]]
    for i in range(3):
        start, end = np.array(path[i]), np.array(path[i + 1])
        direction = (end - start) / np.linalg.norm(end - start)
        c.add_patch(FancyArrowPatch(start + 36 * direction, end - 44 * direction, arrowstyle='-|>', mutation_scale=6,
                                    color=ORANGE, lw=.9, alpha=.9, connectionstyle='arc3,rad=-.15', zorder=3))
    c.plot([-500, -400], [-400, -400], color=INK, lw=.8)
    c.text(-450, -410, '10 cm', fontsize=6.2 * S, color=SOFT, ha='center', va='top')
    c.set_xlim(-525, 0)
    c.set_ylim(-440, 20)
    panel_title(fig, 0, 1, 'a  Arm and targets')
    panel_title(fig, .32, 1, 'b  Practised order')
    panel_title(fig, .66, 1, 'c  Human via-point task')
    save(fig, 'fig3_1_tasks')


def fig_bank():
    """What each discount and each band weights in the future."""
    gammas, lag = H.discounts(), np.geomspace(.004, 6, 900)
    kernels = gammas[:, None] ** (lag / H.DT)
    bands = np.diff(kernels, axis=0) / np.abs(np.diff(-np.log(gammas)))[:, None]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.15))
    for k, c in zip(kernels, plt.cm.Blues(np.linspace(.3, .95, len(kernels)))):
        a.plot(lag, k, color=c, lw=.7)
    for i, band in enumerate(bands):
        b.plot(lag, band / band.max(), color=plt.cm.Oranges(.3 + .65 * i / len(bands)), lw=.7)
    a.set_ylabel('weight on the future')
    b.set_ylabel('weight (peak = 1)')
    a.text(.012, .08, '15 ms', color=SOFT, fontsize=7 * S)
    a.text(1.05, .9, '1.9 s', color=SOFT, fontsize=7 * S)
    for ax, title in ((a, '40 discount factors'), (b, '39 bands (neighbouring differences)')):
        ax.set_xscale('log')
        ax.set_xlim(.004, 6)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([.01, .1, 1], ['10 ms', '100 ms', '1 s'])
        ax.set_xlabel('time into the future')
        ax.set_title(title, loc='left', color=SOFT)
    letter(a, 'a', -.2)
    letter(b, 'b', -.2)
    fig.tight_layout(w_pad=2)
    save(fig, 'fig3_2_bank')


def fig_future_map():
    phi, pct_map, events, arrivals = one_cycle()
    t = np.arange(len(phi)) * H.DT
    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 3.5), gridspec_kw=dict(height_ratios=[1, 2.3], width_ratios=[1, .018],
                                                                           hspace=.12, wspace=.03))
    (a, spare), (b, cax) = axes
    spare.axis('off')
    a.sharex(b)
    a.tick_params(labelbottom=False)
    a.plot(t, np.linalg.norm(phi[:, 2:4], axis=1), color=INK, lw=.9)
    for k, s in enumerate(arrivals):
        for ax in (a, b):
            ax.axvline(s * H.DT, color=MUTED, lw=.5)
        a.text(s * H.DT, 2.85, f'T{ORDER[k]}', ha='center', va='bottom', fontsize=7 * S, color=SOFT)
    a.set_ylim(0, 2.8)
    a.set_yticks([0, 1, 2])
    a.set_ylabel('hand speed\n(m/s)')
    mesh = draw_map(b, pct_map, events)
    b.set_ylabel('band (peak lag)')
    b.set_xlabel('time (s)')
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([.5, .75, .9, 1])
    bar.set_ticklabels(['50th', '75th', '90th', '100th'])
    bar.ax.tick_params(labelsize=6.5, length=2)
    bar.outline.set_linewidth(.4)
    bar.set_label('percentile of change', fontsize=7 * S)
    letter(a, 'a', -.1, .9)
    letter(b, 'b', -.1, .97)
    fig.align_ylabels([a, b])
    save(fig, 'fig3_3_future_map')
    return len(events)


def box(ax, xy, w, h, text, fc='white', ec=None):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.015',
                                fc=fc, ec=ec or MUTED, lw=.8, zorder=2))
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=6.8 * S, color=INK, zorder=3, linespacing=1.3)


def arrow(ax, start, end, rad=0.):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8, color=SOFT, lw=.9, shrinkA=0, shrinkB=0,
                                 connectionstyle=f'arc3,rad={rad}', zorder=1))


def fig_controller():
    fig = plt.figure(figsize=(WIDE, 2.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, .39)
    ax.axis('off')
    box(ax, (.11, .25), .2, .12, 'Selector\n(semi-Markov\ndouble DQN)', fc='#eef4fc', ec=BLUE)
    box(ax, (.47, .33), .23, .085, 'Base controller (SAC)\none 10 ms step', fc='#f3f2ef')
    box(ax, (.47, .17), .23, .12, 'Option to subgoal A to E\nstops within 2 cm\nor after 300 ms', fc='#fdf0ea', ec=ORANGE)
    box(ax, (.87, .25), .2, .12, 'MotorNet arm\ntwo joints,\nsix muscles', fc='#f3f2ef')
    arrow(ax, (.21, .28), (.355, .33))
    arrow(ax, (.21, .22), (.355, .17))
    ax.text(.265, .33, 'choice 0', fontsize=6.4 * S, color=SOFT, ha='center', va='bottom')
    ax.text(.265, .17, 'choices 1 to 5', fontsize=6.4 * S, color=SOFT, ha='center', va='top')
    arrow(ax, (.585, .33), (.77, .28))
    arrow(ax, (.585, .17), (.77, .22))
    ax.text(.68, .25, 'muscle\nexcitations', fontsize=6.4 * S, color=SOFT, ha='center', va='center')
    arrow(ax, (.87, .19), (.11, .19), rad=-.25)
    ax.text(.49, .045, 'state returned to the selector: goal, target, hand position, and muscle lengths, velocities and '
            'activations', fontsize=6.4 * S, color=SOFT, ha='center', va='center')
    save(fig, 'fig3_4_controller')


# ---------------------------------------------------------------- Chapter 4: results

def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def fig_onsets():
    """Share of events in the first third of a reach, each of the 17 agent recordings, against chance."""
    rows = transitions['onsets']['recordings']
    names = [n for a in (1, 2, 3) for n in ([f'agent{a}'] if a > 1 else []) + [f'agent{a}_noise{k}' for k in range(1, 6)]]
    label = lambda n: f'agent {n[5]}, ' + ('noise-free' if '_' not in n else f'noise seed {n[-1]}')
    fig, ax = plt.subplots(figsize=(HALF + .5, 3.6))
    y = np.arange(len(names))[::-1] + np.array([2 if n.startswith('agent1') else 1 if n.startswith('agent2') else 0 for n in names])
    for yi, n in zip(y, names):
        r = rows[n]
        lo, hi = wilson(r['all']['early'], r['all']['events'])
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=1.6, alpha=.35, solid_capstyle='butt')
        ax.plot(r['chance'], yi, '|', color=DARK, ms=7, mew=1.2)
        ax.plot(r['all']['share'], yi, 'o', color=BLUE, ms=4.6, mec='white', mew=.6, zorder=3)
        ax.text(1.03, yi, f"{r['all']['early']}/{r['all']['events']}", va='center', ha='left', fontsize=6.8 * S, color=SOFT,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [label(n) for n in names], fontsize=7 * S)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1)
    percent_axis(ax, [0, .25, .5, .75], 'x')
    ax.set_xlabel('events in the first third of the reach')
    grid(ax, 'x')
    ax.spines['left'].set_visible(False)
    ax.plot([], [], 'o', color=BLUE, ms=4.6, label='rule (95% interval)')
    ax.plot([], [], '|', color=DARK, ms=7, mew=1.2, label='chance (share of time)')
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.0), ncol=2, handlelength=1.2, columnspacing=1.2)
    ax.text(1.03, y[0] + 1.1, 'early/all', ha='left', fontsize=6.8 * S, color=SOFT, transform=ax.get_yaxis_transform())
    save(fig, 'fig4_1_onsets')


def fig_decoding():
    d = muscles['decoding']
    null = np.array(d['null'])
    fig, ax = plt.subplots(figsize=(HALF + .3, 1.75))
    rows = [('six muscle activations', d['muscles'], BLUE), ('position-free kinematics', d['kinematics'], MID)]
    for yi, (name, value, color) in zip((2, 1), rows):
        ax.barh(yi, value, .56, color=color)
        ax.text(value + .012, yi, pct(value, 1), va='center', fontsize=7.5 * S)
    lo, hi = np.percentile(null, [2.5, 97.5])
    ax.barh(0, hi - lo, .56, left=lo, color=PALE)
    ax.plot([null.max()] * 2, [-.28, .28], color=DARK, lw=1)
    ax.text(d['chance'] + .012, 0, f'largest of 1,000 shuffles: {pct(null.max(), 1)}', va='center', fontsize=7 * S, color=SOFT)
    ax.axvline(d['chance'], color=DARK, lw=.8)
    ax.text(d['chance'] + .006, 2.42, 'most common reach', fontsize=6.8 * S, color=SOFT, va='bottom')
    ax.set_yticks([2, 1, 0], ['six muscle activations', 'position-free kinematics', 'muscles, labels shuffled'])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-.5, 2.9)
    ax.set_xlim(0, .62)
    percent_axis(ax, [0, .2, .4, .6], 'x')
    ax.set_xlabel('reaches identified correctly')
    ax.spines['left'].set_visible(False)
    grid(ax, 'x')
    save(fig, 'fig4_2_decoding')


def fig_transfer():
    t = transfer['agent1']
    fig, ax = plt.subplots(figsize=(HALF + .4, 2.6))
    bars = [('agent alone', PALE, 'frozen'), ('re-aim at the goal', MID, 'reaim'), ('subgoals A to E', BLUE, 'selectors')]
    for g, panel in enumerate(('orders96', 'sequences48')):
        for j, (name, color, key) in enumerate(bars):
            x = g * 3.6 + j
            ax.bar(x, t[panel][key], .8, color=color, label=name if g == 0 else None)
            ax.text(x, t[panel][key] + .025, pct(t[panel][key]), ha='center', fontsize=7.5 * S)
        ax.plot(g * 3.6 + 2 + np.array([-.22, 0, .22]), t[panel]['seeds'], 'o', ms=3.4, color=INK, mfc='white', mew=.7, zorder=3)
    ax.plot([], [], 'o', ms=3.4, color=INK, mfc='white', mew=.7, label='selector seeds')
    handles, labels = ax.get_legend_handles_labels()
    pick = [labels.index(n) for n in ('agent alone', 'subgoals A to E', 're-aim at the goal', 'selector seeds')]
    ax.set_xticks([1, 4.6], [f"8 targets\n{t['orders96']['orders']} unseen orders",
                             f"16 targets\n{t['sequences48']['orders']} unseen sequences"])
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.05)
    percent_axis(ax, [0, .25, .5, .75, 1])
    ax.set_ylabel('sequences completed')
    grid(ax)
    ax.legend([handles[i] for i in pick], [labels[i] for i in pick], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
              handlelength=1, columnspacing=1)
    save(fig, 'fig4_3_transfer')


def fig_agent3():
    t = transfer['agent3']
    keys = (('frozen', 'agent\nalone', MID), ('single', 'single\nhorizon', DARK), ('multiscale', 'multi-\ntimescale', BLUE))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1.1, .9], wspace=.55))
    rng = np.random.default_rng(0)
    for i, (key, name, color) in enumerate(keys):
        o = t['orders'][key]
        means = np.array(o['means'])
        a.scatter(i + rng.uniform(-.24, .24, len(means)), means, s=5, color=color, alpha=.35, lw=0)
        m, (lo, hi) = means.mean(), o['ci']
        a.errorbar(i + .38, m, [[m - lo], [hi - m]], fmt='o', ms=4.2, color=color, mec='white', mew=.6, capsize=0, lw=1.4)
        a.text(i, 1.08, pct(m), ha='center', fontsize=7.5 * S)
    a.set_xticks(range(3), [k[1] for k in keys], fontsize=6.8 * S)
    a.tick_params(axis='x', length=0)
    a.set_ylim(-.03, 1.14)
    percent_axis(a, [0, .25, .5, .75, 1])
    a.set_ylabel('completion per order')
    grid(a)
    for y, key in ((1, 'multiscale_minus_frozen'), (0, 'multiscale_minus_single')):
        d, lo, hi = (100 * v for v in t[key])
        b.plot([lo, hi], [y, y], color=BLUE, lw=1.6)
        b.plot(d, y, 'o', color=BLUE, ms=5, mec='white', mew=.6)
        b.text(d, y + .2, f'+{d:.1f} [{lo:.1f}, {hi:.1f}]', ha='center', fontsize=7 * S)
    b.axvline(0, color=DARK, lw=.6)
    b.set_yticks([1, 0], ['against the\nagent alone', 'against single-\nhorizon subgoals'], fontsize=6.8 * S)
    b.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    b.set_ylim(-.5, 1.6)
    b.set_xlim(-5, 80)
    b.set_xlabel('gain (percentage points)')
    grid(b, 'x')
    times = t['duration']
    for i, (key, name, color) in enumerate(keys):
        c.bar(i, times[key], .7, color=color)
        c.text(i, times[key] + .15, f'{times[key]:.1f} s', ha='center', fontsize=7.5 * S)
    c.set_xticks(range(3), ['alone', 'single', 'multi'], fontsize=6.8 * S)
    c.tick_params(axis='x', length=0)
    c.set_ylim(0, 8)
    c.set_ylabel('time to finish (s)')
    grid(c)
    for ax, s, x in ((a, 'a', -.3), (b, 'b', -.62), (c, 'c', -.42)):
        letter(ax, s, x, 1.02)
    save(fig, 'fig4_4_agent3')


def fig_torque():
    torque = muscles['torque']
    names = list(torque)
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.3), sharey=True, gridspec_kw=dict(wspace=.12))
    for j, (ax, joint) in enumerate(zip(axes, ('shoulder', 'elbow'))):
        for i, n in enumerate(names):
            pos, neg, net = torque[n]['positive'][j], torque[n]['negative'][j], torque[n]['net'][j]
            ax.bar(i, pos, .55, color=PALE)
            ax.bar(i, neg, .55, color=PALE)
            ax.plot([i - .27, i + .27], [net, net], color=BLUE, lw=2.2, solid_capstyle='butt')
            ax.text(i, pos + 2, pct(1 - abs(net) / (pos - neg)), ha='center', fontsize=7 * S, color=INK)
        ax.axhline(0, color=DARK, lw=.6)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel('subgoal')
        ax.tick_params(axis='x', length=0)
        ax.set_title(joint, loc='left', color=SOFT)
        grid(ax)
    axes[0].set_ylabel('muscle torque (N m)')
    axes[0].set_ylim(-32, 45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALE), plt.Line2D([], [], color=BLUE, lw=2.2)]
    fig.legend(handles, ['torque from the muscles pulling each way', 'net torque'], loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .99), handlelength=1.4)
    save(fig, 'fig4_5_torque')


def fig_tempo():
    t = transitions['tempo']['tempi']
    keys = list(t)
    w = np.array([float(k) for k in keys])
    get = lambda key: np.array([t[k][key] for k in keys])
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=.4))
    a.plot(w, get('chance'), color=MID, lw=.9)
    a.text(w[-1], get('chance')[-1] - .06, 'chance', ha='right', fontsize=7 * S, color=SOFT)
    a.plot(w, get('best_share'), 'o-', color=PALE, ms=4, lw=1.2, label='best single horizon, chosen afterwards')
    a.plot(w, get('tuned'), 's-', color=DARK, ms=3.8, lw=1.2, label='horizon tuned at normal speed')
    a.plot(w, get('rule'), 'o-', color=BLUE, ms=4.6, lw=1.8, mec='white', mew=.5, label='multi-timescale rule')
    a.set_xscale('log')
    a.set_xticks(w, [f'×{x:g}' for x in w])
    a.minorticks_off()
    a.set_xlabel('duration (× normal)')
    a.set_ylim(0, 1)
    percent_axis(a, [0, .25, .5, .75, 1])
    a.set_ylabel('events in the first third')
    a.legend(loc='upper left', handlelength=1.4, fontsize=6.8 * S)
    grid(a)
    h = get('best_horizon_s')
    b.plot(w, h, 'o-', color=DARK, ms=4.2, lw=1.2, label='best single horizon')
    b.plot(w, h[1] * w, color=MID, lw=.8, label='proportional to duration')
    for x, y in zip(w, h):
        b.text(x * 1.06, y / 1.15, f'{1000 * y:.0f} ms', ha='left', va='top', fontsize=6.8 * S, color=SOFT)
    b.legend(loc='upper left', handlelength=1.4, fontsize=6.8 * S)
    b.set_xlim(.42, 4.4)
    b.set_xscale('log')
    b.set_yscale('log')
    b.set_xticks(w, [f'×{x:g}' for x in w])
    b.set_xlim(.42, 4.4)
    b.set_yticks([.1, .2, .5, 1], ['0.1 s', '0.2 s', '0.5 s', '1 s'])
    b.minorticks_off()
    b.set_ylim(.05, 1.3)
    b.set_xlabel('duration (× normal)')
    b.set_ylabel('best single horizon')
    grid(b)
    letter(a, 'a', -.2, 1.02)
    letter(b, 'b', -.25, 1.02)
    save(fig, 'fig4_6_tempo')


def fig_robustness():
    r = transitions['robustness']
    c = r['conditions']
    order = ['trained', 'mass_x1.15', 'mass_x1.3', 'forearm_x1.3', 'forearm_x1.5', 'forearm_x1.8', 'workspace_x0.8',
             'workspace_x0.9', 'workspace_x1.1', 'agent2', 'agent3', 'tempo_x0.5', 'tempo_x1.8', 'tempo_x3', 'humans']
    names = {'trained': 'agent 1', 'agent2': 'agent 2', 'agent3': 'agent 3', 'humans': 'humans'}
    label = lambda n: names.get(n, n.replace('mass_x', 'both segments ×').replace('_x', ' ×'))
    x = np.arange(len(order))
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDE, 4.3), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=.75))
    a.axvspan(.5, 8.5, color='#f3f2ef', lw=0, zorder=0)
    a.text(4.5, 33, 'heavier limbs and rescaled workspaces', ha='center', fontsize=6.8 * S, color=SOFT)
    a.axhline(0, color=DARK, lw=.6)
    a.plot(x, [100 * (c[n]['tuned'] - c[n]['chance']) for n in order], 's-', color=DARK, ms=3.6, lw=1,
           label=f"single horizon tuned on agent 1 ({1000 * r['tuned_horizon_s']:.0f} ms)")
    a.plot(x, [100 * (c[n]['rule'] - c[n]['chance']) for n in order], 'o-', color=BLUE, ms=4.4, lw=1.5, mec='white', mew=.5,
           label='multi-timescale rule')
    a.set_xticks(x, [label(n) for n in order], rotation=40, ha='right', fontsize=6.8 * S)
    a.set_xlim(-.5, len(order) - .5)
    a.set_ylim(-20, 38)
    a.set_ylabel('events at onset\n(points above chance)')
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.4)
    grid(a)
    worst = 100 * np.array(r['worst_case']['singles'])
    horizons = H.mean_horizons() * 1000
    b.bar(horizons, worst, width=horizons * .1, color=MID, label='each of the 40 single horizons')
    b.axhline(100 * r['worst_case']['rule'], color=BLUE, lw=1.8, label='multi-timescale rule')
    b.axhline(0, color=DARK, lw=.6)
    b.set_xscale('log')
    b.set_xticks([20, 50, 100, 200, 500, 1000, 2000], ['20 ms', '50 ms', '100 ms', '200 ms', '500 ms', '1 s', '2 s'])
    b.minorticks_off()
    b.set_xlabel('single horizon')
    b.set_ylabel('worst of 15 conditions\n(points above chance)')
    b.set_ylim(-20, 8)
    b.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.4)
    grid(b)
    letter(a, 'a', -.1, 1.12)
    letter(b, 'b', -.1, 1.12)
    save(fig, 'fig4_7_robustness')


def fig_via_points():
    v = humans['via_points']
    via = checks['via_singles.json']['50ms']
    bins = np.array(v['bins_frames']) * 10
    centres = (bins[:-1] + bins[1:]) / 2
    events, frames = np.array(v['events_histogram'], float), np.array(v['frames_histogram'], float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.35), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.75))
    a.axvspan(-50, 50, color='#fbe9e1', lw=0, zorder=0)
    a.bar(centres, frames / frames.sum(), 46, color=PALE, label='all moments')
    a.step(np.r_[bins[0], centres, bins[-1]], np.r_[0, events / events.sum(), 0], where='mid', color=ORANGE, lw=1.6,
           label='rule events')
    a.set_xlabel('time from the nearest via-point passage (ms)')
    a.set_ylabel('share')
    percent_axis(a, [0, .2, .4])
    a.set_xlim(-400, 400)
    a.legend(loc='upper left', handlelength=1.4)
    grid(a)
    rows = [('all moments', via['frames'], PALE), ('best single timescale', via['best'], DARK),
            ('multi-timescale rule', via['rule'], ORANGE)]
    for yi, (name, value, color) in zip((0, 1, 2), rows):
        b.barh(yi, value, .58, color=color)
        b.text(value + .015, yi, pct(value), va='center', fontsize=7.5 * S)
    b.set_yticks([0, 1, 2], [r[0] for r in rows], fontsize=7 * S)
    b.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    b.set_xlim(0, .8)
    percent_axis(b, [0, .2, .4, .6], 'x')
    b.set_xlabel('within 50 ms of a passage')
    grid(b, 'x')
    letter(a, 'a', -.17, 1.02)
    letter(b, 'b', -.66, 1.02)
    save(fig, 'fig4_8_via_points')


def fig_human_onsets():
    o, tuned = humans['onsets'], transitions['robustness']['conditions']['humans']['tuned']
    ch = humans['chunking']
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.45), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=.4))
    rows = [('horizon tuned\non agents', tuned, MID), ('best single\n(afterwards)', o['best_discount_share'], DARK),
            ('multi-\ntimescale rule', o['rule'], ORANGE)]
    for x, (name, value, color) in enumerate(rows):
        a.bar(x, value, .62, color=color)
        a.text(x, value + .02, pct(value), ha='center', fontsize=7.5 * S)
    a.axhline(o['chance'], color=INK, lw=.7)
    a.text(-.42, o['chance'] + .02, 'chance', fontsize=7 * S, color=SOFT)
    a.set_xticks(range(3), [r[0] for r in rows], fontsize=6.8 * S)
    a.tick_params(axis='x', length=0)
    a.set_ylim(0, .72)
    percent_axis(a, [0, .2, .4, .6])
    a.set_ylabel('events in the first third')
    grid(a)
    groups = [100 * np.array(ch[k]) for k in ('flat', 'hierarchical')]
    rng = np.random.default_rng(1)
    for i, (g, color) in enumerate(zip(groups, (MID, ORANGE))):
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        b.add_patch(plt.Rectangle((i - .22, q1), .44, q3 - q1, fc='white', ec=INK, lw=.7, zorder=2))
        b.plot([i - .22, i + .22], [med, med], color=INK, lw=1.4, zorder=3)
        b.scatter(i + rng.uniform(-.16, .16, len(g)), g, s=7, color=color, alpha=.75, lw=0, zorder=4)
        b.text(i, 9.0, f'median {med:.1f}', ha='center', fontsize=7 * S, color=SOFT)
    b.set_xticks([0, 1], [f'flat\n(n = {len(groups[0])})', f'hierarchical\n(n = {len(groups[1])})'], fontsize=7 * S)
    b.tick_params(axis='x', length=0)
    b.set_xlim(-.6, 1.6)
    b.set_ylim(0, 9.8)
    b.set_ylabel('events per 100 frames\ninside the second target')
    grid(b)
    letter(a, 'a', -.2, 1.02)
    letter(b, 'b', -.3, 1.02)
    save(fig, 'fig4_9_human_onsets')


def fig_systems():
    lags = H.band_lags()
    systems = [(transitions['bands'], BLUE, 'agents'), (humans['bands'], ORANGE, 'humans')]
    fig, ax = plt.subplots(figsize=(WIDE * .82, 2.9))
    rng = np.random.default_rng(0)
    for data, color, name in systems:
        u = np.array(data['units'], float)
        draws = rng.integers(0, len(u), (2000, len(u)))
        lo, hi = np.nanpercentile(u[draws][..., 0].sum(1) / u[draws][..., 1].sum(1), [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=color, alpha=.15, lw=0)
        ax.plot(lags, data['band_share'], color=color, lw=1.5, label=f'{name}: one band at a time')
        ax.axhline(data['rule'], color=color, lw=.9, alpha=.9)
        ax.text(2.25, data['rule'], f"{name}: rule {pct(data['rule'])}", va='center', fontsize=7 * S, color=SOFT)
    for (src, color, _), (dst, _, _) in ((systems[0], systems[1]), (systems[1], systems[0])):
        g = int(np.nanargmax(src['band_share']))
        y0, y1 = src['band_share'][g], dst['band_share'][g]
        ax.plot(lags[g], y0, 'v', color=color, ms=6, mec='white', mew=.5, zorder=4)
        ax.add_patch(FancyArrowPatch((lags[g], y0 - .03), (lags[g], y1 + .03), arrowstyle='-|>', mutation_scale=7,
                                     color=color, lw=.9))
        ax.plot(lags[g], y1, 'X', color=color, ms=6, mec='white', mew=.5, zorder=4)
    chance = transitions['onsets']['mean_chance']
    ax.axhline(chance, color=DARK, lw=.6)
    ax.text(.0105, chance + .015, 'chance', fontsize=7 * S, color=SOFT)
    ax.set_xscale('log')
    ax.set_xlim(.009, 2)
    ax.set_xticks([.01, .03, .1, .3, 1], ['10 ms', '30 ms', '100 ms', '300 ms', '1 s'])
    ax.minorticks_off()
    ax.set_xlabel('band used on its own (peak lag)')
    ax.set_ylabel('events in the first third')
    ax.set_ylim(0, 1)
    percent_axis(ax, [0, .25, .5, .75, 1])
    ax.legend(loc='upper left', handlelength=1.4)
    grid(ax)
    save(fig, 'fig4_10_systems')


def fig_effects():
    rows_ = transitions['onsets']['recordings'].values()
    early, events, chance = (np.array([f(r) for r in rows_], float) for f in
                             (lambda r: r['all']['early'], lambda r: r['all']['events'], lambda r: r['chance']))
    excess = lambda i: early[i].sum(-1) / events[i].sum(-1) - (chance[i] * events[i]).sum(-1) / events[i].sum(-1)
    draws = np.random.default_rng(0).integers(0, len(early), (10000, len(early)))
    onset = [excess(np.arange(len(early))), *np.percentile(excess(draws), [2.5, 97.5])]
    robust, hum = transitions['robustness'], transitions['robustness']['conditions']['humans']
    via = checks['via_singles.json']['50ms']
    rows = [('Agents', None, None),
            ('events at onset, against chance', onset, BLUE),
            ('completion, 8 targets, against the agent alone', transfer['agent1']['orders96']['selectors_minus_frozen'], BLUE),
            ('completion, 16 targets, against the agent alone', transfer['agent1']['sequences48']['selectors_minus_frozen'], BLUE),
            ('completion, agent 3, against the agent alone', transfer['agent3']['multiscale_minus_frozen'], BLUE),
            ('completion, agent 3, against single-horizon subgoals', transfer['agent3']['multiscale_minus_single'], BLUE),
            ('events at onset, new bodies, against the tuned horizon',
             [robust['rule_minus_tuned']['mean'], *robust['rule_minus_tuned']['ci95']], BLUE),
            ('Humans (10 held out)', None, None),
            ('events at onset, against the horizon tuned on agents',
             [hum['rule'] - hum['tuned'], *robust['humans']['rule_minus_tuned_ci95']], ORANGE),
            ('events at via-points, against the best single timescale', via['rule_minus_best'], ORANGE)]
    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    fig.subplots_adjust(left=.47, right=.8)
    for y, (name, effect, color) in enumerate(rows):
        if effect is None:
            ax.text(-.02, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold', fontsize=7.5 * S)
            continue
        mean, lo, hi = (100 * v for v in effect)
        ax.plot([lo, hi], [y, y], color=color, lw=1.8)
        ax.plot(mean, y, 'o', color=color, ms=5, mec='white', mew=.7)
        ax.text(-.02, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7.2 * S)
        ax.text(1.03, y, f'+{mean:.1f} [{lo:.1f}, {hi:.1f}]', transform=ax.get_yaxis_transform(), va='center', fontsize=7.2 * S)
    ax.axvline(0, color=DARK, lw=.6)
    ax.set_ylim(len(rows) - .5, -.7)
    ax.set_xlim(-5, 75)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('difference (percentage points) with 95% interval')
    grid(ax, 'x')
    save(fig, 'fig4_11_effects')


# ---------------------------------------------------------------- Chapter 6

def fig_inmotion():
    photo('slide_deck/21_inmotion2.png', 'fig6_1_inmotion', 7)


ALL = ['fig_manus', 'fig_overview', 'fig_tasks', 'fig_bank', 'fig_future_map', 'fig_controller', 'fig_onsets', 'fig_decoding',
       'fig_transfer', 'fig_agent3', 'fig_torque', 'fig_tempo', 'fig_robustness', 'fig_via_points', 'fig_human_onsets',
       'fig_systems', 'fig_effects', 'fig_inmotion']

if __name__ == '__main__':
    for name in sys.argv[1:] or ALL:
        globals()[name]()

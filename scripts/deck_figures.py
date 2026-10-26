"""Figures for the final-review deck -> figures/deck_*.png (600 dpi) and an animated png.

    python deck_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'figures'
sys.path.insert(0, str(ROOT / 'scripts'))
import horizons as H

BLUE, ORANGE, INK, MUTED, LIGHT = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'legend.frameon': False, 'figure.facecolor': 'white'})
L1, L2 = 0.309, 0.333                      # m, upper arm and forearm
ORDER = (0, 3, 7, 5, 6, 1, 2, 4)           # practised target order
CENTRE = np.array([-0.017, 0.454])         # fingertip at the start posture
TARGETS = CENTRE + 0.1 * np.column_stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)])
LAGS = H.band_lags()


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(OUT / f'{name}.png')


def one_cycle():
    """One practised cycle of agent 1 (motor noise seed 1): features, change percentiles, events, arrivals."""
    with np.load(ROOT / 'data/recordings/continuous/agent1_noise1.npz') as d:
        keep = (d['hit'] > 0) & (d['repeat'] >= 1)
        phis, stops = d['phi'][keep], d['hit'][keep]
    mean, std = H.load_scale()
    multi = [H.change(p, mean, std)[0] for p in phis]
    refs = H.reference(np.concatenate([m[:, :s] for m, s in zip(multi, stops)], axis=1))
    arrivals = np.cumsum(stops[:8])
    steps = int(arrivals[-1]) + 5
    pct = H.percentiles(multi[0][:, :steps], refs)
    events = H.events(pct, phis[0], steps, TARGETS, -1, 0)
    return phis[0][:steps + 1], pct, events, arrivals


def fig_kernels():
    """What each discount and each band 'looks at' in the future."""
    gammas, lag = H.discounts(), np.geomspace(.005, 4, 800)
    kernels = gammas[:, None] ** (lag / H.DT)
    bands = np.diff(kernels, axis=0) / np.abs(np.diff(-np.log(gammas)))[:, None]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    colors = plt.cm.Blues(np.linspace(.35, 1, len(gammas)))
    for k, c in zip(kernels, colors):
        a.plot(lag, k, color=c, lw=.7)
    for i in range(0, len(bands)):
        b.plot(lag, bands[i] / bands[i].max(), color=plt.cm.Oranges(.3 + .7 * i / len(bands)), lw=.7)
    for ax in (a, b):
        ax.set_xscale('log')
        ax.set_xlabel('time into the future (s)')
        ax.set_xlim(.01, 4)
    a.set_ylabel('weight')
    b.set_ylabel('weight (peak = 1)')
    a.set_title('a  40 discounts: horizons from 15 ms to 1.9 s', loc='left', fontsize=9)
    b.set_title('b  39 bands: neighbour differences, peaks 10 ms to 1.8 s', loc='left', fontsize=9)
    fig.tight_layout()
    save(fig, 'deck_horizon_bank')


def draw_map(ax, pct, events, steps):
    t = np.arange(pct.shape[1] + 1) * H.DT
    r = (LAGS[1] / LAGS[0]) ** .5
    edges = np.geomspace(LAGS[0] / r, LAGS[-1] * r, len(LAGS) + 1)
    shown = np.where(np.isfinite(pct[:, :steps]), pct[:, :steps], 0)
    ax.pcolormesh(t[:steps + 1], edges, shown, cmap='Blues', vmin=.5, vmax=1, shading='flat', rasterized=True)
    for e in events:
        if e['step'] <= steps:
            ax.plot(e['step'] * H.DT, LAGS[e['peak_band']], 'o', ms=4, color=ORANGE, mec='white', mew=.6)
    ax.set_yscale('log')
    ax.set_ylim(edges[0], edges[-1])
    ax.set_ylabel('band peak lag (s)')


def fig_future_map():
    phi, pct, events, arrivals = one_cycle()
    t = np.arange(len(phi)) * H.DT
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
    a.plot(t, np.linalg.norm(phi[:, 2:4], axis=1), color=INK, lw=1)
    for k, s in enumerate(arrivals):
        for ax in (a, b):
            ax.axvline(s * H.DT, color=MUTED, lw=.5, ls='--')
        a.text(s * H.DT, 2.75, f'T{ORDER[k]}', ha='center', va='bottom', fontsize=7, color=MUTED)
    a.set_ylim(0, 2.75)
    a.set_ylabel('fingertip\nspeed (m/s)')
    draw_map(b, pct, events, pct.shape[1])
    b.set_xlabel('time (s)')
    b.set_xlim(0, t[-1])
    b.set_title(f'blue: how sharply each band turns (percentile, shown from 50th);  orange: {len(events)} events '
                '(3+ adjacent bands above the 90th)', loc='left', fontsize=7.5, color=MUTED)
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, 'deck_future_map')


def joints(xy):
    """Shoulder and elbow positions from the fingertip (elbow-flexed solution)."""
    r2 = (xy ** 2).sum()
    q2 = np.arccos(np.clip((r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
    q1 = np.arctan2(xy[1], xy[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([0, 0]), L1 * np.array([np.cos(q1), np.sin(q1)])


def animation(every=2):
    """The arm performing one cycle beside its multi-timescale map, slowed 4x."""
    phi, pct, events, arrivals = one_cycle()
    frames = []
    for s in range(1, pct.shape[1] + 1, every):
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4), dpi=180, gridspec_kw=dict(width_ratios=[1, 1.9]))
        target = ORDER[min(int(np.searchsorted(arrivals, s, 'right')), 7)]
        a.scatter(*TARGETS.T, s=60, facecolor='none', edgecolor=MUTED, lw=.8)
        a.scatter(*TARGETS[target], s=110, facecolor='none', edgecolor=ORANGE, lw=1.6)
        a.plot(*phi[:s + 1, :2].T, color=BLUE, lw=.8, alpha=.5)
        shoulder, elbow = joints(phi[s, :2])
        a.plot(*np.array([shoulder, elbow, phi[s, :2]]).T, '-o', color=INK, lw=2.2, ms=4)
        for e in events:
            if e['step'] <= s:
                a.plot(*e['xy_m'], 'o', ms=5, color=ORANGE, mec='white', mew=.6)
        a.set_xlim(-.25, .3)
        a.set_ylim(-.03, .6)
        a.set_aspect('equal')
        a.axis('off')
        a.set_title(f't = {s * H.DT:.2f} s   (slowed 4x)', loc='left', fontsize=9, color=MUTED)
        draw_map(b, pct, events, s)
        b.set_xlim(0, pct.shape[1] * H.DT)
        b.set_xlabel('time (s)')
        b.set_title('the future of the movement at 39 horizons; orange = transition', loc='left', fontsize=9, color=MUTED)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    palette = frames[-1].quantize(128, method=Image.Quantize.FASTOCTREE)   # the last frame holds every colour
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames += [frames[-1]] * 15
    frames[0].save(OUT / 'deck_arm_animation.png', save_all=True, append_images=frames[1:],
                   duration=every * H.DT * 4 * 1000, loop=0, dpi=(600, 600))
    print(OUT / 'deck_arm_animation.png')


if __name__ == '__main__':
    fig_kernels()
    fig_future_map()
    animation()

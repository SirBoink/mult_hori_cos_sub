"""Looping GIFs for the deck (PowerPoint plays GIFs; it does not play animated PNG).

    python deck_gifs.py   -> slide_deck/4_study_overview_animation.gif, slide_deck/8_controller_sequence.gif
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'slide_deck'
L1, L2 = 0.309, 0.333                      # m, upper arm and forearm
ORDER = (0, 3, 7, 5, 6, 1, 2, 4)           # practised target order
CENTRE = np.array([-0.017, 0.454])         # fingertip at the start posture
TARGETS = CENTRE + 0.1 * np.column_stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)])
GREY, RED, GREEN, BLUE, ORANGE, INK = '#8a8a8a', '#d62728', '#2e8b57', '#3b8ed8', '#e8662a', '#1c1c1c'
SLOW, EVERY = 3, 2                         # shown 3x slower; one frame per 2 steps of 10 ms


def overview_gif():
    """Slide 4: the animated PNG from deck_figures.py as a GIF that opens on its finished state."""
    apng = Image.open(ROOT / 'figures/deck_arm_animation.png')
    frames = []
    for k in range(apng.n_frames):
        apng.seek(k)
        frames.append(apng.copy())
    frames = frames[:-15]                  # drop the end hold; the finished frame now opens the loop
    durations = [1500] + [80] * len(frames)
    frames = [frames[-1]] + frames
    frames[0].save(OUT / '4_study_overview_animation.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1)


def joints(xy):
    """Elbow position from the fingertip (elbow-flexed solution), shoulder at the origin."""
    r2 = (xy ** 2).sum()
    q2 = np.arccos(np.clip((r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
    q1 = np.arctan2(xy[1], xy[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return L1 * np.array([np.cos(q1), np.sin(q1)])


def controller_frame(path, step, reached):
    fig, ax = plt.subplots(figsize=(5.5, 5.08), dpi=160)
    current = None if reached == 8 else ORDER[reached]
    for k, (x, y) in enumerate(TARGETS):
        done = k in ORDER[:reached]
        colour = RED if k == current else GREEN if done else GREY
        ax.add_patch(Circle((x, y), 0.02, facecolor='#d9efe0' if done else 'none', edgecolor=colour, lw=2.2 if k == current else 1.3))
        ax.plot(x, y, 'o', ms=3.5, color=colour)
        ax.text(x + 0.016, y + 0.016, f'T{k}', fontsize=9, color=colour)
    ax.plot(*path[:step + 1].T, color=BLUE, lw=1.4, alpha=.35)
    hand = path[step]
    elbow = joints(hand)
    ax.plot([0, elbow[0]], [0, elbow[1]], color=INK, lw=6, solid_capstyle='round')
    ax.plot([elbow[0], hand[0]], [elbow[1], hand[1]], color=BLUE, lw=6, solid_capstyle='round')
    ax.plot([0, elbow[0]], [0, elbow[1]], 'o', color=INK, ms=7)
    ax.plot(*hand, 'o', color=ORANGE, ms=7)
    status = 'sequence complete' if current is None else f'current target: T{current}'
    ax.text(-0.15, 0.612, f't = {step * 0.01:.2f} s    {status}', fontsize=11, color=INK)
    for n, k in enumerate(ORDER):
        colour = GREEN if n < reached else RED if n == reached else GREY
        ax.text(-0.145 + n * 0.057, -0.025, f'{n + 1}:T{k}', fontsize=8.5, color=colour)
    ax.set_xlim(-0.16, 0.31)
    ax.set_ylim(-0.045, 0.645)
    ax.set_aspect('equal')
    ax.grid(color='#e6e6e6', lw=.6)
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_title(f'SAC solving the practised sequence (shown {SLOW}x slower)', fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return image


def controller_gif():
    """Slide 8: agent 1 completing its practised eight-target sequence from the start posture."""
    with np.load(ROOT / 'data/recordings/continuous/agent1.npz') as d:
        rows = np.where(d['repeat'] == 0)[0][:8]
        hits = d['hit'][rows]
        path = np.concatenate([d['phi'][i, :n, :2] for i, n in zip(rows, hits)] + [d['phi'][rows[-1], hits[-1]:hits[-1] + 1, :2]])
    arrivals = np.cumsum(hits)
    steps = list(range(0, len(path), EVERY)) + [len(path) - 1]
    frames = [controller_frame(path, s, int((arrivals <= s).sum())) for s in steps]
    palette = Image.fromarray(np.hstack([np.asarray(f) for f in frames[::8] + [frames[-1]]])).quantize(255, method=Image.Quantize.FASTOCTREE)
    colours = np.array(palette.getpalette()[:765]).reshape(-1, 3)
    colours[np.argmin(((255 - colours) ** 2).sum(1))] = 255   # keep the background pure white
    palette.putpalette(colours.ravel().tolist())
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [800] + [EVERY * 10 * SLOW] * (len(frames) - 2) + [1800]
    frames[0].save(OUT / '8_controller_sequence.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1)


if __name__ == '__main__':
    overview_gif()
    controller_gif()
    for name in ('4_study_overview_animation.gif', '8_controller_sequence.gif'):
        im = Image.open(OUT / name)
        print(name, im.size, im.n_frames, 'frames, loop', im.info.get('loop'), f'{(OUT / name).stat().st_size / 1e6:.1f} MB')

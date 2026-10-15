"""Multi-timescale prediction of a movement and the coherence rule that marks its transitions.

Discounted sums of future features at 40 log-spaced discounts form a sampled Laplace transform of the future.
Differences between neighbouring discounts give 39 bands, each peaking at one future lag (10 ms to 1.8 s). An event is
a moment when the direction of at least three adjacent bands changes strongly (above the 90th percentile) together.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
DT = 0.01                  # s per step
SHORTEST_LAG = 0.01        # s, peak lag of the first band
LONGEST_LAG = 1.8          # s, peak lag of the last band
BANDS = 39
PERCENTILE = 0.9           # change threshold
MIN_BANDS = 3              # adjacent bands an event must span
MAX_MISSING = 0.005        # largest discount weight allowed beyond the end of a recording
TARGET_EXCLUSION = 0.025   # m, subgoals lie further than this from every target
SEPARATION = 0.005         # m, minimum distance between subgoals
PHASE_TOLERANCE = 0.1      # repeated events closer than this in phase are merged
CANDIDATES = 5             # single-horizon subgoals per library
SUBGOALS = 5               # largest multi-timescale library
ONSET = 1 / 3              # first third of a movement element


def discounts(bands=BANDS):
    """bands + 1 discounts whose neighbouring differences peak at log-spaced lags."""
    peaks = np.geomspace(SHORTEST_LAG, LONGEST_LAG, bands)
    ratio = peaks[1] / peaks[0]
    return np.exp(-DT / (DT * (ratio - 1) / (ratio * np.log(ratio)) * ratio ** np.arange(bands + 1)))


def mean_horizons():
    return DT / (1 - discounts())


def band_lags():
    return np.geomspace(SHORTEST_LAG, LONGEST_LAG, BANDS)


def load_scale(columns=slice(None)):
    """Fixed z-scoring constants of the agents' features (x, y, vx, vy, six muscle activations)."""
    scale = json.loads((ROOT / 'data/feature_scale.json').read_text())
    return np.array(scale['mean'])[columns], np.array(scale['std'])[columns]


def returns(phi, gammas):
    phi, gammas = np.asarray(phi, dtype=np.float64), np.asarray(gammas, dtype=np.float64)
    result = np.empty((len(gammas), len(phi), phi.shape[1]))
    future = np.zeros((len(gammas), phi.shape[1]))
    for k in range(len(phi) - 1, -1, -1):
        future = phi[k] + gammas[:, None] * future
        result[:, k] = future
    return result


def cosine_change(vectors):
    a, b = vectors[:, :-1], vectors[:, 1:]
    norms = np.linalg.norm(vectors, axis=-1)
    denominator = norms[:, :-1] * norms[:, 1:]
    valid = denominator > np.finfo(np.float64).eps
    cosine = np.divide(np.sum(a * b, axis=-1), denominator, out=np.zeros_like(denominator), where=valid)
    return np.where(valid, 1 - np.clip(cosine, -1, 1), np.nan)


def missing_mass(gammas, remaining):
    a, b = np.asarray(gammas[:-1]), np.asarray(gammas[1:])
    total = 1 / (1 - b) - 1 / (1 - a)
    rem = np.asarray(remaining)
    return (b[:, None] ** rem / (1 - b[:, None]) - a[:, None] ** rem / (1 - a[:, None])) / total[:, None]


def change(phi, mean, std, gammas=None):
    """Step-to-step cosine change of every band and of every single discount; NaN where the future is too short."""
    gammas = discounts() if gammas is None else np.asarray(gammas)
    psi = returns((phi - mean) / std, gammas)
    bands = np.diff(psi, axis=0) / np.abs(np.diff(-np.log(gammas)))[:, None, None]
    multi, single = cosine_change(bands), cosine_change(psi)
    remaining = np.arange(len(phi) - 1, 0, -1)
    multi[missing_mass(gammas, remaining) > MAX_MISSING] = np.nan
    single[gammas[:, None] ** remaining > MAX_MISSING] = np.nan
    return multi, single


def reference(values):
    return [np.sort(row[np.isfinite(row)]) for row in values]


def percentiles(values, references):
    result = np.full_like(values, np.nan, dtype=np.float64)
    for i, ref in enumerate(references):
        valid = np.isfinite(values[i])
        if len(ref):
            result[i, valid] = (np.searchsorted(ref, values[i, valid], 'left')
                                + np.searchsorted(ref, values[i, valid], 'right')) / (2 * len(ref))
    return result


def nearest_target(xy, goals):
    return float(np.linalg.norm(goals - np.asarray(xy), axis=1).min())


def events(percentile, phi, stop, goals, target, element, min_bands=MIN_BANDS):
    """Connected patches above threshold that span min_bands adjacent bands, placed at their peak."""
    active = percentile[:, :max(0, stop)] >= PERCENTILE
    labels, count = label(active, structure=np.ones((3, 3), dtype=int))
    output = []
    for component in range(1, count + 1):
        cells = np.argwhere(labels == component)
        bands = set(cells[:, 0])
        if not any(set(range(int(b), int(b) + min_bands)) <= bands for b in bands):
            continue
        band, column = max(cells, key=lambda v: (percentile[tuple(v)], -v[1], -v[0]))
        step = int(column) + 1
        output.append(dict(element=int(element), target=int(target), step=step, phase=step / max(stop, 1),
                           bands=sorted(map(int, bands)), peak_band=int(band),
                           peak_percentile=float(percentile[band, column]), xy_m=phi[step, :2].tolist(),
                           nearest_target_m=nearest_target(phi[step, :2], goals)))
    return output


def same_event(a, b):
    overlap = set(a['bands']) & set(b['bands'])
    return (a['target'] == b['target'] and abs(a['phase'] - b['phase']) <= PHASE_TOLERANCE
            and any(set(range(v, v + 3)) <= overlap for v in overlap))


def detect(phis, stops, targets, ids, goals, columns=slice(None), min_bands=MIN_BANDS):
    """Events of every element, percentiles pooled over the elements given; also returns band and discount changes."""
    mean, std = load_scale(columns)
    bands, singles = [], []
    for phi, stop in zip(phis, stops):
        multi, single = change(phi[:, columns], mean, std)
        bands.append(multi[:, :stop])
        singles.append(single[:, :stop])
    refs = reference(np.concatenate(bands, axis=1))
    found = [events(percentiles(b, refs), phi, stop, goals, target, i, min_bands)
             for b, phi, stop, target, i in zip(bands, phis, stops, targets, ids)]
    return found, bands, singles


def merge(found):
    """Merge repeated events in element order, drop those near a target, sort by (target, step)."""
    kept = []
    for event in (e for element in found for e in element):
        if not any(same_event(event, old) for old in kept):
            kept.append(event)
    return sorted((e for e in kept if e['nearest_target_m'] > TARGET_EXCLUSION), key=lambda e: (e['target'], e['step']))


def separated(xy, kept):
    return all(np.linalg.norm(np.asarray(xy) - k['xy_m']) >= SEPARATION for k in kept)


def strongest(found, size=SUBGOALS):
    """At most `size` events, highest percentile first and mutually separated, then in (target, step) order."""
    kept = []
    for event in sorted(found, key=lambda e: -e['peak_percentile']):
        if len(kept) < size and separated(event['xy_m'], kept):
            kept.append(event)
    return sorted(kept, key=lambda e: (e['target'], e['step']))


def single_horizon_points(score, trajectories, stops, goals):
    """The CANDIDATES strongest separated local maxima of one discount's change, away from the targets."""
    candidates = []
    for element, (row, phi, stop) in enumerate(zip(score, trajectories, stops)):
        row = row[:stop]
        for column in find_peaks(np.where(np.isfinite(row), row, -np.inf))[0]:
            xy = phi[column + 1, :2]
            if np.linalg.norm(goals - xy, axis=1).min() > TARGET_EXCLUSION:
                candidates.append(dict(xy_m=xy.tolist(), score=float(row[column]), element=element, step=int(column) + 1))
    selected = []
    for item in sorted(candidates, key=lambda v: (-v['score'], v['element'], v['step'])):
        if all(np.linalg.norm(np.asarray(item['xy_m']) - old['xy_m']) > SEPARATION for old in selected):
            selected.append(item)
        if len(selected) == CANDIDATES:
            break
    return selected


def rule_phases(bands, phis, stops, min_bands, goals):
    """Event phases per element, percentiles pooled over the elements given."""
    refs = reference(np.concatenate(bands, axis=1))
    return [[e['phase'] for e in events(percentiles(b, refs), phi, s, goals, -1, 0, min_bands)]
            for b, phi, s in zip(bands, phis, stops)]


def peaks(rows, stops):
    """(element, rank, phase) of every local maximum of one score, ranks pooled over the elements."""
    ref = np.sort(np.concatenate([r[np.isfinite(r)] for r in rows]))
    out = []
    for i, (row, s) in enumerate(zip(rows, stops)):
        rank = (np.searchsorted(ref, row, 'left') + np.searchsorted(ref, row, 'right')) / (2 * len(ref))
        out += [(i, rank[k], (k + 1) / s) for k in find_peaks(np.where(np.isfinite(row), row, -np.inf))[0]]
    return np.array(out).reshape(-1, 3)


def tally(phases):
    """[events in the first third, events, elements with such an event, elements]."""
    return np.array([sum(p < ONSET for e in phases for p in e), sum(len(e) for e in phases),
                     sum(any(p < ONSET for p in e) for e in phases), len(phases)])


def top(peak_rows, k, n_elements):
    """Tally of the k highest-ranked peaks (all of them if there are fewer)."""
    chosen = peak_rows[np.argsort(-peak_rows[:, 1], kind='stable')[:k]]
    return tally([[p for i, _, p in chosen if i == e] for e in range(n_elements)])


def above(peak_rows, n_elements, q=PERCENTILE):
    """Tally of the peaks ranked at or above q."""
    return tally([[p for i, rk, p in peak_rows if i == e and rk >= q] for e in range(n_elements)])


def chance(stops):
    """Tally-shaped count of frames in the first third (frames k/s, k = 1..s)."""
    return np.array([sum(int(np.ceil(s / 3)) - 1 for s in stops), sum(stops), len(stops), len(stops)])

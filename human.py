"""Human sequential reaching (Cuevas Rivera, Szpilska & Kiebel; human_data/): 20 participants move from a start
circle through a primary and a secondary target and back; fingertip at 100 Hz. Participants 401-410 set the feature
scaling; 411-420 are held out.
"""
import csv
import io
import itertools
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import horizons as H

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'human_data/bids_data.zip'
DEVELOPMENT, HELDOUT = range(401, 411), range(411, 421)
PAD = 1530            # rest frames appended to every trial so each frame has the future the bank needs
STOP_SPEED = 0.1      # m/s, end of the last segment (the agents' acquisition speed)
NAMES = ['pt_start', 'pt_left', 'pt_right', 'st_left', 'st_top', 'st_right']
SEQUENCES = {0: [1, 3, 0], 1: [1, 5, 0], 2: [1, 4, 0], 3: [2, 3, 0], 4: [2, 5, 0], 5: [2, 4, 0]}   # targets after the start


def circles():
    """Target centres (mm, data frame: origin at the sheet's top-right marker, y forward) and radii (mm)."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        tree = ET.fromstring(archive.read('bids/code/experiment_sheet.svg'))
    found = {c.get('id'): c for c in tree.iter('{http://www.w3.org/2000/svg}circle')}
    centres = np.array([[float(found[n].get('cx')), float(found[n].get('cy'))] for n in NAMES])
    radii = np.array([float(found[n].get('r')) for n in NAMES])
    marker = next(r for r in tree.iter('{http://www.w3.org/2000/svg}rect') if r.get('id') == 'mk_tr')
    corner = np.array([float(marker.get('x')) + float(marker.get('width')) / 2, float(marker.get('y')) + float(marker.get('height')) / 2])
    return np.column_stack([centres[:, 0] - corner[0], corner[1] - centres[:, 1]]), radii


CENTRES, RADII = circles()


def read_trials(participant):
    """(sequence id, fingertip xy in mm, current-target labels) of every sequential trial."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        for k in range(6):
            name = f'bids/sub-{participant}/motion/sub-{participant}_task-sequential{k}_tracksys-qualisys_motion.tsv'
            rows = csv.DictReader(io.StringIO(archive.read(name).decode()), delimiter='\t')
            groups = [list(g) for _, g in itertools.groupby(rows, key=lambda r: r['unique_trial_id'])]
            assert len(groups) == 20, name
            for group in groups:
                assert np.allclose(np.diff([float(r['time']) for r in group]), .01, atol=1e-6), name
                yield k, np.array([[float(r['SensorID_0_x']), float(r['SensorID_0_y'])] for r in group]), [r['ctarget'] for r in group]


def features(xy_m):
    """Position and smoothed velocity (m, m/s)."""
    return np.column_stack([xy_m, np.gradient(gaussian_filter1d(xy_m, 2, axis=0), .01, axis=0)])


def pad(phi):
    return np.vstack([phi, np.tile([*phi[-1, :2], 0., 0.], (PAD, 1))])


def onset_segments(participant):
    """Per trial: features and (onset, length) of segment 2 (primary -> secondary target) and segment 3 (back to the
    start, ending when speed first falls below STOP_SPEED after its peak). Segment boundaries are the dataset's own."""
    for _, xy_mm, labels in read_trials(participant):
        phi = features(xy_mm / 1000)
        runs = [label for label, _ in itertools.groupby(labels)]
        assert runs[0] in ('pt_left', 'pt_right') and runs[1].startswith('st_') and runs[2:] in (['pt_start'], ['pt_start', 'end'])
        second, third = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]][:2]
        speed = np.linalg.norm(phi[:, 2:], axis=1)
        peak = third + int(np.argmax(speed[third:]))
        slow = np.flatnonzero(speed[peak:] < STOP_SPEED)
        segments = [(second, third - second)]
        if len(slow):
            segments.append((third, peak + int(slow[0]) - third))
        yield phi, segments


def development_scale():
    frames = np.concatenate([phi for p in DEVELOPMENT for phi, _ in onset_segments(p)])
    return frames.mean(0), frames.std(0)


def segment_changes(participant, mean, std, gammas=None):
    """Band and single-discount changes of segments 2 and 3 of every trial, with their padded features and lengths."""
    bands, singles, phis, stops = [], [], [], []
    for phi, segments in onset_segments(participant):
        phi = pad(phi)
        multi, single = H.change(phi, mean, std, gammas)
        for onset, length in segments:
            bands.append(multi[:, onset:onset + length])
            singles.append(single[:, onset:onset + length])
            phis.append(phi[onset:])
            stops.append(length)
    return bands, singles, phis, stops


def inside(xy_mm, target):
    return np.linalg.norm(xy_mm - CENTRES[target], axis=-1) <= RADII[target] + 1


def movement_bounds(xy_mm, targets):
    """Movement onset (speed > 50 mm/s for 10 frames) and the speed minimum inside each via target."""
    speed = np.linalg.norm(np.gradient(xy_mm, .01, axis=0), axis=1)
    fast = speed > 50
    onset = next(i for i in range(len(fast) - 9) if fast[i:i + 10].all())
    bounds, i = [onset], onset
    for t in targets[:-1]:
        frames = np.flatnonzero(inside(xy_mm, t))
        frames = frames[frames >= i]
        first_visit = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)[0]
        i = int(first_visit[np.argmin(speed[first_visit])])
        bounds.append(i)
    return bounds

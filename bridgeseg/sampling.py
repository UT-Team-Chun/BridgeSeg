"""Label-free point samplers and the dataset used to train/evaluate with them.

Every network input is 4,096 points. The *mode* decides which part of the bridge a sample
covers: ``window`` (default; a slice across the bridge axis with full width and height, see
``AxisWindowDataset``) or ``global`` (the whole bridge, as in the AiC paper, see
``BridgeCloudDataset``). The *sampler* decides how the points are drawn inside it.

Train and inference MUST use the same mode, window and sampler, otherwise the point-density
statistics the network learnt no longer hold; checkpoints store them for ``predict()``.

Samplers (all label-free unless noted)
--------------------------------------
``density``  (default) density-balanced sampling: probability proportional to
             1 / (number of points in the point's voxel). Every occupied region of the
             bridge gets roughly the same number of samples, so dense ground / deck surfaces
             close to the scanner no longer drown thin members (piers, parapets, bracing).
``uniform``  every point equally likely (keeps the scanner's density pattern).
``stratified`` the sampler of the AiC paper: fixed class proportions chosen from the
             ground-truth labels. Needs labels, so it can only be used to reproduce the
             paper's protocol on labelled data – never for new data.

To add your own sampler (e.g. curvature- or height-aware), implement a function
``f(xyz) -> log_weights`` and register it in ``SAMPLERS``.
"""
import glob
import logging
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def normalize_points(xyz):
    """Centre at the centroid and scale into the unit sphere (same as the paper)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    p = xyz - xyz.mean(0)
    r = np.sqrt((p ** 2).sum(1)).max()
    return (p / max(r, 1e-12)).astype(np.float32)


def density_log_weights(xyz, voxel_frac=1 / 300):
    """log(1 / points-per-voxel); voxel edge = ``voxel_frac`` x bounding-box diagonal."""
    xyz = np.asarray(xyz, dtype=np.float64)
    lo = xyz.min(0)
    v = max(np.linalg.norm(xyz.max(0) - lo) * voxel_frac, 1e-9)
    key = np.floor((xyz - lo) / v).astype(np.int64)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    return -np.log(cnt[inv.ravel()].astype(np.float64))


def uniform_log_weights(xyz):
    return np.zeros(len(xyz))


SAMPLERS = {'density': density_log_weights, 'uniform': uniform_log_weights}


def weighted_subset(log_w, n, rng):
    """Draw ``n`` distinct indices with probability ~ exp(log_w) (Gumbel top-k)."""
    if n >= len(log_w):
        return np.arange(len(log_w))
    keys = log_w + rng.gumbel(size=len(log_w))
    return np.sort(np.argpartition(-keys, n)[:n])


def sample_pool(xyz, sampler='density', pool_size=2_000_000, seed=0):
    """Indices of a sampler-distributed pool of at most ``pool_size`` points.
    Uniform random 4,096-point draws from this pool follow the sampler's distribution."""
    rng = np.random.default_rng(seed)
    return weighted_subset(SAMPLERS[sampler](xyz), pool_size, rng)


def augment(points, colors, rng):
    """Random rotation about Z, scaling, translation and colour jitter (paper settings)."""
    t = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], np.float32)
    points = points @ R * rng.uniform(0.9, 1.1) + rng.uniform(0.01, 0.1, (1, 3)).astype(np.float32)
    colors = np.clip(colors + rng.normal(0, 0.02, colors.shape), 0, 1)
    return points.astype(np.float32), colors.astype(np.float32)


class BridgeCloudDataset(Dataset):
    """Whole-bridge 4,096-point samples drawn with a label-free sampler.

    Each .h5 file (``points``, ``colors``, optional ``labels``) is normalised as a whole and
    reduced once to a sampler-distributed pool (cached as ``<file>.<sampler>_pool.npz`` next
    to it). ``__getitem__`` returns a uniform random draw of ``num_points`` from one pool.
    Number of samples per file = original points x ``sample_rate`` / ``num_points``.
    """

    def __init__(self, data_dir, num_points=4096, sampler='density', pool_size=2_000_000,
                 sample_rate=0.4, transform=False, seed=0):
        self.num_points, self.transform = num_points, transform
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if not self.files:
            raise FileNotFoundError(f'no .h5 files in {data_dir}')
        self.pools, self.index = [], []
        for fi, path in enumerate(self.files):
            cache = f'{path}.{sampler}_pool{pool_size}.npz'
            if os.path.exists(cache):
                d = dict(np.load(cache))
            else:
                with h5py.File(path, 'r') as f:
                    xyz = np.asarray(f['points'])
                    rgb = np.asarray(f['colors'], np.float32) if 'colors' in f else np.ones_like(xyz, np.float32)
                    lab = np.asarray(f['labels'], np.int64) if 'labels' in f else np.full(len(xyz), -1)
                idx = sample_pool(xyz, sampler, pool_size, seed)
                d = {'points': normalize_points(xyz)[idx], 'colors': rgb[idx], 'labels': lab[idx],
                     'n_total': np.int64(len(xyz))}
                np.savez(cache, **d)
                log.info(f'{os.path.basename(path)}: pool {len(idx):,} / {len(xyz):,} points -> {cache}')
            self.pools.append(d)
            n_samples = max(1, int(int(d['n_total']) * sample_rate / num_points))
            self.index += [fi] * n_samples
        self.seed, self._calls = seed, 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        # torch.initial_seed() differs per worker and per epoch -> fresh samples every epoch
        self._calls += 1
        rng = np.random.default_rng([torch.initial_seed() % (1 << 32), self.seed, i, self._calls])
        d = self.pools[self.index[i]]
        sel = rng.choice(len(d['points']), self.num_points, replace=len(d['points']) < self.num_points)
        p, c = d['points'][sel], d['colors'][sel]
        if self.transform:
            p, c = augment(p, c, rng)
        return {'points': p.astype(np.float32), 'colors': c.astype(np.float32),
                'labels': d['labels'][sel].astype(np.int64)}


# ---------------------------------------------------------------------------------------
# Bridge-axis window sampler
# ---------------------------------------------------------------------------------------
# Each sample is a slice of the bridge cut perpendicular to its longitudinal axis:
# ``window`` metres long, but with the FULL cross-section and FULL height, so every sample
# still contains the complete vertical hierarchy (pier < girder < deck < superstructure)
# that the MBrSOL loss relies on, at a much higher point density than whole-bridge samples.
# Coordinates keep a metric scale: centred on the window and divided by ``window / 2``.

def bridge_axis(xyz, n=200_000, seed=0):
    """Unit vector (x, y) of the bridge's longitudinal axis (first principal axis in plan)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(xyz)[rng.choice(len(xyz), min(n, len(xyz)), replace=False), :2].astype(np.float64)
    w, v = np.linalg.eigh(np.cov((p - p.mean(0)).T))
    return v[:, np.argmax(w)]


def window_normalize(points, window):
    """Centre a window's points and scale them by window / 2 (keeps metric proportions)."""
    c = points.mean(0)
    return ((points - c) / (window / 2)).astype(np.float32)


def _sorted_pool(xyz, rgb, lab, sampler, pool_size, seed):
    idx = sample_pool(xyz, sampler, pool_size, seed)
    ax = bridge_axis(xyz)
    s = (np.asarray(xyz)[idx, :2] @ ax).astype(np.float64)
    o = np.argsort(s)
    idx = idx[o]
    return idx, s[o]


class AxisWindowDataset(Dataset):
    """Training samples: ``num_points`` points from a random ``window``-metre slice along the
    bridge axis (from a ``sampler``-distributed pool). Samples per file = N x rate / num_points."""

    def __init__(self, data_dir, num_points=4096, window=10.0, sampler='density',
                 pool_size=4_000_000, sample_rate=0.4, transform=False, seed=0):
        self.num_points, self.window, self.transform, self.seed = num_points, window, transform, seed
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if not self.files:
            raise FileNotFoundError(f'no .h5 files in {data_dir}')
        self.pools, self.index, self._calls = [], [], 0
        for fi, path in enumerate(self.files):
            cache = f'{path}.axis_{sampler}_pool{pool_size}.npz'
            if os.path.exists(cache):
                d = dict(np.load(cache))
            else:
                with h5py.File(path, 'r') as f:
                    xyz = np.asarray(f['points'])
                    rgb = np.asarray(f['colors'], np.float32) if 'colors' in f else np.ones_like(xyz, np.float32)
                    lab = np.asarray(f['labels'], np.int64) if 'labels' in f else np.full(len(xyz), -1)
                idx, s = _sorted_pool(xyz, rgb, lab, sampler, pool_size, seed)
                d = {'points': (xyz[idx] - xyz.min(0)).astype(np.float32), 'colors': rgb[idx],
                     'labels': lab[idx], 's': s, 'n_total': np.int64(len(xyz))}
                np.savez(cache, **d)
                log.info(f'{os.path.basename(path)}: axis pool {len(idx):,} / {len(xyz):,} -> {cache}')
            self.pools.append(d)
            self.index += [fi] * max(1, int(int(d['n_total']) * sample_rate / num_points))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        self._calls += 1
        rng = np.random.default_rng([torch.initial_seed() % (1 << 32), self.seed, i, self._calls])
        d = self.pools[self.index[i]]
        s = d['s']
        for _ in range(10):                       # avoid near-empty windows at the ends
            s0 = rng.uniform(s[0], s[-1])
            a, b = np.searchsorted(s, [s0 - self.window / 2, s0 + self.window / 2])
            if b - a >= self.num_points // 4:
                break
        sel = rng.choice(np.arange(a, b), self.num_points, replace=(b - a) < self.num_points)
        p = window_normalize(d['points'][sel], self.window)
        c = d['colors'][sel]
        if self.transform:
            p, c = augment(p, c, rng)
        return {'points': p, 'colors': c.astype(np.float32), 'labels': d['labels'][sel].astype(np.int64)}


def axis_window_samples(xyz, window=10.0, num_points=4096, sampler='density', pool_size=4_000_000,
                        votes=3, overlap=0.5, seed=0):
    """Yield (pool_indices (num_points,), normalised points) covering the whole bridge with
    overlapping windows; every pool point is covered ``votes`` times per window it lies in.
    Returns (pool_idx, list_of_sample_index_arrays) – index arrays point into pool_idx."""
    rng = np.random.default_rng(seed)
    pool_idx, s = _sorted_pool(xyz, None, None, sampler, pool_size, seed)
    step = window * (1 - overlap)
    starts = np.arange(s[0] - window / 2 + step, s[-1] + step, step)
    samples = []
    for s0 in starts:
        a, b = np.searchsorted(s, [s0 - window / 2, s0 + window / 2])
        if b - a < 16:
            continue
        ids = np.arange(a, b)
        for _ in range(votes):
            perm = rng.permutation(ids)
            pad = (-len(perm)) % num_points
            if pad:
                perm = np.concatenate([perm, rng.choice(ids, pad, replace=len(ids) < pad)])
            samples += list(perm.reshape(-1, num_points))
    return pool_idx, samples

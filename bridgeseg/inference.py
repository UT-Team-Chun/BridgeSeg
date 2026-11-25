"""Label-free inference on a whole bridge point cloud.

Inference mirrors training (see :mod:`bridgeseg.sampling`), without using any labels:

1. normalise the whole cloud (centroid, max radius);
2. reduce it to a pool of at most ``max_points`` points with the *same sampler* the model
   was trained with (``density`` by default; read from the checkpoint);
3. split random permutations of the pool into 4,096-point samples and predict each one;
4. average the class probabilities over ``votes`` permutations;
5. copy the result to every original point by nearest-neighbour look-up.
"""
import numpy as np
import torch
import torch.nn as nn

from .model import EnhancedPointNet2
from .sampling import normalize_points, sample_pool


def _bn_batch_statistics(model, enable):
    """Use batch statistics in BatchNorm at inference (without changing stored statistics).

    The random farthest-point sampling inside the network makes the stored running
    statistics unreliable; batch statistics give stable predictions (same setting as the
    evaluation in the paper's revision).
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if enable:
                m._saved_momentum = m.momentum
                m.momentum = 0.0          # running stats are not updated
                m.train()
            else:
                m.momentum = getattr(m, '_saved_momentum', 0.1)


def load_model(checkpoint, num_classes=5, k_neighbors=None, device=None):
    """Build BridgeSeg and load weights.

    Accepts checkpoints written by ``scripts/train.py`` (dict with ``model_state_dict`` and
    ``config``) as well as plain state dicts. ``k_neighbors`` defaults to the value stored in
    the checkpoint, otherwise 32 (the paper's default).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
    state = ck
    if isinstance(ck, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in ck:
                state = ck[key]
                break
    k = k_neighbors or cfg.get('k') or 32
    model = EnhancedPointNet2(num_classes=cfg.get('num_classes', num_classes),
                              radii_scale=cfg.get('radii_scale', 1.0), k_neighbors=k,
                              fusion=cfg.get('fusion', 'index'))   # old checkpoints: 'index'
    model.load_state_dict(state)
    model.to(device).eval()
    model.bridgeseg_config = {'k_neighbors': k, **cfg}
    return model


def _logits_bcn(out, C):
    """Return logits as [B, C, N] whatever layout the network uses."""
    return out if out.shape[1] == C else out.transpose(1, 2)


@torch.no_grad()
def predict(model, xyz, rgb=None, num_points=4096, votes=3, max_points=None,
            batch_size=16, seed=0, return_probs=False, progress=False, sampler=None,
            mode=None, window=None):
    """Predict a class label for every point (no labels used).

    Parameters
    ----------
    model : network from :func:`load_model`
    xyz : (N, 3) array, coordinates of the whole bridge in metres (any origin)
    rgb : (N, 3) array in [0, 1], optional (ones if the scanner has no colour)
    votes : random passes whose probabilities are averaged
    max_points : size of the sampler pool (predictions are copied back to all points)
    sampler, mode, window : how samples are drawn; default = the values stored in the
        checkpoint (they must match training). ``mode`` is 'window' (bridge-axis slices of
        ``window`` metres) or 'global' (whole-bridge samples).

    Returns
    -------
    labels : (N,) int64; if ``return_probs`` also (N, C) float32 class probabilities
    """
    cfg = getattr(model, 'bridgeseg_config', {})
    sampler = sampler or cfg.get('sampler', 'density')
    mode = mode or cfg.get('mode', 'global')
    window = window or cfg.get('window', 10.0)
    if sampler == 'stratified':
        raise ValueError("the 'stratified' (paper) sampler needs ground-truth labels; "
                         "use a model trained with a label-free sampler")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    xyz = np.asarray(xyz)
    n_all = len(xyz)
    rgb = np.ones((n_all, 3), np.float32) if rgb is None else np.asarray(rgb, np.float32)
    C = model.num_classes

    if mode == 'window':
        from .sampling import axis_window_samples, window_normalize
        sub, samples = axis_window_samples(xyz, window, num_points, sampler,
                                           max_points or cfg.get('pool_size', 4_000_000), votes, seed=seed)
        loc = (xyz[sub] - xyz.min(0)).astype(np.float32)
        get_pts = lambda ids: np.stack([window_normalize(loc[i], window) for i in ids])
    else:
        sub = sample_pool(xyz, sampler, max_points or cfg.get('pool_size', 2_000_000), seed)
        pts = normalize_points(xyz)[sub]
        samples = []
        for _ in range(votes):
            perm = rng.permutation(len(sub))
            pad = (-len(sub)) % num_points
            if pad:
                perm = np.concatenate([perm, rng.choice(len(sub), pad, replace=len(sub) < pad)])
            samples += list(perm.reshape(-1, num_points))
        get_pts = lambda ids: pts[np.stack(ids)]
    col = rgb[sub]

    prob = torch.zeros((len(sub), C), device=device)
    model.eval()
    _bn_batch_statistics(model, True)
    try:
        it = range(0, len(samples), batch_size)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc='predict')
        for s0 in it:
            ids = samples[s0:s0 + batch_size]
            if len(ids) == 1:                          # BatchNorm needs >1 sample
                ids = ids + samples[:1]
            idx = np.stack(ids)
            p = torch.from_numpy(get_pts(ids)).to(device)
            c = torch.from_numpy(col[idx]).to(device)
            out = torch.softmax(_logits_bcn(model(p, c), C), 1).transpose(1, 2).reshape(-1, C)
            prob.index_add_(0, torch.from_numpy(idx.ravel()).to(device), out)
    finally:
        _bn_batch_statistics(model, False)
        model.eval()

    prob = (prob / prob.sum(1, keepdim=True).clamp(min=1e-9)).cpu().numpy()
    if len(sub) < n_all:                                 # back to every original point
        from scipy.spatial import cKDTree
        _, nn_idx = cKDTree(xyz[sub]).query(xyz, k=1, workers=-1)
        prob = prob[nn_idx]
    labels = prob.argmax(1).astype(np.int64)
    return (labels, prob.astype(np.float32)) if return_probs else labels

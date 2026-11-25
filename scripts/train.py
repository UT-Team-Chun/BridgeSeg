"""Train (and evaluate) BridgeSeg on .h5 point clouds.

    # train, select the best epoch on --val_dir, then evaluate on --test_dir
    python scripts/train.py --train_dir data/train --val_dir data/val --test_dir data/test --out runs/my_run

    # evaluate an existing checkpoint only
    python scripts/train.py --eval_only --checkpoint runs/my_run/best_model.pth --test_dir data/test

Each directory holds .h5 files made by scripts/las_to_h5.py. --val_dir is used only to pick the
best epoch, so it must not contain the test bridges. Outputs: best_model.pth, train.log and
result.json in --out.

Defaults (= the released checkpoint):
  --mode window   each sample is a --window (10 m) slice across the bridge axis with full width
                  and height; 'global' = whole-bridge samples as in the AiC paper
  --sampler density  label-free, density-balanced point sampling ('uniform' = plain random;
                  'stratified' = the AiC paper's label-dependent protocol, reproduction only)
  --fusion knn    decoder upsampling fix ('index' = original paper code)
  --alpha 0       MBrSOL hierarchy re-weighting off (paper: --alpha 80 --height_ref class --margin 0.3)
  --sched cosine  learning-rate schedule
Test metrics (--test_dir) for label-free samplers are computed on EVERY point of each test
bridge with the same routine as scripts/predict.py (no labels used for prediction).
"""
import argparse, datetime, json, os, sys, time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bridgeseg.model import EnhancedPointNet2, BridgeStructureLoss
from bridgeseg.dataset import BriPCDMulti
from bridgeseg.sampling import BridgeCloudDataset, AxisWindowDataset
from bridgeseg.inference import predict
from bridgeseg.metrics import confusion_matrix
import glob, h5py
from bridgeseg.io import CLASS_NAMES
from bridgeseg.inference import _bn_batch_statistics
from bridgeseg.metrics import segmentation_scores

NUM_CLASSES = 5


class AverageMeter:
    def __init__(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset, num_classes, n_sample=4000):
    counts = torch.zeros(num_classes)
    for i in torch.randperm(len(dataset))[:min(n_sample, len(dataset))]:
        lbl = torch.as_tensor(dataset[int(i)]['labels'])
        counts += torch.bincount(lbl.clamp(0, num_classes - 1), minlength=num_classes).float()
    w = (counts.sum() + 1e-6) / ((counts + 1e-6) * num_classes)
    return torch.clamp(w, 0.5, 3.0)


def set_eval(model, bn_batch_stats):
    model.eval()
    if bn_batch_stats:
        _bn_batch_statistics(model, True)


@torch.no_grad()
def evaluate(model, loader, device, seed, bn_batch_stats, max_batches=0):
    torch.manual_seed(seed)
    set_eval(model, bn_batch_stats)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        pts, col, lbl = batch['points'].to(device), batch['colors'].to(device), batch['labels'].to(device)
        out = model(pts, col)
        pred = (out if out.shape[1] == NUM_CLASSES else out.transpose(1, 2)).argmax(1)
        t, p = lbl.reshape(-1), pred.reshape(-1)
        k = (t >= 0) & (t < NUM_CLASSES)
        cm += torch.bincount(t[k] * NUM_CLASSES + p[k], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    if bn_batch_stats:
        _bn_batch_statistics(model, False)
    return segmentation_scores(cm.cpu().numpy(), CLASS_NAMES)


def make_loader(data_dir, sample_rate, batch_size, train, workers, sampler='density', pool_size=2_000_000,
                mode='global', window=10.0):
    if sampler == 'stratified':
        ds = BriPCDMulti(data_dir=data_dir, num_points=4096, block_size=1.0, sample_rate=sample_rate, transform=train)
    elif mode == 'window':
        ds = AxisWindowDataset(data_dir, 4096, window, sampler, pool_size, sample_rate, transform=train)
    else:
        ds = BridgeCloudDataset(data_dir, 4096, sampler, pool_size, sample_rate, transform=train)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                          pin_memory=True, drop_last=train)


def build_model(a):
    """BridgeSeg network (other networks can be plugged in here for comparisons)."""
    return EnhancedPointNet2(NUM_CLASSES, radii_scale=a.radii_scale, k_neighbors=a.k, fusion=a.fusion)


def main(argv=None, build=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train_dir'); ap.add_argument('--val_dir'); ap.add_argument('--test_dir')
    ap.add_argument('--out', default='runs/bridgeseg')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--k', type=int, default=32, help='KNN size of the BriStruc-Embedding')
    ap.add_argument('--radii_scale', type=float, default=1.0, help='scales all ball-query radii')
    ap.add_argument('--fusion', choices=['knn', 'index'], default='knn',
                    help="decoder upsampling before the fusion head ('index' = original paper code)")
    ap.add_argument('--sched', choices=['cosine', 'plateau'], default='cosine',
                    help="learning-rate schedule ('plateau' = paper: x0.1 after 5 epochs without gain)")
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='MBrSOL hierarchy weight (0 = class-weighted CE only; paper: 80)')
    ap.add_argument('--height_ref', choices=['class', 'sample'], default='sample',
                    help="MBrSOL height frame ('class' = paper code, 'sample' = common z-range of the sample)")
    ap.add_argument('--margin', type=float, default=0.3, help='MBrSOL relative height margin')
    ap.add_argument('--warmup', type=int, default=4, help='epochs to ramp alpha from 0 to its target')
    ap.add_argument('--weight_clamp', type=float, default=5.0, help='clamp loss class weights to [0.3, x] (0 = off)')
    ap.add_argument('--clip', type=float, default=5.0, help='gradient-norm clipping (0 = off)')
    ap.add_argument('--sample_rate', type=float, default=0.4, help='blocks per file = N * rate / 4096')
    ap.add_argument('--val_sample_rate', type=float, default=0.1, help='validation samples per file = N * rate / 4096')
    ap.add_argument('--test_sample_rate', type=float, default=0.5, help='(stratified sampler only)')
    ap.add_argument('--sampler', choices=['density', 'uniform', 'stratified'], default='density')
    ap.add_argument('--pool_size', type=int, default=0, help='sampler pool per bridge (0 = 2M global / 4M window)')
    ap.add_argument('--mode', choices=['window', 'global'], default='window',
                    help="'window': slices of --window metres along the bridge axis (full height); "
                         "'global': whole-bridge samples as in the paper")
    ap.add_argument('--window', type=float, default=10.0, help='slice length in metres (mode=window)')
    ap.add_argument('--votes', type=int, default=3, help='test-time passes (label-free samplers)')
    ap.add_argument('--bn_eval', choices=['batch', 'running'], default='batch',
                    help="BatchNorm statistics at evaluation; 'batch' is robust and recommended")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_batches', type=int, default=0, help='limit batches per epoch (quick tests)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--checkpoint', default=None, help='weights to evaluate (with --eval_only)')
    a = ap.parse_args(argv)
    if build is not None:
        globals()['build_model'] = build
    a.pool_size = a.pool_size or (4_000_000 if a.mode == 'window' else 2_000_000)

    set_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bn_batch = a.bn_eval == 'batch'

    def log(msg):
        line = f'[{datetime.datetime.now():%H:%M:%S}] {msg}'
        print(line, flush=True)
        with open(out / 'train.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    config = {'num_classes': NUM_CLASSES, 'k': a.k, 'radii_scale': a.radii_scale, 'fusion': a.fusion, 'sched': a.sched, 'alpha': a.alpha, 'height_ref': a.height_ref,
              'margin': a.margin, 'epochs': a.epochs, 'batch_size': a.batch_size, 'lr': a.lr,
              'sample_rate': a.sample_rate, 'sampler': a.sampler, 'mode': a.mode, 'window': a.window, 'pool_size': a.pool_size, 'seed': a.seed, 'train_dir': a.train_dir, 'val_dir': a.val_dir}
    model = build_model(a).to(device)
    result = {'config': config, 'n_params': sum(p.numel() for p in model.parameters())}
    log(f'config {config} | params {result["n_params"]:,} | device {device}')

    if not a.eval_only:
        assert a.train_dir and a.val_dir, '--train_dir and --val_dir are required for training'
        train_ds, train_loader = make_loader(a.train_dir, a.sample_rate, a.batch_size, True, a.workers, a.sampler, a.pool_size, a.mode, a.window)
        val_ds, val_loader = make_loader(a.val_dir, a.val_sample_rate, a.batch_size, False, a.workers, a.sampler, a.pool_size, a.mode, a.window)
        log(f'train blocks {len(train_ds)} | val blocks {len(val_ds)}')
        cw = compute_class_weights(train_ds, NUM_CLASSES)
        log(f'class weights {[round(x, 3) for x in cw.tolist()]}')
        criterion = BridgeStructureLoss(num_classes=NUM_CLASSES, alpha=a.alpha, rel_margin=a.margin,
                                        class_weights=cw, weight_clamp=a.weight_clamp or None,
                                        height_ref=a.height_ref).to(device)
        optimizer = optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        if a.sched == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=a.lr * 0.01)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        best, t0, hist = -1.0, time.time(), []
        for epoch in range(1, a.epochs + 1):
            criterion.alpha = a.alpha * min(1.0, epoch / max(1, a.warmup))
            model.train(); tl, ta = AverageMeter(), AverageMeter(); te = time.time()
            for bi, batch in enumerate(train_loader):
                if a.max_batches and bi >= a.max_batches:
                    break
                pts, col, lbl = batch['points'].to(device), batch['colors'].to(device), batch['labels'].to(device)
                optimizer.zero_grad()
                logits = model(pts, col)
                if logits.shape[1] != NUM_CLASSES:
                    logits = logits.transpose(1, 2)
                loss = criterion(logits, lbl, points=pts)
                loss.backward()
                if a.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                optimizer.step()
                tl.update(loss.item()); ta.update(logits.argmax(1).eq(lbl).float().mean().item())
            val = evaluate(model, val_loader, device, a.seed, bn_batch, a.max_batches)
            scheduler.step() if a.sched == 'cosine' else scheduler.step(val['mIoU'])
            hist.append({'epoch': epoch, 'loss': tl.avg, 'train_acc': 100 * ta.avg, 'val_mIoU': val['mIoU'], 'val_OA': val['OA']})
            log(f'epoch {epoch}/{a.epochs} | loss {tl.avg:.4f} | train acc {100 * ta.avg:.2f}% | '
                f'val OA {val["OA"]:.2f}% mIoU {val["mIoU"]:.2f}% | {(time.time() - te) / 60:.1f} min')
            if val['mIoU'] > best:
                best = val['mIoU']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config,
                            'val_mIoU': best}, out / 'best_model.pth')
        result.update({'best_val_mIoU': best, 'history': hist, 'train_minutes': (time.time() - t0) / 60})
        ckpt = out / 'best_model.pth'
    else:
        assert a.checkpoint, '--eval_only needs --checkpoint'
        ckpt = Path(a.checkpoint)

    if a.test_dir:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state))
        cfg = state.get('config', {}) if isinstance(state, dict) else {}
        sampler, mode, window = cfg.get('sampler', a.sampler), cfg.get('mode', a.mode), cfg.get('window', a.window)
        if sampler == 'stratified':
            test_ds, test_loader = make_loader(a.test_dir, a.test_sample_rate, a.batch_size, False, a.workers, sampler)
            s = evaluate(model, test_loader, device, a.seed, bn_batch)
            log(f'TEST [stratified, label-dependent protocol] ({len(test_ds)} samples)')
        else:                                   # label-free, every point of every test bridge
            model.num_classes = NUM_CLASSES
            cm, per_file = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64), {}
            for path in sorted(glob.glob(os.path.join(a.test_dir, '*.h5'))):
                with h5py.File(path, 'r') as f:
                    xyz = np.asarray(f['points']); gt = np.asarray(f['labels'])
                    rgb = np.asarray(f['colors'], np.float32) if 'colors' in f else None
                pred = predict(model, xyz, rgb, votes=a.votes, max_points=cfg.get('pool_size', a.pool_size),
                               sampler=sampler, mode=mode, window=window, seed=a.seed)
                c = confusion_matrix(gt, pred, NUM_CLASSES); cm += c
                per_file[Path(path).stem] = sf = segmentation_scores(c, CLASS_NAMES)
                log(f'  {Path(path).stem}: {len(xyz):,} pts  OA {sf["OA"]:.2f}  mIoU {sf["mIoU"]:.2f}')
            s = segmentation_scores(cm, CLASS_NAMES); s['per_file'] = per_file
            log(f'TEST [{mode}/{sampler}, label-free, all points]')
        result['test'] = s
        log(f'TEST OA {s["OA"]:.2f}  mIoU {s["mIoU"]:.2f}  mAcc {s["mAcc"]:.2f}')
        log(f'per-class IoU {s["IoU"]}')
    json.dump(result, open(out / 'result.json', 'w'), indent=2)
    log(f'wrote {out / "result.json"}')


if __name__ == '__main__':
    main()

"""Segment new bridge point clouds with a trained BridgeSeg model (no labels needed).

    python scripts/predict.py --checkpoint checkpoints/bridgeseg_ds2_window.pth --input bridge.las --output results/

Writes ``<name>_pred.las`` with the predicted class in the LAS ``classification`` field and
a ``confidence`` extra field. If the input already contains labels, OA / mIoU / per-class
IoU are also computed and saved to ``<name>_metrics.json``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridgeseg import load_model, predict, read_point_cloud, write_las, CLASS_NAMES
from bridgeseg.metrics import confusion_matrix, segmentation_scores

EXTS = ('.las', '.laz', '.h5', '.npy', '.npz', '.txt', '.csv', '.xyz', '.pts')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True, help='a point cloud file or a directory of files')
    ap.add_argument('--output', default='results')
    ap.add_argument('--votes', type=int, default=3, help='random passes averaged per point')
    ap.add_argument('--max_points', type=int, default=0,
                    help='size of the sampler pool used for inference (0 = value stored in the checkpoint)')
    ap.add_argument('--sampler', default=None, choices=['density', 'uniform'],
                    help='point sampler; default = the one stored in the checkpoint (must match training)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--no_color', action='store_true', help='ignore RGB (feed ones), e.g. for intensity-only scans')
    ap.add_argument('--evaluate', action='store_true', help='compute metrics if the input has labels (default: auto)')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    model = load_model(a.checkpoint, device=a.device)
    print(f'model loaded ({sum(p.numel() for p in model.parameters()):,} parameters, '
          f'K = {model.bridgeseg_config["k_neighbors"]})')
    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    for f in files:
        t = time.time()
        xyz, rgb, gt = read_point_cloud(f)
        if a.no_color:
            rgb = np.ones_like(rgb)
        labels, prob = predict(model, xyz, rgb, votes=a.votes, max_points=a.max_points or None, sampler=a.sampler,
                               batch_size=a.batch_size, return_probs=True, progress=True)
        write_las(out / f'{f.stem}_pred.las', xyz, labels, rgb, extra={'confidence': prob.max(1)})
        dist = {CLASS_NAMES[i]: round(100 * float((labels == i).mean()), 2) for i in range(len(CLASS_NAMES))}
        print(f'{f.name}: {len(xyz):,} points, {time.time() - t:.0f} s, predicted share (%) {dist}')
        if gt is not None and (a.evaluate or gt.max() > 0):
            s = segmentation_scores(confusion_matrix(gt, labels))
            print(f'  OA {s["OA"]:.2f}  mIoU {s["mIoU"]:.2f}  IoU {s["IoU"]}')
            json.dump(s, open(out / f'{f.stem}_metrics.json', 'w'), indent=2)


if __name__ == '__main__':
    main()

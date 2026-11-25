"""Convert point clouds (.las/.laz/.npy/.txt/...) to the .h5 training format.

    python scripts/las_to_h5.py --input raw_las_dir --output data/train
    python scripts/las_to_h5.py --input bridge.las --output data/test --label_map "7:0,5:4,6:4"

The .h5 file holds ``points`` (N x 3), ``colors`` (N x 3, in [0, 1]) and ``labels`` (N).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridgeseg.io import read_point_cloud, write_h5

EXTS = ('.las', '.laz', '.npy', '.npz', '.txt', '.csv', '.xyz', '.pts', '.h5')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='a file or a directory of files')
    ap.add_argument('--output', required=True, help='output directory for .h5 files')
    ap.add_argument('--label_field', default='classification', help='LAS field holding the labels')
    ap.add_argument('--label_map', default='', help='remap labels, e.g. "7:0,5:4,6:4"')
    ap.add_argument('--num_classes', type=int, default=5,
                    help='labels outside [0, num_classes) are set to 0 (background)')
    a = ap.parse_args()

    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    remap = dict(tuple(int(x) for x in kv.split(':')) for kv in a.label_map.split(',') if kv)
    for f in files:
        xyz, rgb, lab = read_point_cloud(f, a.label_field)
        if lab is not None:
            if remap:
                new = lab.copy()
                for k, v in remap.items():
                    new[lab == k] = v
                lab = new
            bad = (lab < 0) | (lab >= a.num_classes)
            if bad.any():
                print(f'  {f.name}: {bad.sum()} points with labels outside 0..{a.num_classes - 1} -> 0')
                lab[bad] = 0
        write_h5(out / (f.stem + '.h5'), xyz, rgb, lab)
        dist = '' if lab is None else ' labels ' + str(dict(zip(*np.unique(lab, return_counts=True))))
        print(f'{f.name}: {len(xyz):,} points{dist}')


if __name__ == '__main__':
    main()

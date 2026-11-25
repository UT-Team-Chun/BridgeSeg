# BridgeSeg — structure-oriented semantic segmentation of bridge point clouds

Official code for **"Structure-oriented deep learning for semantic segmentation of bridge point
clouds"** (Y. Chen, C. Lin, T. Yamane, S. Kubo, P.-J. Chun, *Automation in Construction*, 2026).

BridgeSeg labels every point of a bridge LiDAR / photogrammetry point cloud as one of five
structural classes. It combines:

- a **BriStruc-Embedding** of local geometry (k-nearest-neighbour structure encoding);
- a **multi-scale PointNet++ backbone** (3 set-abstraction levels, two ball-query radii each)
  with an attentive feature-propagation decoder and a multi-scale fusion head;
- the **MBrSOL loss** option, which re-weights classes whose predictions break the vertical
  hierarchy of a bridge (pier/abutment below girder below deck below superstructure).

The network has 7.94 M parameters. A checkpoint trained on the Cambridge bridge benchmark
(DS-2) is included.

---

## TL;DR

```bash
pip install -r requirements.txt                       # after installing PyTorch
python scripts/predict.py --checkpoint checkpoints/bridgeseg_ds2_window.pth --input my_bridge.las --output results/
# -> results/my_bridge_pred.las  (class in the LAS "classification" field, plus a "confidence" field)
```

```python
import bridgeseg
xyz, rgb, _ = bridgeseg.read_point_cloud("my_bridge.las")   # (N,3) metres, (N,3) in [0,1]
model = bridgeseg.load_model("checkpoints/bridgeseg_ds2_window.pth")
labels = bridgeseg.predict(model, xyz, rgb)                  # (N,) int64 in 0..4
```

No labels are needed for prediction. Input = **one bridge per file, coordinates in metres**.

---

## 1. Classes

| id | name | what it covers |
|---:|---|---|
| 0 | `background` | ground, vegetation, vehicles, anything not part of the bridge |
| 1 | `pier_abutment` | piers, columns, abutments |
| 2 | `girder` | main girders / beams |
| 3 | `deck` | deck slab |
| 4 | `superstructure` | parapets, railings, barriers and other members above the deck |

The ids are fixed; `bridgeseg.CLASS_NAMES` holds the names in this order.

## 2. Installation

```bash
git clone https://github.com/UT-Team-Chun/BridgeSeg.git
cd BridgeSeg
# 1) install PyTorch for your CUDA version, see https://pytorch.org/get-started/locally/
# 2) the rest
pip install -r requirements.txt
```

The code is pure PyTorch, with no custom CUDA extensions to compile. It was tested with
Python 3.10, torch 2.1.0, CUDA 12.1 and an RTX 4090 on Windows 11. It also runs on Linux and on
the CPU, but slowly.

## 3. Repository layout

```
bridgeseg/                 Python package (import bridgeseg)
  model.py                 EnhancedPointNet2 (= BridgeSeg) network and BridgeStructureLoss (MBrSOL)
  modules.py               BriStruc-Embedding and geometric feature modules
  pointnet2_utils.py       multi-scale set abstraction / feature propagation layers
  sampling.py              label-free samplers, bridge-axis windows, training datasets
  dataset.py               BriPCDMulti: the paper's label-stratified loader (reproduction only)
  inference.py             load_model(), predict()
  io.py                    read_point_cloud(), write_las(), write_h5(), CLASS_NAMES
  metrics.py               confusion_matrix(), segmentation_scores()  (OA, mIoU, mAcc, per-class IoU)
scripts/
  las_to_h5.py             convert labelled point clouds to the .h5 training format
  train.py                 train + validate + test
  predict.py               segment new point clouds and write .las results
checkpoints/
  bridgeseg_ds2_window.pth released weights (DS-2, window mode, see §8)
```

## 4. Data format

**For prediction**, any of `.las .laz .h5 .npy .npz .txt .csv .xyz .pts` works:

- `.las/.laz`: x, y, z; RGB if present; labels from the `classification` field if present.
- `.npy/.npz/.txt/.csv`: columns `x y z [r g b] [label]`. Colours may be 0–1 or 0–255.
- If the scan has no colour, all-ones colour is used. You can force this with `--no_color`.
- Coordinates must be in **metres** (window mode cuts 10 m slices). Any origin is fine.

**For training**, convert to `.h5`, one file per bridge, with these datasets:

| dataset | shape | dtype | meaning |
|---|---|---|---|
| `points` | N × 3 | float | coordinates in metres (any origin), z pointing up |
| `colors` | N × 3 | float32 | RGB in [0, 1] |
| `labels` | N | int | class id 0–4 (see §1) |

```bash
python scripts/las_to_h5.py --input raw_las/ --output data/train
# remap your own label ids to the 5-class scheme, e.g. 7 -> 0 and 5, 6 -> 4:
python scripts/las_to_h5.py --input raw_las/ --output data/train --label_map "7:0,5:4,6:4"
```

Arrange the data as `data/train/*.h5`, `data/val/*.h5` and `data/test/*.h5`, splitting by bridge
(never split one bridge across sets, and never use test bridges for validation).

## 5. How a point cloud is fed to the network

Every network input is **4,096 points**. Two choices define which 4,096 points: the **mode**
(which part of the bridge a sample covers) and the **sampler** (how points are drawn inside it).

### 5.1 Mode

| mode | a sample is… | coordinates | when to use |
|---|---|---|---|
| `window` (default) | a slice **across** the bridge axis, `--window` metres long (default 10 m), with the **full width and full height** | centred on the slice and divided by `window / 2` (metric scale is kept) | general use; best accuracy (§8) |
| `global` | the **whole bridge** | centred and scaled into the unit sphere (as in the paper) | very small bridges / reproducing the paper |

The bridge axis is found automatically (first principal axis of the cloud in plan view), so the
cloud does not need to be aligned. Because each window keeps the whole cross-section and height,
every sample still contains the complete vertical order pier → girder → deck → superstructure,
but at 6–8× the point density of a whole-bridge sample (for the 60–85 m DS-2 bridges), so thin members (railings, bracing)
are resolved much better.

At inference the bridge is covered by windows with 50 % overlap; each window is predicted
`votes` times with different random point subsets, probabilities are averaged, and the result
is copied to every original point by nearest-neighbour lookup.

### 5.2 Sampler

| sampler | label-free? | idea | use it when |
|---|---|---|---|
| `density` (default) | yes | probability ∝ 1 / (points in the point's voxel; voxel = bbox diagonal / 300), so every occupied region gets a similar number of samples | general use; scans with very uneven density, e.g. dense ground near the scanner and thin piers far away |
| `uniform` | yes | every point equally likely | the point density is already even, e.g. after voxel down-sampling |
| `stratified` | **no** | fixed class proportions (0.1/0.2/0.2/0.2/0.3) chosen **from ground-truth labels** | only to reproduce the evaluation protocol of the paper (§8.3) |

**Rule: train and predict with the same mode, window and sampler.** The checkpoint stores
them, and `predict()` uses them automatically.

### 5.3 Swapping the sampler for your target

The sampler is not the core of the method and can be changed to suit the structure you care
about, e.g. height-balanced for tall piers or curvature-aware for slender members. Write a
function `f(xyz) -> log_weights` (one value per point, higher = more likely) and register it:

```python
import numpy as np
from bridgeseg import sampling

def height_balanced(xyz, bins=40):
    z = xyz[:, 2]
    b = np.clip(((z - z.min()) / (np.ptp(z) + 1e-9) * bins).astype(int), 0, bins - 1)
    return -np.log(np.bincount(b, minlength=bins)[b])     # each height band equally likely

sampling.SAMPLERS['height'] = height_balanced
```

Then train with `--sampler height` (add the name to the `--sampler` choices in
`scripts/train.py` / `predict.py`). Likewise `--window` can be tuned to the span length of
your bridges. Retrain whenever you change mode, window or sampler.

## 6. Prediction

```bash
python scripts/predict.py --checkpoint checkpoints/bridgeseg_ds2_window.pth --input data/new_bridges/ --output results/
```

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | required | weights (`.pth`) |
| `--input` | required | a file or a directory of files (one bridge per file) |
| `--output` | `results` | output directory |
| `--votes` | 3 | random passes averaged per window; more is smoother but slower |
| `--max_points` | from checkpoint | size of the sampler pool (4 M for the released model) |
| `--sampler` | from checkpoint | override the sampler (normally leave unset) |
| `--no_color` | off | ignore RGB, e.g. for intensity-only scans |
| `--batch_size` | 16 | samples per GPU batch; lower it if you run out of GPU memory |

**Outputs** per input file:

- `<name>_pred.las`: the original points; the predicted class in `classification`, the
  maximum softmax probability in the extra field `confidence`.
- `<name>_metrics.json`: only if the input contains labels. Holds OA, mIoU, mAcc,
  per-class IoU and the confusion matrix.

A 33 M-point, 80 m bridge takes about 6–7 min on an RTX 4090 (`--votes 1` ≈ 3× faster).

**Python API**

```python
bridgeseg.load_model(checkpoint, num_classes=5, k_neighbors=None, device=None) -> model
bridgeseg.predict(model, xyz, rgb=None, votes=3, max_points=None, batch_size=16, seed=0,
                  return_probs=False, sampler=None, mode=None, window=None) -> labels[, probs]
bridgeseg.read_point_cloud(path) -> (xyz, rgb, labels_or_None)
bridgeseg.write_las(path, xyz, labels, rgb=None, extra=None)
bridgeseg.confusion_matrix(gt, pred, num_classes=5) -> (C, C) array
bridgeseg.segmentation_scores(cm) -> {'OA', 'mIoU', 'mAcc', 'IoU': {class: value}, ...}   # percent
```

`sampler`, `mode`, `window` and `max_points` default to the values stored in the checkpoint.

## 7. Training on your own data

```bash
python scripts/train.py --train_dir data/train --val_dir data/val --test_dir data/test --out runs/my_run
```

| flag | default | meaning |
|---|---|---|
| `--mode` | `window` | `window` / `global` (see §5.1) |
| `--window` | 10.0 | slice length in metres (window mode) |
| `--sampler` | `density` | `density` / `uniform` / `stratified` (see §5.2) |
| `--epochs` | 15 | the best epoch on `--val_dir` (mIoU) is kept |
| `--batch_size` / `--lr` | 16 / 1e-3 | Adam, weight decay 1e-4 |
| `--sched` | `cosine` | `cosine` or `plateau` (paper: ×0.1 after 5 epochs without gain) |
| `--k` | 32 | KNN size of the BriStruc-Embedding |
| `--radii_scale` | 1.0 | multiplies all ball-query radii |
| `--fusion` | `knn` | decoder upsampling before the fusion head (`index` = original paper code, see §8.4) |
| `--alpha` | 0 | MBrSOL hierarchy weight (0 = class-weighted cross-entropy only; paper: 80) |
| `--height_ref` | `sample` | MBrSOL height frame (`class` = original paper code, see §8.4) |
| `--margin` | 0.3 | MBrSOL relative-height margin (use ~0.02 with `--height_ref sample`) |
| `--sample_rate` | 0.4 | samples per bridge per epoch = N_points × rate / 4096 |
| `--val_sample_rate` | 0.1 | the same, for validation |
| `--pool_size` | 0 | sampler pool per bridge (0 = 4 M window / 2 M global) |
| `--votes` | 3 | test-time passes |
| `--workers` | 4 | data-loader processes |
| `--seed` | 42 | random seed |
| `--eval_only --checkpoint X` | | only test an existing checkpoint |

**Outputs in `--out`:**

- `best_model.pth`, which stores the weights and the `config` (mode, window, sampler, …).
- `train.log`.
- `result.json`, with the training history and test metrics.

For label-free samplers the test metrics are computed on **every point** of every test bridge
with the same routine as `predict.py`; this is the number to expect on new data. The first run
caches each bridge's sampler pool next to its `.h5` file (`*.axis_density_pool4000000.npz`);
delete those files if the data change.

About 19 min per epoch on DS-2 (7 training bridges, ~13 k samples per epoch) on an RTX 4090,
i.e. ~5 h for 15 epochs.

## 8. Results

### 8.1 Released checkpoint and comparison (DS-2, label-free, every point)

Cambridge bridge benchmark (DS-2). Train: cb1, cb2, cb3, cb4, cb5, cb7, cb8. Validation (epoch
selection only): cb10. Test: cb6 + cb9 (67 M points). All networks use **the same pipeline**:
window mode (10 m), density sampler, the same loss (class-weighted cross-entropy,
`--alpha 0`), 15 epochs, cosine schedule, 3 votes. Test scores are computed on every point
without using labels. Mean of 2 seeds (42, 7); individual runs in brackets.

| network | params | test OA | **test mIoU** | background | pier/abut. | girder | deck | superstr. | cb10 val mIoU† |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BridgeSeg** | 7.9 M | 97.4 | **91.6** (90.9 / 92.3) | 96.4 | **87.1** | 96.3 | 92.8 | **85.5** | **83.8** |
| Transformer‡ | 19.0 M | **97.7** | 91.1 (90.7 / 91.5) | **97.0** | 85.3 | **97.3** | **95.2** | 80.8 | 73.2 |
| PointNet++ | 0.8 M | 92.2 | 79.7 (1 seed) | 90.2 | 75.3 | 85.7 | 74.9 | 72.2 | 64.9 |

† best validation mIoU on the unseen bridge cb10 (sampled points), i.e. how well the network
transfers to a bridge it has never seen. ‡ a global self-attention point transformer over the
4,096 points of each window (the `PointTransformerV3` baseline of the paper's comparison code,
not the official PTv3 with serialisation and grid pooling).

`checkpoints/bridgeseg_ds2_window.pth` is the seed-42 BridgeSeg run (chosen by its cb10
validation score): test OA 97.15, mIoU 90.93 (cb6 88.63, cb9 92.28).

Take-aways:

- BridgeSeg and the 2.4× larger transformer are on par on the test bridges; BridgeSeg is better
  on the thin / rare classes (pier, superstructure) and transfers clearly better to the unseen
  bridge cb10 (+10 mIoU). Both are far ahead of plain PointNet++ (+12 mIoU).
- How points are fed matters as much as the network: with whole-bridge samples (`--mode
  global`) the same BridgeSeg reaches 82.7 test mIoU (superstructure 48.7) because 4,096 points
  over a whole bridge are too sparse for railings and parapets.

### 8.2 Ablation (BridgeSeg, window mode, test mIoU, single seed)

| change | test mIoU |
|---|---:|
| released setting (`--fusion knn --alpha 0`) | 90.9 / 92.3 (2 seeds) |
| + MBrSOL, original height frame (`--alpha 80 --height_ref class --margin 0.3`) | 88.7 |
| + MBrSOL, common height frame (`--alpha 20 --height_ref sample --margin 0.02`) | 90.4 |
| original decoder fusion (`--fusion index`) + MBrSOL α 80, plateau schedule† | 87.7 |
| whole-bridge samples (`--mode global`) + MBrSOL α 80, plateau schedule† | 82.7 |

† trained before the validation split was fixed (epoch selected on cb6/cb9), so slightly optimistic.

With dense window samples the network already respects the vertical hierarchy, and the MBrSOL
re-weighting does not add accuracy (with the original height frame it costs ~2 mIoU). It stays
available through `--alpha`.

### 8.3 The paper's evaluation protocol

**Read this before comparing with the paper.** The experiments in the paper used the loader in
`bridgeseg/dataset.py` (`BriPCDMulti`, `--sampler stratified --mode global`). For both training
**and testing**, it draws each 4,096-point whole-bridge sample with fixed class proportions
chosen from the ground-truth labels, and the scores are computed on those samples. That
protocol cannot be applied to unlabelled data, and a model trained with it depends on
label-stratified input: on label-free input its accuracy drops sharply (≈ 16–20 mIoU on cb6/cb9).
`scripts/train.py --sampler stratified --mode global --fusion index --alpha 80 --height_ref
class --sched plateau` reproduces the paper's setting. For any use on new data, use a
label-free model such as the released checkpoint; its numbers in §8.1 are measured on every
point of the test bridges without labels.

### 8.4 Changes with respect to the paper's code

Two implementation issues were found while preparing this release; both options are kept
so that the paper's checkpoints and numbers remain reproducible:

- **Decoder fusion (`--fusion`).** The original fusion head brought the level-1/level-2 decoder
  features back to all points with `F.interpolate` along the point-index axis. The coarse
  levels come from farthest-point sampling, whose order is unrelated to the input order, so
  each point received the features of an arbitrary coarse point. `knn` interpolates from the
  3 nearest coarse points instead (same parameters). Old checkpoints load with `index`
  automatically.
- **MBrSOL height frame (`--height_ref`).** The original loss normalised each class's height
  by that class's own bounding box, so the "A below B" checks did not compare heights on a
  common scale. `sample` normalises all classes by the z-range of the whole sample.

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| almost everything predicted as one class | the model was trained with `stratified` sampling, or the file holds several bridges / is not in metres; use the released checkpoint and one bridge per file |
| strange results at the bridge ends or on curved bridges | window mode follows one straight axis; split long curved bridges into roughly straight parts |
| CUDA out of memory | `--batch_size 8` (or 4) |
| very slow on a huge cloud | `--votes 1`, or a smaller `--max_points` (e.g. 2000000) |
| colours look wrong / no RGB | `--no_color`; colours are expected in [0, 1] (0–255 and 16-bit LAS are rescaled automatically) |
| class ids of your data differ | remap with `las_to_h5.py --label_map` |
| model does not transfer to a very different bridge type | fine-tune: train on a few labelled bridges of the new type, from scratch or from the released checkpoint |

## 10. Citation

```bibtex
@article{chen2026bridgeseg,
  title   = {Structure-oriented deep learning for semantic segmentation of bridge point clouds},
  author  = {Chen, Yu and Lin, Chao and Yamane, Tatsuro and Kubo, Shiori and Chun, Pang-jo},
  journal = {Automation in Construction},
  year    = {2026}
}
```

License: MIT (see `LICENSE`).

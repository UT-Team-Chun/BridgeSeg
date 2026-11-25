"""Segmentation metrics: OA, mAcc, mIoU, per-class IoU / precision / recall / F1."""
import numpy as np

from .io import CLASS_NAMES


def confusion_matrix(gt, pred, num_classes=5):
    """Rows = ground truth, columns = prediction. Labels outside [0, num_classes) are ignored."""
    gt = np.asarray(gt).astype(np.int64).ravel()
    pred = np.asarray(pred).astype(np.int64).ravel()
    m = (gt >= 0) & (gt < num_classes)
    return np.bincount(gt[m] * num_classes + pred[m], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def segmentation_scores(cm, class_names=CLASS_NAMES):
    """Metrics in percent. mIoU / mAcc average only over classes present in the ground truth."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt, pr = cm.sum(1), cm.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1e-9)
    rec = tp / np.maximum(gt, 1e-9)
    prec = tp / np.maximum(pr, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = gt > 0
    names = list(class_names)[:len(cm)]
    return {
        'OA': 100 * tp.sum() / max(cm.sum(), 1e-9),
        'mAcc': 100 * rec[present].mean(),
        'mIoU': 100 * iou[present].mean(),
        'IoU': {n: round(100 * iou[i], 2) for i, n in enumerate(names)},
        'Precision': {n: round(100 * prec[i], 2) for i, n in enumerate(names)},
        'Recall': {n: round(100 * rec[i], 2) for i, n in enumerate(names)},
        'F1': {n: round(100 * f1[i], 2) for i, n in enumerate(names)},
        'support': {n: int(gt[i]) for i, n in enumerate(names)},
        'confusion_matrix': cm.astype(np.int64).tolist(),
    }

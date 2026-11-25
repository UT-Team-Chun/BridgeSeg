"""Reading and writing point clouds (.las/.laz, .h5, .npy/.npz, .txt/.csv)."""
from pathlib import Path

import numpy as np

# Five-class scheme of the paper. Integer label = index in this list.
CLASS_NAMES = ['background', 'pier_abutment', 'girder', 'deck', 'superstructure']


def read_point_cloud(path, label_field='classification'):
    """Read a point cloud file.

    Returns
    -------
    xyz : (N, 3) float64   coordinates in metres
    rgb : (N, 3) float32   colors scaled to [0, 1] (all ones if the file has no color)
    labels : (N,) int64 or None   ground-truth labels if present
    """
    path = Path(path)
    ext = path.suffix.lower()
    labels = None
    if ext in ('.las', '.laz'):
        import laspy
        las = laspy.read(str(path))
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        names = set(las.point_format.dimension_names)
        if {'red', 'green', 'blue'} <= names:
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
            rgb /= 65535.0 if rgb.max() > 255 else 255.0
        else:
            rgb = np.ones_like(xyz, dtype=np.float32)
        if label_field in names:
            labels = np.asarray(las[label_field]).astype(np.int64)
    elif ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            xyz = np.asarray(f['points'], dtype=np.float64)
            rgb = np.asarray(f['colors'], dtype=np.float32) if 'colors' in f else np.ones_like(xyz, np.float32)
            labels = np.asarray(f['labels']).astype(np.int64) if 'labels' in f else None
    elif ext in ('.npy', '.npz', '.txt', '.csv', '.xyz', '.pts'):
        if ext == '.npy':
            arr = np.load(path)
        elif ext == '.npz':
            d = np.load(path)
            arr = d[d.files[0]]
        else:
            arr = np.loadtxt(path, delimiter=',' if ext == '.csv' else None)
        # columns: x y z [r g b] [label]
        xyz = arr[:, :3].astype(np.float64)
        rgb = arr[:, 3:6].astype(np.float32) if arr.shape[1] >= 6 else np.ones_like(xyz, np.float32)
        if rgb.max() > 1.0:
            rgb /= 65535.0 if rgb.max() > 255 else 255.0
        if arr.shape[1] in (4, 7):
            labels = arr[:, -1].astype(np.int64)
    else:
        raise ValueError(f'Unsupported file type: {path}')
    return xyz, rgb, labels


def write_las(path, xyz, labels, rgb=None, extra=None):
    """Write points with predicted labels (stored in the LAS `classification` field).

    ``extra`` may map a field name to an (N,) float array (e.g. per-point confidence).
    """
    import laspy
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.offsets = np.floor(xyz.min(0))
    header.scales = np.array([0.001, 0.001, 0.001])
    for name in (extra or {}):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = np.asarray(labels, dtype=np.uint8)
    if rgb is not None:
        c = np.clip(np.asarray(rgb, dtype=np.float64), 0, 1) * 65535
        las.red, las.green, las.blue = c[:, 0].astype(np.uint16), c[:, 1].astype(np.uint16), c[:, 2].astype(np.uint16)
    for name, val in (extra or {}).items():
        las[name] = np.asarray(val, dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def write_h5(path, xyz, rgb, labels=None):
    """Write the training format used by :class:`bridgeseg.dataset.BriPCDMulti`."""
    import h5py
    with h5py.File(path, 'w') as f:
        f.create_dataset('points', data=np.asarray(xyz, np.float64), compression='gzip')
        f.create_dataset('colors', data=np.asarray(rgb, np.float32), compression='gzip')
        lab = np.zeros(len(xyz), np.int64) if labels is None else np.asarray(labels, np.int64)
        f.create_dataset('labels', data=lab, compression='gzip')

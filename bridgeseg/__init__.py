"""BridgeSeg: structure-oriented semantic segmentation of bridge point clouds.

Quick use::

    import bridgeseg
    model = bridgeseg.load_model('checkpoints/bridgeseg_ds2.pth')          # trained network
    labels = bridgeseg.predict(model, xyz, rgb)                              # (N,) int labels

See README.md for the full documentation.
"""
from .model import BridgeSeg, EnhancedPointNet2, BridgeStructureLoss
from .io import read_point_cloud, write_las, CLASS_NAMES
from .inference import load_model, predict
from .metrics import confusion_matrix, segmentation_scores

__version__ = '1.0.0'
__all__ = ['BridgeSeg', 'EnhancedPointNet2', 'BridgeStructureLoss', 'read_point_cloud', 'write_las',
           'CLASS_NAMES', 'load_model', 'predict', 'confusion_matrix', 'segmentation_scores']

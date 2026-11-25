"""BridgeSeg network (EnhancedPointNet2) and the MBrSOL structure-oriented loss.

Chen et al. (2026), "Structure-oriented deep learning for semantic segmentation of
bridge point clouds", Automation in Construction 192, 107243.

The code below is the architecture used for the published results; parameter names are
unchanged so that checkpoints trained with the original scripts load directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GeometricFeatureExtraction, BridgeStructureEncoding, \
    ColorFeatureExtraction, CompositeFeatureFusion
from .pointnet2_utils import MultiScaleSetAbstraction, EnhancedFeaturePropagation, square_distance, index_points

__all__ = ['EnhancedPointNet2', 'BridgeSeg', 'BridgeStructureLoss']


class EnhancedPointNet2(nn.Module):
    def __init__(self, num_classes=5, radii_scale=1.0, k_neighbors=32, fusion='knn'):
        super(EnhancedPointNet2, self).__init__()
        # fusion: how the decoder outputs of levels 2 and 1 are brought back to all N points
        # before the multi-scale fusion head.
        #   'knn'   (default) inverse-distance interpolation from the 3 nearest level points.
        #   'index' the original implementation: F.interpolate along the point-index axis.
        #           Level points come out of farthest-point sampling in an order unrelated to
        #           the input order, so this pairs each point with an arbitrary point of the
        #           coarser level. Kept only so that checkpoints trained with it still load
        #           and behave identically; both variants have exactly the same parameters.
        assert fusion in ('knn', 'index')
        self.fusion_mode = fusion
        # [round2] radii_scale and k_neighbors were added for the Reviewer-2 hyperparameter
        # sensitivity study. Defaults (radii_scale=1.0, k_neighbors=32) reproduce the original
        # paper behaviour exactly, so ref_bridgeseg_v0's own scripts are unaffected.
        input_ch = 3
        #self.pos_encoding = EnhancedPositionalEncoding(input_ch, 4, 64,)
        self.bri_enc = BridgeStructureEncoding(input_ch, k_neighbors, 4)

        # color feature branch
        self.color_encoder = ColorFeatureExtraction(3, 6)
        self.feature_fusion = CompositeFeatureFusion(input_ch, 6)

        in_chanel = input_ch + 3 # 3(xyz) + 3(RGB)

        # Encoder (query-ball radii scaled by radii_scale for the sensitivity sweep)
        _r = radii_scale
        # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1*_r, 0.2*_r], [16, 32], in_chanel, [64, 64, 128])
        # 2nd layer: input = 128*2 (Multi-scale connection)
        self.sa2 = MultiScaleSetAbstraction(512, [0.2*_r, 0.4*_r], [16, 32], 259, [128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(128, [0.4*_r, 0.8*_r], [16, 32], 515, [256, 256, 512])

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # Decoder
        self.fp3 = EnhancedFeaturePropagation(1536, [1024, 256])  # multi: 512*2 + 256*2 ,1536
        self.fp2 = EnhancedFeaturePropagation(512, [256, 256])  # multi:  256*2 ,512
        self.fp1 = EnhancedFeaturePropagation(256 + input_ch, [256, 128]) # multi:  128*2 ,256

        self.fusion = MultiScaleFeatureFusion(
            in_channels_list=[256, 256, 128],  # output channels of fp3, fp2, fp1
            out_channels=128
        )
        # multi-scale aggregation + classification head
        self.final_fusion = nn.Sequential(
            nn.Conv1d(384, 128, 1), #128+256+256
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
        self.num_classes = num_classes
        if not hasattr(self, 'cls_head') or self.cls_head is None:
            self.cls_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, xyz, features=None):
        """
        xyz: [B, N, 3]
        points: [B, N, 32] (RGB)
        """

        # Add positional encoding
        pos_enc = self.bri_enc(xyz) # [B, 64, N]
        # Change the order of dimensions
        features = features.transpose(1, 2)  # [B, 3, N]
        color_features = self.color_encoder(features, xyz)  # [B, 12, N]
        fused_features = self.feature_fusion(pos_enc, color_features)  # [B, input_ch, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_features = self.sa1(xyz, fused_features)   #[B,70,N] -> [B, 128, N]
        #l1_features = self.geometric1(l1_features, l1_xyz)

        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l2_features = self.geometric2(l2_features, l2_xyz)

        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l3_features = self.geometric3(l3_features, l3_xyz)

        # Decoder
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(xyz, l1_xyz, fused_features, l1_features)

        if self.fusion_mode == 'knn':
            l2_features = knn_interpolate(xyz, l2_xyz, l2_features)
            l1_features = knn_interpolate(xyz, l1_xyz, l1_features)
        fused_features = self.fusion([l2_features, l1_features, l0_features])

        # per-point classification
        x = self.final_fusion(fused_features)

        return x

def knn_interpolate(xyz_dst, xyz_src, feat_src, k=3):
    """Inverse-distance interpolation of feat_src [B,C,S] at xyz_src [B,S,3] onto xyz_dst [B,N,3]."""
    d, idx = square_distance(xyz_dst, xyz_src).topk(k, dim=-1, largest=False)   # [B,N,k]
    w = 1.0 / (d.clamp(min=0) + 1e-8)
    w = w / w.sum(-1, keepdim=True)
    g = index_points(feat_src.transpose(1, 2), idx)                             # [B,N,k,C]
    return (g * w.unsqueeze(-1)).sum(2).transpose(1, 2)                          # [B,C,N]


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ))

    def forward(self, features_list):
        out_features = []
        l0_features = features_list[2]
        for feat, conv in zip(features_list, self.convs):
            feat = F.interpolate(feat, size=l0_features.shape[2])
            out_features.append(conv(feat))

        return torch.cat(out_features, dim=1)

class BridgeStructureLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=20.0, rel_margin=0.2, class_weights=None,
                 weight_clamp=None, height_ref='class'):
        super().__init__()
        # height_ref: frame in which the mean heights of the predicted components are compared.
        #   'class'  original implementation: each class is normalised by its own (masked)
        #            bounding box, so the heights of different classes are not on a common scale.
        #   'sample' all classes are normalised by the z-range of the whole sample, so
        #            "girder below deck" is checked on one common vertical scale.
        assert height_ref in ('class', 'sample')
        self.height_ref = height_ref
        self.alpha = alpha
        self.rel_margin = rel_margin
        # [round2] weight_clamp: when set (e.g. 5.0), the background-only penalty term is
        # skipped (background is excluded from hierarchy constraints per the paper's
        # Algorithm 3) and the final per-class CE weights are clamped to [0.3, weight_clamp].
        # This bounds the dynamic-weight magnitude and keeps training stable under the
        # weighted-stratified block sampling. weight_clamp=None reproduces the original loss.
        self.weight_clamp = weight_clamp

        # vertical hierarchy: pier/abutment < girder < deck < superstructure
        self.hierarchy = {
            1: {'name': 'abutment', 'below': [2, 3,4], 'require': []},
            2: {'name': 'girder', 'above': [1], 'below': [3,4], 'require': []},
            3: {'name': 'deck', 'above': [1,2], 'below': [4], 'require': []},
            4: {'name': 'parapet', 'above': [1,2,3], 'require': []},
            0: {'name': 'other'}
        }

        # default class weights
        default_weights = torch.tensor([1.5, 1.0, 1.2, 1.5, 1.0])
        self.base_weights = default_weights if class_weights is None else class_weights
        self.register_buffer('base_weights_buffer', self.base_weights)

    def _get_relative_position(self, points, mask):
        masked_points = points * mask.unsqueeze(-1)
        min_vals = masked_points.amin(dim=1, keepdim=True)
        max_vals = masked_points.amax(dim=1, keepdim=True)
        range_vals = max_vals - min_vals + 1e-7
        rel_pos = (masked_points - min_vals) / range_vals
        z_mean = (rel_pos[..., 2] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return z_mean

    def forward(self, outputs, labels, points):
        outputs = outputs.transpose(1, 2)
        B, N = labels.shape
        device = outputs.device
        preds = torch.argmax(outputs, dim=-1)

        weights = self.base_weights_buffer.repeat(B, 1).to(device)

        # ---- which classes are present in each block ----

        exist_mask = {
            cid: (labels == cid).float().sum(dim=1) > 0
            for cid in [1, 2, 3, 4]
        }

        rel_pos = {}
        if self.height_ref == 'sample':
            z = points[..., 2]
            zmin, zmax = z.amin(1, keepdim=True), z.amax(1, keepdim=True)
            zr = (z - zmin) / (zmax - zmin + 1e-7)                              # [B, N] in [0, 1]
        for cid in [1, 2, 3, 4]:
            mask = preds == cid
            if not mask.any():
                rel_pos[cid] = torch.zeros(B, device=device)
            elif self.height_ref == 'sample':
                m = mask.float()
                rel_pos[cid] = (zr * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                rel_pos[cid] = self._get_relative_position(points, mask)

        # ---- hierarchy violations raise the weight of the offending classes ----

        for cid in [1, 2, 3, 4]:
            info = self.hierarchy.get(cid, {})


            required = all([exist_mask[rid].any().item() for rid in info.get('require', [])])
            if not required:
                continue

            if 'above' in info:
                for lower_cid in info['above']:
                    if not exist_mask.get(lower_cid, torch.tensor(False)).any():
                        continue

                    pos_diff = rel_pos[cid] - rel_pos[lower_cid]
                    violation = F.relu(-pos_diff + self.rel_margin)
                    weights[:, cid] += self.alpha * violation
                    weights[:, lower_cid] += self.alpha * violation * 0.5

            if 'below' in info:
                for upper_cid in info['below']:
                    if not exist_mask.get(upper_cid, torch.tensor(False)).any():
                        continue

                    pos_diff = rel_pos[upper_cid] - rel_pos[cid]
                    violation = F.relu(-pos_diff + self.rel_margin)
                    weights[:, cid] += self.alpha * violation
                    weights[:, upper_cid] += self.alpha * violation * 0.3

        # ---- dynamic class re-weighting ----

        if self.weight_clamp is None:
            # original behaviour: background-only dynamic penalty
            other_pred = (preds == 0).float().mean(dim=1)
            weights[:, 0] += self.alpha * (1 - other_pred)

        class_dist = torch.bincount(labels.view(-1), minlength=5).float().clamp(min=1)
        class_weights = (1 / class_dist.sqrt()).to(device)
        class_weights[1] *= 2.0  # up-weight pier/abutment
        class_weights[4] *= 2.0  # up-weight superstructure

        final_w = weights.mean(dim=0) * class_weights
        if self.weight_clamp is not None:
            final_w = final_w.clamp(min=0.3, max=self.weight_clamp)

        return F.cross_entropy(
            outputs.reshape(-1, 5),
            labels.reshape(-1),
            weight=final_w,
            label_smoothing=0.2
        )


# Friendly alias: ``BridgeSeg(num_classes=5)`` is the network described in the paper.
BridgeSeg = EnhancedPointNet2

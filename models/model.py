# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .attention_modules import GeometricFeatureExtraction, EnhancedPositionalEncoding, BridgeStructureEncoding, \
    ColorFeatureExtraction, CompositeFeatureFusion
from .pointnet2_utils import FeaturePropagation, SetAbstraction, MultiScaleSetAbstraction, EnhancedFeaturePropagation

class PointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Encoder
        self.sa1 = SetAbstraction(1024, 0.1, 32, 6, [64, 64, 128]) #
        self.sa2 = SetAbstraction(256, 0.2, 32, 131, [128, 128, 256]) #128+3=131
        self.sa3 = SetAbstraction(64, 0.4, 32, 259, [256, 256, 512]) #

        # Decoder
        self.fp3 = FeaturePropagation(768, [256, 256]) # 512+256=768
        self.fp2 = FeaturePropagation(384, [256, 128]) # 256+128=384
        self.fp1 = FeaturePropagation(128, [128, 128, 128]) # 128+3=131

        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, 3] (RGB)
        """

        # Change the order of dimensions
        points = points.transpose(1, 2)  # [B, 3, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)

        return out

class EnhancedPointNet2(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedPointNet2, self).__init__()
        input_ch = 3
        #self.pos_encoding = EnhancedPositionalEncoding(input_ch, 4, 64,)
        self.bri_enc = BridgeStructureEncoding(input_ch, 32, 4)

        # Color feature processing module
        self.color_encoder = ColorFeatureExtraction(3, 6)
        self.feature_fusion = CompositeFeatureFusion(input_ch, 6)

        in_chanel = input_ch + 3 # 3(xyz) + 3(RGB)

        # Encoder
        # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2], [16, 32], in_chanel, [64, 64, 128])
        # 2nd layer: input = 128*2 (Multi-scale connection)
        self.sa2 = MultiScaleSetAbstraction(512, [0.2, 0.4], [16, 32], 259, [128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(128, [0.4, 0.8], [16, 32], 515, [256, 256, 512])

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # Decoder
        self.fp3 = EnhancedFeaturePropagation(1536, [1024, 256])  # multi: 512*2 + 256*2 ,1536
        self.fp2 = EnhancedFeaturePropagation(512, [256, 256])  # multi:  256*2 ,512
        self.fp1 = EnhancedFeaturePropagation(256 + input_ch, [256, 128]) # multi:  128*2 ,256

        self.fusion = MultiScaleFeatureFusion(
            in_channels_list=[256, 256, 128],  # Output channels of fp3, fp2, fp1
            out_channels=128
        )
        # Add multi-scale feature aggregation Final layers
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

        fused_features = self.fusion([l2_features, l1_features, l0_features])

        # Final classification
        x = self.final_fusion(fused_features)

        return x

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
    def __init__(self, num_classes=5, alpha=20.0, rel_margin=0.2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.rel_margin = rel_margin

        # Define hierarchical relationships between classes
        self.hierarchy = {
            1: {'name': 'abutment', 'below': [2, 3, 4], 'require': []},
            2: {'name': 'girder', 'above': [1], 'below': [3, 4], 'require': []},
            3: {'name': 'deck', 'above': [1, 2], 'below': [4], 'require': []},
            4: {'name': 'parapet', 'above': [1, 2, 3], 'require': []},  # Relax constraints for class 4
            0: {'name': 'other'}
        }

        # Default class weights
        default_weights = torch.tensor([1.5, 1.0, 1.2, 1.5, 1.0])
        self.base_weights = default_weights if class_weights is None else class_weights
        self.register_buffer('base_weights_buffer', self.base_weights)

    def _get_relative_position(self, points, mask):
        """
        Calculate the relative position of points within a mask.
        Args:
            points: [B, N, 3] Point cloud coordinates
            mask: [B, N] Binary mask for the points
        Returns:
            z_mean: [B] Mean relative z-coordinate for the masked points
        """
        masked_points = points * mask.unsqueeze(-1)
        min_vals = masked_points.amin(dim=1, keepdim=True)
        max_vals = masked_points.amax(dim=1, keepdim=True)
        range_vals = max_vals - min_vals + 1e-7
        rel_pos = (masked_points - min_vals) / range_vals
        z_mean = (rel_pos[..., 2] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return z_mean

    def forward(self, outputs, labels, points):
        """
        Compute the loss with hierarchical constraints and dynamic class weights.
        Args:
            outputs: [B, num_classes, N] Predicted logits
            labels: [B, N] Ground truth labels
            points: [B, N, 3] Point cloud coordinates
        Returns:
            loss: Computed loss value
        """
        outputs = outputs.transpose(1, 2)  # [B, N, num_classes]
        B, N = labels.shape
        device = outputs.device
        preds = torch.argmax(outputs, dim=-1)  # Predicted class indices

        weights = self.base_weights_buffer.repeat(B, 1).to(device)

        # ============ Existence Check ============

        exist_mask = {
            cid: (labels == cid).float().sum(dim=1) > 0
            for cid in [1, 2, 3, 4]
        }

        # Calculate relative positions for each class
        rel_pos = {}
        for cid in [1, 2, 3, 4]:
            mask = preds == cid
            rel_pos[cid] = self._get_relative_position(points, mask) if mask.any() else torch.zeros(B, device=device)

        # ============ Repair Hierarchical Constraint Logic ============

        for cid in [1, 2, 3, 4]:
            info = self.hierarchy.get(cid, {})

            # Use any() for existence check
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

        # ============ Dynamic Adjustment of Class Weights ============

        other_pred = (preds == 0).float().mean(dim=1)
        weights[:, 0] += self.alpha * (1 - other_pred)

        class_dist = torch.bincount(labels.view(-1), minlength=5).float().clamp(min=1)
        class_weights = (1 / class_dist.sqrt()).to(device)
        class_weights[1] *= 2.0  # Increase weight for class 1
        class_weights[4] *= 2.0  # Increase weight for class 4

        return F.cross_entropy(
            outputs.reshape(-1, 5),
            labels.reshape(-1),
            weight=weights.mean(dim=0) * class_weights,
            label_smoothing=0.2
        )

# PointNet basic module
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# PointNet semantic segmentation model
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=5, feature_transform=True):
        super(PointNetSeg, self).__init__()
        self.stn = STN3d()
        self.feature_transform = feature_transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)  # Add back this layer
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)  # Add back this layer
        
        self.fc1 = nn.Linear(2048, 512)  # Fix input dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Fix input dimension
        self.dropout = nn.Dropout(p=0.3)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.mlp_64 = nn.Sequential(
            nn.Conv1d(64, 64, 1),  # Modify input channels to in_channels + 3
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1)
            )

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        
        # If additional features are provided, concatenate with coordinates
        if features is not None:
            point_cloud = torch.cat([xyz, features], dim=2)
        else:
            point_cloud = xyz
            
        point_cloud = point_cloud.transpose(2, 1)
        
        # Apply spatial transformation
        trans = self.stn(point_cloud[:, :3, :])
        point_cloud_transformed = torch.bmm(point_cloud[:, :3, :].transpose(2, 1), trans).transpose(2, 1)
        
        # MLP processing
        x = F.relu(self.bn1(self.conv1(point_cloud_transformed)))
        x = self.mlp_64(x)  # Add MLP processing
        x = self.mlp_64(x)  # Add MLP processing
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # Add back this layer
        
        # Global features
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, 2048)  # Fix dimension
        
        # Classification output
        x = F.relu(self.bn6(self.fc1(global_feature)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Expand output to [B, N, C] format to maintain consistency with other models
        x = x.unsqueeze(1).repeat(1, N, 1)
        
        return x

# DGCNN model implementation
class DGCNN(nn.Module):
    def __init__(self, num_classes=5, k=64):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(320, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size, num_dims, num_points = x.size()
        
        if idx is None:
            idx = self.knn(x, k)
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        
        return feature

    def forward(self, xyz, features=None):
        B, N, C = xyz.shape
        
        # If features are provided, concatenate with coordinates
        if features is not None:
            xyz = torch.cat([xyz, features], dim=2)
            
        x = xyz.transpose(2, 1)
        
        batch_size = x.size(0)
        x = x[:, :3, :]  # Only use coordinate information
        
        # EdgeConv module
        x1 = self.get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = self.get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = self.get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = self.get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        
        # Global features
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        # MLP classifier
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        # Expand output to [B, N, C] format
        x = x.unsqueeze(1).repeat(1, N, 1)
        
        return x

# Create a simple dataset class
class RandomPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024):
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xyz = torch.randn(self.num_points, 3)
        colors = torch.rand(self.num_points, 3)
        xyz = xyz.float()
        colors = colors.float()
        return {
            'xyz': xyz,
            'colors': colors
        }


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    num_points = 1024
    dataset = RandomPointCloudDataset(num_samples=100, num_points=num_points)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    pretrain_model = PointCloudPretraining()
    pretrain_model = pretrain_model.to(device)
    pretrain_model = pretrain(pretrain_model, train_loader, epochs=2, device=device)
    main_model = EnhancedPointNet2()
    pretrain_model = pretrain_model.to(device)
    pretrain_model.eval()
    main_model = main_model.to(device)
    main_model.eval()
    print("=" * 50)
    print("Basic functionality test")
    print(f"Input xyz shape: {xyz.shape}")
    print(f"Input features shape: {features.shape}")
    try:
        xyz = xyz.to(device)
        features = features.to(device)
        output = main_model(xyz, features)
        reconstructed_xyz, predicted_quaternion= pretrain_model(xyz, features)
        print(f"main model output shape: {output.shape}")
        print(f"pretrain model output shape: {reconstructed_xyz.shape}")
        print("Model forward pass test passed!")
    except Exception as e:
        print(f"Model runtime error: {str(e)}")
    print("\n" + "=" * 50)
    print("Model statistics")
    total_params = sum(p.numel() for p in main_model.parameters())
    trainable_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\n" + "=" * 50)
    print("Memory usage test")
    batch_sizes = [4, 16]
    for bs in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            xyz_test = torch.randn(bs, num_points, 3).to(device)
            features_test = torch.randn(bs, num_points, 3).to(device)
            torch.cuda.reset_peak_memory_stats()
            output = main_model(xyz_test, features_test)
            memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Batch size {bs}: Peak memory usage {memory:.2f} MB")
    print("\nTest completed!")

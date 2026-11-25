# models/pointnet2_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Squared Euclidean distance between two point sets."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, nsample]
    Return:
        new_points:, indexed points data, [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    max_idx = points.shape[1] - 1
    idx = torch.clamp(idx, 0, max_idx)
    idx = idx.clamp(0, points.shape[1] - 1)
    new_points = points[batch_indices, idx, :]

    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """Sample and group."""
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


# def farthest_point_sample(xyz, npoint):
#     """
#     """
#     avs_net = AVSNet().to(xyz.device)
#     indices = avs_net(xyz, npoint)
#
#     return indices

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, C, N]
        """
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()
            points = points.transpose(1, 2)

        # Sample and group
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points shape: [B, npoint, nsample, C]
        B, npoint, nsample, C = new_points.shape
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # [B, C, npoint, nsample]

        # Point feature embedding
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling
        new_points = torch.max(new_points, -1)[0]  # [B, C, npoint]

        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: input points position data, [B, N, C]
        xyz2: sampled input points position data, [B, S, C]
        points1: input points data, [B, D, N]
        points2: input points data, [B, D, S]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2.transpose(1, 2), idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.transpose(1, 2)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class EnhancedFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        self.attention = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 4, 1),
            nn.BatchNorm1d(in_channel // 4),
            nn.ReLU(),
            nn.Conv1d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )

        self.skip_connection = (in_channel == mlp[-1])

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.boundary_aware = nn.Sequential(
            nn.Conv1d(3, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, mlp[-1], 1)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :4], idx[:, :, :4]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2.transpose(1, 2), idx) *
                weight.view(B, N, 4, 1),
                dim=2
            )

        if points1 is not None:
            points1 = points1.transpose(1, 2)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2)

        attention_weights = self.attention(new_points)
        new_points = new_points * attention_weights

        edge_features = self.boundary_aware(xyz1.transpose(1, 2))

        identity = new_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        if self.skip_connection:
            new_points = new_points + identity

        new_points = new_points + edge_features

        return new_points


class MultiScaleSetAbstraction(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(radius_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel

            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, C, N) features
        """
        B, N, _ = xyz.shape
        S = self.npoint

        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        multi_scale_features = []

        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

            if points is not None:
                grouped_points = index_points(points.transpose(1, 2), idx)  # [B, npoint, nsample, C]
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm

            grouped_points = grouped_points.permute(0, 3, 1, 2)  # [B, C+3, npoint, nsample]

            for j, (conv, bn) in enumerate(zip(self.conv_blocks[i], self.bn_blocks[i])):
                grouped_points = F.relu(bn(conv(grouped_points)))

            grouped_points = torch.max(grouped_points, -1)[0]  # [B, C, npoint]
            multi_scale_features.append(grouped_points)

        new_points = torch.cat(multi_scale_features, dim=1)
        return new_xyz, new_points


class AVSNet(nn.Module):
    def __init__(self, V0=0.02, Kp=0.5, Ki=0.1, max_iter=10):
        super(AVSNet, self).__init__()
        self.V0 = V0
        self.Kp = Kp
        self.Ki = Ki
        self.max_iter = max_iter
        self.register_buffer('V0_tensor', torch.tensor(V0, dtype=torch.float))

    def voxel_downsample(self, xyz, voxel_size):
        """Voxel down-sampling."""
        B, N, C = xyz.shape
        device = xyz.device

        voxel_coords = torch.floor(xyz / voxel_size).long()

        results = []
        for b in range(B):
            coords = voxel_coords[b]
            unique_keys = coords[:, 0] * 1000000 + coords[:, 1] * 1000 + coords[:, 2]

            unique_keys, inverse_indices = torch.unique(unique_keys, return_inverse=True)

            voxel_points = xyz[b]
            centroids = torch.zeros((len(unique_keys), C), device=device)
            count = torch.zeros(len(unique_keys), device=device)

            centroids.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, C), voxel_points)
            count.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))

            centroids = centroids / count.unsqueeze(1).clamp(min=1)
            results.append(centroids)

        max_length = max([len(r) for r in results])

        padded_results = []
        for r in results:
            if len(r) < max_length:
                padding = torch.zeros((max_length - len(r), C), device=device)
                r = torch.cat([r, padding], dim=0)
            padded_results.append(r)

        return torch.stack(padded_results)

    def adapt_voxel_size(self, xyz, npoint):
        """Adaptively adjust the voxel size."""
        B, N, _ = xyz.shape
        device = xyz.device
        target_ratio = N / npoint

        scale = torch.zeros(1, device=device)
        sum_err = torch.zeros(1, device=device)

        for _ in range(self.max_iter):
            voxel_size = self.V0_tensor * torch.exp(scale)

            sampled_points = self.voxel_downsample(xyz, voxel_size)
            current_ratio = N / sampled_points.shape[1]

            err = target_ratio - current_ratio
            sum_err += err

            diff = self.Kp * err + self.Ki * sum_err
            scale += 0.01 * (torch.sigmoid(diff) - 0.5)

            if abs(sampled_points.shape[1] - npoint) <= npoint * 0.1:
                break

        return voxel_size

    def forward(self, xyz, npoint):
        """
        Args:
            xyz: input points (B, N, 3)
            npoint: number of points to sample
        Returns:
            indices: indices of sampled points (B, npoint)
        """
        B, N, C = xyz.shape
        device = xyz.device

        voxel_size = self.adapt_voxel_size(xyz, npoint)

        sampled_points = self.voxel_downsample(xyz, voxel_size)

        indices = []
        for b in range(B):
            dist = torch.sum((xyz[b].unsqueeze(1) - sampled_points[b].unsqueeze(0)) ** 2, dim=2)
            batch_indices = torch.argmin(dist, dim=0)

            if len(batch_indices) > npoint:
                perm = torch.randperm(len(batch_indices), device=device)[:npoint]
                batch_indices = batch_indices[perm]
            elif len(batch_indices) < npoint:
                padding = batch_indices[-1].repeat(npoint - len(batch_indices))
                batch_indices = torch.cat([batch_indices, padding])

            indices.append(batch_indices)

        indices = torch.stack(indices)
        return indices

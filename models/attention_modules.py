# models/attention_modules.py
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, channels=64, freq_bands=16):
        super().__init__()
        self.channels = channels
        self.freq_bands = freq_bands

        # Generate sine waves of different frequencies
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)  # [freq_bands]

        # Initialize the linear layer in __init__
        self.linear_proj = nn.Linear(6 * freq_bands, channels)

    def forward(self, xyz):
        """
        xyz: (B, N, 3) Input point cloud coordinates
        return: (B, N, channels) Position encoding
        """
        # 1. Expand coordinates
        B, N, _ = xyz.shape

        # 2. Calculate sine and cosine encodings at different frequencies
        pos_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                pos_enc.append(func(xyz * freq))

        # 3. Concatenate all encodings
        pos_enc = torch.cat(pos_enc, dim=-1)  # [B, N, 6*freq_bands]

        # 4. Project through MLP to specified dimension
        pos_enc = self.linear_proj(pos_enc)  # [B, N, channels]

        # 5. Transpose to match the expected shape
        pos_enc = pos_enc.transpose(1, 2)  # [B, channels, N]

        return pos_enc

    def to(self, device):
        """
        Override to method to ensure all components move to the same device
        """
        super().to(device)
        self.linear_proj = self.linear_proj.to(device)
        return self


class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Boundary feature extraction network
        self.boundary_net = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        # Spatial relation encoding
        self.spatial_relation = nn.Sequential(
            nn.Conv1d(4, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels + 64, in_channels // 2, 1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, xyz):
        """
        x: [B, C, N] Features
        xyz: [B, N, 3] Coordinates
        """
        B, C, N = x.shape

        # Calculate the relative position and distance between points
        xyz_transpose = xyz.transpose(1, 2)  # [B, 3, N]

        # Calculate local features
        inner_product = torch.matmul(xyz, xyz.transpose(1, 2))
        xx = torch.sum(xyz ** 2, dim=2, keepdim=True)
        distance = xx + xx.transpose(1, 2) - 2 * inner_product  # [B, N, N]

        # Select K nearest neighbors
        K = 16
        knn_idx = torch.topk(distance, k=K, dim=-1, largest=False)[1]  # [B, N, K]

        # Get relative coordinates and distance features
        batch_idx = torch.arange(B, device=xyz.device).view(-1, 1, 1)
        knn_xyz = xyz[batch_idx, knn_idx, :]  # [B, N, K, 3]
        relative_xyz = knn_xyz - xyz.unsqueeze(2)  # [B, N, K, 3]
        relative_distance = torch.norm(relative_xyz, dim=-1, keepdim=True)  # [B, N, K, 1]

        # Calculate the average relative position and distance, and adjust the dimensions
        mean_relative_xyz = torch.mean(relative_xyz, dim=2)  # [B, N, 3]
        mean_relative_xyz = mean_relative_xyz.permute(0, 2, 1)  # [B, 3, N]

        mean_distance = torch.mean(relative_distance, dim=2)  # [B, N, 1]
        mean_distance = mean_distance.permute(0, 2, 1)  # [B, 1, N]

        # Prepare spatial relation features
        spatial_features = torch.cat([
            mean_relative_xyz,  # [B, 3, N]
            mean_distance  # [B, 1, N]
        ], dim=1)  # [B, 4, N]

        # Extract spatial relation features
        spatial_feat = self.spatial_relation(spatial_features)  # [B, 64, N]

        # Get local feature differences
        knn_features = x.unsqueeze(2).expand(-1, -1, N, -1)  # [B, C, N, N]
        knn_features = torch.gather(knn_features, 3,
                                    knn_idx.unsqueeze(1).expand(-1, C, -1, -1))  # [B, C, N, K]

        # Calculate local feature differences
        local_diff = knn_features - x.unsqueeze(3)  # [B, C, N, K]

        # Merge local features
        boundary_feat = torch.cat([
            x,
            torch.max(local_diff, dim=3)[0]  # Max pooling aggregation
        ], dim=1)  # [B, 2C, N]

        # Extract boundary features
        boundary_feat = self.boundary_net(boundary_feat)

        # Calculate attention weights
        attention_input = torch.cat([x, spatial_feat], dim=1)
        attention_weights = self.attention(attention_input)

        # Feature enhancement
        enhanced_feat = x + boundary_feat * attention_weights

        return enhanced_feat


class StructuralAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Structure feature extraction
        self.structure_net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

        # Global context
        self.global_context = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract structure features
        struct_feat = self.structure_net(x)

        # Global context attention
        context = self.global_context(x)

        return x + struct_feat * context



# In attention_modules.py
class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, N]
        """
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca

        # Spatial attention
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa

        return x+x_sa

class GeometricFeatureExtraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + 16, in_channels, 1),  # Modify input channel to in_channels + 3
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

        self.br_pos=BridgeStructureEncoding(channels=16)

    def forward(self, x, xyz):
        """
        Extract geometric features
        x: [B, C, N] Features
        xyz: [B, N, 3] Coordinates
        """
        # Calculate normals
        #normals = compute_normals(xyz)  # [B, N, 3]
        pos_enc=self.br_pos(xyz) #[B,C,N]

        # Merge features
        geometric_features = torch.cat([
            x,
            pos_enc #.transpose(1, 2)  # [B, 3, N]
        ], dim=1)  # [B, C+16, N]

        return self.mlp(geometric_features)



def square_distance(src, dst):
    """
    Calculate pairwise squared distance between two sets of points.

    Args:
        src: Source points, shape (B, N, C)
        dst: Target points, shape (B, M, C)
    Returns:
        dist: Pairwise distance matrix, shape (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Get corresponding points from point cloud based on indices

    Args:
        points: Input point cloud, shape (B, N, C)
        idx: Indices, shape (B, N, k)
    Returns:
        indexed_points: Indexed points, shape (B, N, k, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def compute_normals(xyz):
    """
    Compute point cloud normals
    xyz: [B, N, 3]
    return: [B, N, 3]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Use KNN to find neighboring points
    k = 20
    dist = square_distance(xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]  # [B, N, k]

    # Get neighboring points
    neighbors = index_points(xyz, idx)  # [B, N, k, 3]

    # Compute covariance matrix
    neighbors = neighbors - xyz.unsqueeze(2)  # Centering
    covariance = torch.matmul(neighbors.transpose(2, 3), neighbors)  # [B, N, 3, 3]

    # Eigenvalue decomposition
    eigenvals, eigenvects = torch.linalg.eigh(covariance)

    # Take the eigenvector corresponding to the smallest eigenvalue as the normal vector
    normals = eigenvects[..., 0]  # [B, N, 3]

    return normals


class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, channels=32, freq_bands=4, k_neighbors=16, max_radius=10.0):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.freq_bands = freq_bands
        self.max_radius = max_radius

        # Generate different frequencies (for relative position encoding)
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)

        # MLP for relative position encoding
        self.relative_mlp = nn.Sequential(
            nn.Conv2d(6 * freq_bands + 4, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        # MLP for structure-aware encoding
        self.structure_mlp = nn.Sequential(
            nn.Conv2d(22, channels // 2, 1),  # 22-dim: 9-dim + 3-dim + 4-dim + 3-dim + 3-dim
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        # Residual projection layer
        self.residual_proj = nn.Linear(3, channels)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def get_relative_encoding(self, xyz, neighbors, center):
        """Compute relative position encoding"""
        B, N, k, _ = neighbors.shape

        # Compute relative position and distance
        rel_pos = neighbors - center  # (B, N, k, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, N, k, 1)
        diff_normalized = rel_pos / (dist + 1e-8)  # Normalize relative position

        # Compute frequency encoding
        pos_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                pos_enc.append(func(rel_pos * freq))
        pos_enc = torch.cat(pos_enc, dim=-1)  # (B, N, k, 6*freq_bands)

        # Combine features
        rel_features = torch.cat([pos_enc, dist, diff_normalized], dim=-1)

        # Process through MLP
        rel_features = rel_features.permute(0, 3, 1, 2)  # (B, N, k, channels//2) → (B, channels//2, N, k)
        encoded = self.relative_mlp(rel_features)  # [B, C//2, N, K]
        encoded = encoded.permute(0, 2, 3, 1)  # [B, N, K, channels//2]

        return encoded.mean(dim=2)  # (B, N, channels//2)

    def get_structure_encoding(self, rel_pos):
        """Compute enhanced structure-aware encoding
        rel_pos: (B, N, k, 3) Relative position vectors
        """
        B, N, k, _ = rel_pos.shape

        # 1. Local covariance matrix features
        # Compute covariance matrix for local point cloud
        rel_pos_2d = rel_pos.view(B * N, k, 3)
        cov_matrix = torch.bmm(rel_pos_2d.transpose(1, 2), rel_pos_2d) / (k - 1)  # (B*N, 3, 3)
        cov_matrix = cov_matrix.view(B, N, 9)  # Flatten covariance matrix

        # 2. Main direction features (PCA)
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.view(B * N, 3, 3))
            eigenvalues = eigenvalues.view(B, N, 3)  # Eigenvalues represent the importance of main directions

            # Compute local surface anisotropy features
            anisotropy = (eigenvalues[..., 0] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            planarity = (eigenvalues[..., 1] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            sphericity = eigenvalues[..., 2] / (eigenvalues[..., 0] + 1e-8)

            # Combine PCA features
            pca_features = torch.stack([anisotropy, planarity, sphericity], dim=-1)  # (B, N, 3)
        except:
            # If PCA computation fails, use zero vector
            pca_features = torch.zeros((B, N, 3), device=rel_pos.device)

        # 3. Local geometric features
        # Compute distance from each point to local center
        center = rel_pos.mean(dim=2, keepdim=True)  # (B, N, 1, 3)
        distances = torch.norm(rel_pos - center, dim=-1)  # (B, N, k)

        # Compute local radius and density
        local_radius = distances.max(dim=-1)[0]  # (B, N)
        point_density = k / (local_radius + 1e-8)  # (B, N)

        # Compute local curvature (using distance variation)
        sorted_distances, _ = distances.sort(dim=-1)
        curvature = sorted_distances[..., 1:] - sorted_distances[..., :-1]  # Difference of adjacent distances
        mean_curvature = curvature.mean(dim=-1)  # (B, N)

        # 4. Direction consistency features
        # Compute directional changes between neighboring points
        normalized_rel_pos = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8)
        direction_similarity = torch.bmm(
            normalized_rel_pos.view(B * N, k, 3),
            normalized_rel_pos.view(B * N, k, 3).transpose(1, 2)
        ).view(B, N, k, k)
        direction_consistency = direction_similarity.mean(dim=(-1, -2))  # (B, N)

        # 5. Combine all features
        geometric_features = torch.stack([
            local_radius,
            point_density,
            mean_curvature,
            direction_consistency
        ], dim=-1)  # (B, N, 4)

        # Compute basic statistical features (retain some statistical information)
        mean = rel_pos.mean(dim=2)  # (B, N, 3)
        std = rel_pos.std(dim=2)  # (B, N, 3)

        # Combine all features
        struct_features = torch.cat([
            cov_matrix,  # 9-dim: Covariance matrix features
            pca_features,  # 3-dim: PCA features
            geometric_features,  # 4-dim: Geometric features
            mean,  # 3-dim: Mean position
            std,  # 3-dim: Standard deviation
        ], dim=-1)  # (B, N, 22)

        struct_features = struct_features.permute(0, 2, 1).unsqueeze(2)  # (B, 22, N) -> (B, 22, 1, N)
        struct_features = self.structure_mlp(struct_features)  # (B, 22, 1, N) -> (B, channels//2, 1, N)
        struct_features = struct_features.squeeze(2)  # (B, channels//2, 1, N) -> (B, channels//2, N)
        return struct_features.permute(0, 2, 1)  # Add this line to convert (B, channels//2, N) to (B, N, channels//2)

    def forward(self, xyz):
        """
        xyz: (B, N, 3) Input point cloud coordinates
        return: (B, channels, N) Position encoding
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # 1. Find k-nearest neighbors
        dist = torch.cdist(xyz, xyz)  # (B, N, N)
        k = min(self.k, N)
        _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k)

        # Get neighbor coordinates
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        neighbors = xyz.view(B * N, -1)[idx].view(B, N, k, -1)  # (B, N, k, 3)
        center = xyz.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, 3)

        # 2. Compute relative position encoding
        rel_encoding = self.get_relative_encoding(xyz, neighbors, center)  # (B, N, channels//2)
        #print(f"rel_encoding shape before fusion: {rel_encoding.shape}")

        # 3. Compute structure-aware encoding
        rel_pos = neighbors - center
        struct_encoding = self.get_structure_encoding(rel_pos)  # (B, N, channels//2)
        #print(f"struct_encoding shape before fusion: {struct_encoding.shape}")

        # 4. Combine all features
        combined_encoding = torch.cat([rel_encoding, struct_encoding], dim=-1)  # (B, N, channels)
        #print(f"combined_encoding shape before fusion: {combined_encoding.shape}")
        #encoded = self.fusion_layer(combined_encoding)  # (B, N, channels)
        final_encoding = combined_encoding

        return final_encoding.transpose(1, 2)  # (B, channels, N)


class BridgeStructureEncoding(nn.Module):
    def __init__(self, channels=32, k_neighbors=16, freq_bands=4,
                 min_scale=0.05, max_scale=100.0, grid_size=1.0):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.freq_bands = freq_bands
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.grid_size = grid_size  # Grid size for absolute position encoding

        # Generate frequencies
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)

        # Calculate input feature dimensions
        self.abs_pos_dim = 6 * freq_bands  # Absolute position encoding dimension
        self.rel_pos_dim = 3  # Relative position dimension
        self.local_struct_dim = 13  # Local structure feature dimension
        self.total_dim = self.abs_pos_dim + self.rel_pos_dim + self.local_struct_dim

        # MLP for feature fusion
        self.structure_mlp = nn.Sequential(
            nn.Conv2d(self.total_dim, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def compute_absolute_position_encoding(self, xyz):
        """
        Compute absolute position encoding
        Args:
            xyz: [B, N, 3] Point cloud coordinates
        Returns:
            abs_enc: [B, N, 6*freq_bands] Absolute position encoding
        """
        B, N, _ = xyz.shape

        # 1. Quantize coordinates (ensure locality of position encoding)
        grid_xyz = torch.floor(xyz / self.grid_size) * self.grid_size

        # 2. Compute absolute position encoding for each point
        abs_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                # Encode on quantized coordinates
                enc = func(grid_xyz * freq)
                abs_enc.append(enc)

        abs_enc = torch.cat(abs_enc, dim=-1)  # [B, N, 6*freq_bands]
        return abs_enc

    def forward(self, xyz):
        B, N, _ = xyz.shape
        device = xyz.device

        # 1. Compute absolute position encoding (same for the same point)
        abs_pos_enc = self.compute_absolute_position_encoding(xyz)  # [B, N, 6*freq_bands]

        # 2. K-nearest neighbor search
        dist = torch.cdist(xyz, xyz)
        k = min(self.k, N)
        _, idx = dist.topk(k, dim=-1, largest=False)

        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        # 3. Get neighbor points
        neighbors = xyz.view(B * N, -1)[idx].view(B, N, k, -1)
        center = xyz.unsqueeze(2).expand(-1, -1, k, -1)

        # 4. Compute relative position
        rel_pos = neighbors - center  # [B, N, k, 3]

        # 5. Compute local structure features
        structure_features = self.get_structure_features(rel_pos)  # [B, N, 13]

        # 6. Combine features
        # Expand absolute position encoding and structure features to k neighbors
        abs_pos_enc = abs_pos_enc.unsqueeze(2).expand(-1, -1, k, -1)
        structure_features = structure_features.unsqueeze(2).expand(-1, -1, k, -1)

        combined_features = torch.cat([
            abs_pos_enc,  # [B, N, k, 6*freq_bands] Absolute position encoding
            rel_pos,  # [B, N, k, 3] Relative position
            structure_features  # [B, N, k, 13] Local structure features
        ], dim=-1) # [B, N, k, 6*freq_bands + 3 + 13]

        # 7. Process through MLP
        combined_features = combined_features.permute(0, 3, 1, 2) # [B, 6*freq_bands + 3 + 13, N, k]
        encoded = self.structure_mlp(combined_features)
        encoded = torch.max(encoded, dim=-1)[0]  # [B, channels, N]

        return encoded

    # get_structure_features method remains unchanged

    def get_structure_features(self, rel_pos):
        """
        Args:
            rel_pos: [2, 4096, 16, 3]
        Returns:
            features: [2, 4096, 13]
        """
        B, N, k, _ = rel_pos.shape

        # 1. Local main direction features
        rel_pos_2d = rel_pos.view(B * N, k, 3)
        cov_matrix = torch.bmm(rel_pos_2d.transpose(1, 2), rel_pos_2d) / (k - 1)

        try:
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            eigenvalues = eigenvalues.view(B, N, 3)

            linearity = (eigenvalues[..., 0] - eigenvalues[..., 1]) / (eigenvalues[..., 0] + 1e-8)
            planarity = (eigenvalues[..., 1] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            sphericity = eigenvalues[..., 2] / (eigenvalues[..., 0] + 1e-8)

            structure_features = torch.stack([linearity, planarity, sphericity], dim=-1)
        except:
            structure_features = torch.zeros((B, N, 3), device=rel_pos.device)

        # 2. Local statistical features
        center = rel_pos.mean(dim=2, keepdim=True)
        distances = torch.norm(rel_pos - center, dim=-1)

        local_stats = torch.stack([
            distances.max(dim=-1)[0],  # local_radius
            distances.mean(dim=-1),  # mean_dist
            distances.std(dim=-1)  # std_dist
        ], dim=-1)  # [B, N, 3]

        # 3. Direction features
        normalized_pos = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8)
        direction_sim = torch.bmm(
            normalized_pos.view(B * N, k, 3),
            normalized_pos.view(B * N, k, 3).transpose(1, 2)
        ).view(B, N, k, k)
        direction_consistency = direction_sim.mean(dim=(-1, -2))

        # 4. Z-axis features
        z_stats = torch.stack([
            rel_pos[..., 2].std(dim=-1),  # z_variation
            rel_pos[..., 2].max(dim=-1)[0] - rel_pos[..., 2].min(dim=-1)[0]  # z_range
        ], dim=-1)  # [B, N, 2]

        # 5. Mean relative position
        mean_rel_pos = rel_pos.mean(dim=2)  # [B, N, 3]

        # Combine all features (ensure total 13 dimensions)
        features = torch.cat([
            structure_features,  # 3 dimensions
            local_stats,  # 3 dimensions
            direction_consistency.unsqueeze(-1),  # 1 dimension
            z_stats,  # 2 dimensions
            mean_rel_pos,  # 3 dimensions
            torch.norm(rel_pos.std(dim=2), dim=-1, keepdim=True)  # 1 dimension
        ], dim=-1)  # Total 13 dimensions

        # Dimension verification
        assert features.shape[-1] == 13, f"Features should have 13 dimensions, but got {features.shape[-1]}"

        return features


class ColorFeatureExtraction(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Local color feature extraction
        self.color_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Color attention mechanism
        self.color_attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

        # Color context encoding
        self.color_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, colors, xyz):
        """
        Args:
            colors: [B, 3, N] Input color features
            xyz: [B, N, 3] Input point cloud coordinates
        Returns:
            Enhanced global color features [B, 32, N]
        """
        # Step 1: Extract basic color features
        color_features = self.color_mlp(colors)  # [B, 32, N]

        # Step 2: Compute local color similarity
        B, _, N = color_features.shape

        # Find spatial neighbors
        dist = torch.cdist(xyz, xyz)
        k = 16  # Number of nearest neighbors
        _, idx = dist.topk(k=k, dim=-1, largest=False)

        # Gather neighbor color features
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        neighbors_features = color_features.view(B * N, -1)[idx.view(-1)].view(B, N, k, -1)

        # Step 3: Compute color attention
        color_weights = self.color_attention(color_features)
        enhanced_local = color_features * color_weights

        # Step 4: Compute global color context
        context_weights = self.color_context(color_features)
        enhanced_global = enhanced_local * context_weights

        return enhanced_global  # [B, 32, N]


class CompositeFeatureFusion(nn.Module):
    def __init__(self, spatial_channels, color_channels):
        super().__init__()
        total_channels = spatial_channels + color_channels
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(total_channels, spatial_channels, 1),
            nn.BatchNorm1d(spatial_channels),
            nn.ReLU()
        )

    def forward(self, spatial_features, color_features):
        # Print dimensions for debugging
        #print(f"Spatial features shape: {spatial_features.shape}")
        #print(f"Color features shape: {color_features.shape}")
        fused_features = torch.cat([spatial_features, color_features], dim=1)
        #print(f"Fused features shape: {fused_features.shape}")
        return self.fusion_mlp(fused_features)


if __name__ == '__main__':
    # Test PositionalEncoding
    xyz = torch.randn(2, 4096, 3) #B, N, C
    # Instantiate EnhancedPositionalEncoding class
    #pos_enc = EnhancedPositionalEncoding(channels=32, freq_bands=16, k_neighbors=3)

    pos_enc_2 = BridgeStructureEncoding(channels=32, k_neighbors=16, freq_bands=4)

    # Call forward method
    #output = pos_enc(xyz)

    output_2 = pos_enc_2(xyz)

    # Print output shape
    #print(f"Output shape: {output.shape}")
    print(f"Output shape: {output_2.shape}")
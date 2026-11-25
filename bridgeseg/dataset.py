import hashlib
import logging
import multiprocessing
import os

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BriPCDMulti(Dataset):
    def __init__(self, data_dir, file_list=None, num_points=4096, transform=False, block_size=1.0,
                 sample_rate=0.5, logger=None):
        """
        Args:
            data_dir (str): directory with .h5 files
            num_points (int): number of points per block
            transform (bool): apply data augmentation
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.logger = logger if logger else logging.getLogger(__name__)

        self.cache_dir = os.path.join(self.data_dir, 'cache_global')
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_id = self._get_cache_id(data_dir, file_list)

        self.block_info = []

        if file_list is None:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        else:
            self.file_list = [f for f in file_list if f.endswith('.h5')]

        if len(self.file_list) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        self.logger.info(f"Found {len(self.file_list)} h5 files:")
        for f in self.file_list:
            self.logger.info(f"  - {f}")

        self._preprocess_files(cache_id)

    def _get_cache_id(self, data_dir, file_list=None):
        """Hash of file names and mtimes, used to name the cache."""
        if file_list is None:
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
        else:
            files = sorted([f for f in file_list if f.endswith('.h5')])

        content = []
        for f in files:
            file_path = os.path.join(data_dir, f)
            mtime = os.path.getmtime(file_path)
            content.append(f"{f}_{mtime}")

        content = "_".join(content)
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def normalize_points(self, points):
        """Center the cloud and scale it into the unit sphere."""
        centroid = np.mean(points, axis=0)
        points = points - centroid

        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points

    def _load_hdf5_file(self, hdf5_filename):
        """Load points/colors/labels from an HDF5 file."""
        file_path = os.path.join(self.data_dir, hdf5_filename)
        self.logger.info(f"Loading {file_path}")
        with h5py.File(file_path, 'r') as f:
            points = np.array(f['points'])
            colors = np.array(f['colors'])
            labels = np.array(f['labels'])

        self.logger.info(f"Loaded {len(points)} points from {hdf5_filename}")
        return points, colors, labels

    def _preprocess_file(self, args):
        """Pre-compute and cache the blocks of one file."""
        filename, cache_id = args
        cache_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}.npz")
        block_infos = []

        if os.path.exists(cache_file):
            self.logger.info(f"Cache exists for {filename}, skipping preprocessing.")
            data = np.load(cache_file)
            num_blocks = data['num_blocks']
            for i in range(num_blocks):
                block_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}_block_{i}.npz")
                block_infos.append((block_file, i))
            return block_infos

        self.logger.info(f"Processing {filename}")

        points, colors, labels = self._load_hdf5_file(filename)

        if points is not None:
            blocks = self._assign_points_to_blocks(points, colors, labels, filename)

            num_blocks = len(blocks)
            for i, block in enumerate(blocks):
                block_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}_block_{i}.npz")
                np.savez_compressed(
                    block_file,
                    points=block['points'].astype(np.float32),
                    colors=block['colors'].astype(np.float32),
                    labels=block['labels'].astype(np.int64),
                    original_points=block['original_points'].astype(np.float32),
                    original_colors=block['original_colors'].astype(np.float32),
                    indices=block['indices'].astype(np.int64)
                )
                block_infos.append((block_file, i))

            np.savez_compressed(cache_file, num_blocks=num_blocks)

            self.logger.info(f"Created {num_blocks} blocks for {filename}")
            return block_infos
        else:
            self.logger.warning(f"No points loaded for {filename}")
            return []

    def _preprocess_files(self, cache_id):
        """Pre-process all files in parallel."""
        all_args = [(filename, cache_id) for filename in self.file_list]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._preprocess_file, all_args)

        for block_infos in results:
            self.block_info.extend(block_infos)

        self.logger.info(f"Total blocks created: {len(self.block_info)}")

    def _assign_points_to_blocks(self, points, colors, labels, filename):
        """Sample training/evaluation blocks from one point cloud."""

        import numba
        import numpy as np

        @numba.jit(nopython=True)
        def find_points_in_block(points, block_min, block_max, z_threshold=2.0):
            mask = np.zeros(len(points), dtype=np.bool_)
            for i in range(len(points)):
                if (block_min[0] <= points[i, 0] <= block_max[0] and
                        block_min[1] <= points[i, 1] <= block_max[1]):
                    z_center = (block_min[2] + block_max[2]) / 2
                    if abs(points[i, 2] - z_center) <= z_threshold:
                        mask[i] = True
            return np.where(mask)[0]

        def check_block_validity(block_indices, min_labels=1):
            """
            Check whether a sampled block is valid.
            """
            if len(block_indices) < self.num_points:
                return False

            block_labels = labels[block_indices]
            unique_labels = np.unique(block_labels)
            return len(unique_labels) >= min_labels

        def weighted_stratified_sampling(points, labels, colors, num_points, desired_class_proportions):
            """
            Weighted stratified sampling with the desired class proportions.
            Args:
                points: point coordinates (N, 3)
                labels: labels (N,)
                colors: colors (N, 3)
                num_points: number of points per block
                desired_class_proportions: dict {class: proportion}
            Returns:
                selected_indices: indices of the selected points
            """
            unique_classes = np.unique(labels)
            selected_indices = []
            total_proportion = sum(desired_class_proportions.values())

            normalized_proportions = {k: v / total_proportion for k, v in desired_class_proportions.items()}

            desired_num_points_per_class = {}
            for cls in unique_classes:
                proportion = normalized_proportions.get(cls, 0)
                desired_num_points_per_class[cls] = int(proportion * num_points)

            total_desired = sum(desired_num_points_per_class.values())
            diff = num_points - total_desired
            if diff != 0:
                max_class = max(desired_num_points_per_class, key=desired_num_points_per_class.get)
                desired_num_points_per_class[max_class] += diff

            for cls in unique_classes:
                class_indices = np.where(labels == cls)[0]
                n_samples = desired_num_points_per_class.get(cls, 0)
                if n_samples <= 0 or len(class_indices) == 0:
                    continue
                if len(class_indices) >= n_samples:
                    selected = np.random.choice(class_indices, n_samples, replace=False)
                else:
                    selected = np.random.choice(class_indices, n_samples, replace=True)
                selected_indices.extend(selected)

            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)
            return selected_indices

        blocks = []
        global_blocks = []
        local_blocks = []
        num_pcd = len(points)
        pcd_iter = int(num_pcd * self.sample_rate / self.num_points)
        normal_points = self.normalize_points(points)

        self.logger.info(f"Total points: {num_pcd}, Iterations: {pcd_iter}")

        desired_class_proportions = {
            0: 0.1,
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.3
        }

        for _ in range(pcd_iter):
            indices = weighted_stratified_sampling(
                points=points,
                labels=labels,
                colors=colors,
                num_points=self.num_points,
                desired_class_proportions=desired_class_proportions
            )

            block = {
                'points': normal_points[indices],
                'colors': colors[indices],
                'labels': labels[indices],
                'original_points': points[indices],
                'original_colors': colors[indices],
                'file_name': filename,
                'indices': indices
            }
            global_blocks.append(block)

        #     n_points = points.shape[0]  # N
        #     center = points[np.random.choice(n_points)][: 3]
        #     block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        #     block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        #
        #     block_indices = find_points_in_block(points, block_min, block_max)
        #     if check_block_validity(block_indices):
        #         sampled_indices = np.random.choice(block_indices, self.num_points, replace=False)
        #         block = {
        #             'points': normal_points[sampled_indices],
        #             'colors': colors[sampled_indices],
        #             'labels': labels[sampled_indices],
        #             'original_colors': colors[sampled_indices],
        #         }
        #         local_blocks.append(block)
        #
        #     end_time = time.time()
        #
        # blocks = global_blocks + local_blocks

        return global_blocks

    def __len__(self):
        return len(self.block_info)

    def __getitem__(self, idx):
        block_file, block_idx = self.block_info[idx]

        data = np.load(block_file)
        points = data['points']
        colors = data['colors']
        labels = data['labels']
        original_points = data['original_points']
        original_colors = data['original_colors']
        indices = data['indices']

        if self.transform:
            points, colors = self.apply_transform(points, colors)

        return {
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'labels': labels.astype(np.int64),
            'original_points': original_points.astype(np.float32),
            'original_colors': original_colors.astype(np.float32),
            'file_name': os.path.basename(block_file),
            'indices': indices.astype(np.int64)
        }

    def apply_transform(self, points, colors, keep_original=True):
        """Random rotation about Z, translation, scaling and color jitter."""
        if not self.transform:
            return points, colors

        if keep_original:
            points = points.copy()
            if colors is not None:
                colors = colors.copy()

        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rotation_matrix)

        translation = np.random.uniform(0.01, 0.1, size=(1, 3))
        points += translation

        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        if colors is not None:
            color_noise = np.random.normal(0, 0.02, colors.shape)
            colors = np.clip(colors + color_noise, 0, 1)

        return points, colors

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataset = BriPCDMulti(
        data_dir='../../data/CB/all/train',
        num_points=4096,
        block_size=1,
        sample_rate=0.2,
        transform=True,
        logger=logger
    )

    dataset = BriPCDMulti(
        data_dir='../../data/CB/all/val',
        num_points=4096,
        block_size=1,
        sample_rate=0.2,
        logger=logger
    )

    dataset = BriPCDMulti(
        data_dir='../../data/CB/all/test',
        num_points=4096,
        block_size=1,
        sample_rate=0.2,
        logger=logger
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for batch in dataloader:
        print(f"Batch points shape: {batch['points'].shape}")
        print(f"Batch colors shape: {batch['colors'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break

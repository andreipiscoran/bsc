import logging
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

try:
    import kagglehub
except Exception:
    kagglehub = None


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def download_faceforensicspp_dataset(dataset_ref="xdxd003/ff-c23"):
    """Download FaceForensics++ from Kaggle and return local dataset directory."""
    if kagglehub is None:
        raise ImportError(
            "kagglehub is required for FaceForensics++ download. "
            "Install with: pip install kagglehub[pandas-datasets]"
        )

    logger.info("Downloading dataset from Kaggle: %s", dataset_ref)
    download_path = kagglehub.dataset_download(dataset_ref)
    logger.info("Dataset downloaded to: %s", download_path)
    return download_path


class CIFAKEImageDataset(Dataset):
    """Custom Dataset for CIFAKE that loads images from directories."""

    def __init__(self, root_dir, transform=None, max_samples=None, split=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []

        self.real_aliases = {'real', 'original'}
        self.fake_aliases = {'fake', 'manipulated', 'deepfake', 'edited', 'synthetic'}

        logger.info(f"Scanning directory: {root_dir}")

        if self.split and self.split not in {'train', 'test'}:
            raise ValueError("split must be one of: None, 'train', 'test'")

        if os.path.exists(os.path.join(root_dir, 'train')):
            splits = [self.split] if self.split else ['train', 'test']
            for split in splits:
                split_dir = os.path.join(root_dir, split)
                if os.path.exists(split_dir):
                    self._collect_labeled_files(split_dir, max_samples=max_samples)

        elif os.path.exists(os.path.join(root_dir, 'REAL')) and os.path.exists(os.path.join(root_dir, 'FAKE')):
            self._collect_labeled_files(root_dir, max_samples=max_samples)
        else:
            self._collect_by_path_inference(root_dir, max_samples=max_samples)

        logger.info(f"Total dataset size: {len(self.image_paths)} images")

    def _is_image_file(self, file_name):
        return file_name.lower().endswith(IMAGE_EXTENSIONS)

    def _infer_label_from_parts(self, path_parts):
        lowered = {part.lower() for part in path_parts}
        if lowered.intersection(self.real_aliases):
            return 0
        if lowered.intersection(self.fake_aliases):
            return 1
        return None

    def _collect_labeled_files(self, root_dir, max_samples=None):
        by_label = {0: [], 1: []}

        for current_root, _, files in os.walk(root_dir):
            label = self._infer_label_from_parts(Path(current_root).parts)
            if label is None:
                continue

            image_files = [fname for fname in files if self._is_image_file(fname)]
            for fname in image_files:
                by_label[label].append(os.path.join(current_root, fname))

        for label, paths in by_label.items():
            selected = paths[:max_samples] if max_samples else paths
            self.image_paths.extend(selected)
            self.labels.extend([label] * len(selected))

        logger.info(
            "Found %s REAL and %s FAKE images",
            len(by_label[0]) if not max_samples else min(len(by_label[0]), max_samples),
            len(by_label[1]) if not max_samples else min(len(by_label[1]), max_samples),
        )

    def _collect_by_path_inference(self, root_dir, max_samples=None):
        by_label = {0: [], 1: []}

        for current_root, _, files in os.walk(root_dir):
            parts = Path(current_root).parts
            label = self._infer_label_from_parts(parts)

            for fname in files:
                if not self._is_image_file(fname):
                    continue

                if label is None:
                    inferred_from_name = self._infer_label_from_parts((fname,))
                    label_to_use = inferred_from_name
                else:
                    label_to_use = label

                if label_to_use is None:
                    continue

                by_label[label_to_use].append(os.path.join(current_root, fname))

        real_paths = by_label[0]
        fake_paths = by_label[1]
        logger.info("Found %s REAL and %s FAKE images", len(real_paths), len(fake_paths))

        if max_samples:
            per_class = max(1, max_samples // 2)
            real_paths = real_paths[:per_class]
            fake_paths = fake_paths[:per_class]

        self.image_paths.extend(real_paths)
        self.labels.extend([0] * len(real_paths))
        self.image_paths.extend(fake_paths)
        self.labels.extend([1] * len(fake_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

"""
Dataset Loader for Deepfake Detection

Supports loading face image datasets in the standardized structure:
    dataset/
        train/
            real/
            fake/
        validation/
            real/
            fake/
        test/
            real/
            fake/

Also supports unlabeled datasets for self-supervised pretraining.
Compatible with: FaceForensics++, Celeb-DF, WildDeepfake, DFDC, DeeperForensics, FaceShifter.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ml.preprocessing.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_ssl_augmentation,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class DeepfakeDataset(Dataset):
    """
    Labeled deepfake detection dataset.
    Loads images from real/ and fake/ subdirectories.
    Label: 0 = REAL, 1 = FAKE.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 320,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.image_size = image_size

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Collect samples
        self.samples: List[Tuple[str, int]] = []

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        if real_dir.exists():
            for path in self._collect_images(real_dir):
                self.samples.append((str(path), 0))

        if fake_dir.exists():
            for path in self._collect_images(fake_dir):
                self.samples.append((str(path), 1))

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

        # Shuffle and limit
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(
            "Loaded %d samples from %s/%s (real=%d, fake=%d)",
            len(self.samples), root_dir, split,
            sum(1 for _, l in self.samples if l == 0),
            sum(1 for _, l in self.samples if l == 1),
        )

        # Transforms
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def _collect_images(self, directory: Path) -> List[Path]:
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.rglob(f"*{ext}"))
        return sorted(images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            # Return a blank image if loading fails
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))

        tensor = self.transform(image)
        return tensor, label


class UnlabeledImageDataset(Dataset):
    """
    Unlabeled dataset for self-supervised pretraining.
    Returns two differently augmented views of each image.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 320,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root_dir)
        self.image_size = image_size

        # Collect all images recursively
        self.image_paths: List[str] = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(
                str(p) for p in self.root.rglob(f"*{ext}")
            )

        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")

        random.shuffle(self.image_paths)
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        logger.info("Unlabeled dataset: %d images from %s", len(self.image_paths), root_dir)

        # Two independent augmentation pipelines
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.ssl_aug = get_ssl_augmentation(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of the same image."""
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = self.base_transform(image)  # (C, H, W) float [0,1]

        view1 = self.ssl_aug(base)
        view2 = self.ssl_aug(base)

        return view1, view2


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    image_size: int = 320,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from a standard dataset directory.

    Returns dict with keys 'train', 'val', 'test'.
    """
    loaders = {}

    for split, augment in [("train", True), ("validation", False), ("test", False)]:
        split_dir = Path(dataset_path) / split
        if not split_dir.exists():
            logger.warning("Split '%s' not found at %s, skipping", split, split_dir)
            continue

        ds = DeepfakeDataset(
            root_dir=dataset_path,
            split=split,
            image_size=image_size,
            augment=augment,
            max_samples=max_samples,
        )

        key = "val" if split == "validation" else split
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=min(num_workers, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders


def create_ssl_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    image_size: int = 320,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for self-supervised pretraining (unlabeled)."""
    ds = UnlabeledImageDataset(
        root_dir=dataset_path,
        image_size=image_size,
        max_samples=max_samples,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

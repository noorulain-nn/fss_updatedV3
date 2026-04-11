import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, Omniglot, Caltech101
# from data.fss_dataset.pascal5i_reader import Pascal5iReader
import numpy as np
import random
from collections import defaultdict

# Module-level class for pickling support (required for Windows multiprocessing)
class RemappedSubset(torch.utils.data.Dataset):
	"""Custom dataset class that maps old labels to new sequential labels"""
	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = indices
	
	def __len__(self):
		return len(self.indices)
	
	def __getitem__(self, idx):
		original_idx, new_label = self.indices[idx]
		image, _ = self.dataset[original_idx]
		return image, new_label

# ─────────────────────────────────────────────────────────────────
# PASCAL-5i SUPPORT  (added for FSS)
# ─────────────────────────────────────────────────────────────────
from PIL import Image
import torchvision.transforms as T

# ── Corrected Pascal-5i section — replace previous version ──────
import torch.nn.functional as F_dl   # avoid name collision with APM's F
from scipy.io import loadmat

class Pascal5iEpisodic(torch.utils.data.Dataset):
    """
    Episodic wrapper around Pascal5iReader.

    The raw reader returns (img, mask) pairs.
    This wrapper builds proper few-shot episodes:
      - pick a target class
      - sample k_shot support images + 1 query image for that class
      - return binary masks (1=target class, 0=everything else)

    n_episodes controls how many episodes are pre-generated.
    All episodes are deterministic given the seed.
    """
    def __init__(self, pascal5i_reader, k_shot=5,
                 img_size=473, n_episodes=1000, seed=42):
        self.reader     = pascal5i_reader
        self.k_shot     = k_shot
        self.img_size   = img_size
        self.label_set  = pascal5i_reader.label_set  # list of class IDs

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Pre-generate episodes for full reproducibility
        rng = np.random.RandomState(seed)
        self.episodes = []

        for _ in range(n_episodes * 5):   # oversample, filter below
            if len(self.episodes) >= n_episodes:
                break

            # Pick a class from the label set (1-indexed inside reader)
            # label_set index 0 → class_img_map key 1, etc.
            ls_idx  = rng.randint(0, len(self.label_set))
            cls_key = ls_idx + 1   # class_img_map uses 1-based keys

            available = pascal5i_reader.get_img_containing_class(cls_key)
            if len(available) < k_shot + 1:
                continue

            chosen          = rng.choice(available, k_shot + 1, replace=False)
            support_indices = list(chosen[:k_shot])
            query_index     = int(chosen[k_shot])
            self.episodes.append((cls_key, support_indices, query_index))

    def __len__(self):
        return len(self.episodes)

    def _process(self, img_tensor, mask_tensor, cls_key):
        """
        Resize image and mask to img_size, normalize image,
        binarize mask for cls_key.
        """
        # img_tensor: [3, H, W]  (already tensor from reader)
        img_r = F_dl.interpolate(
            img_tensor.unsqueeze(0).float(),
            size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=True
        ).squeeze(0)                                   # [3, img_size, img_size]

        # Normalize
        img_n = (img_r - self.mean) / self.std        # [3, img_size, img_size]

        # mask_tensor: [H, W] long
        mask_r = F_dl.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze().long()                             # [img_size, img_size]

        # Binary mask: 1 where target class, 0 everywhere else
        binary = (mask_r == cls_key).long()            # [img_size, img_size]

        return img_n, binary

    def __getitem__(self, idx):
        cls_key, support_indices, query_index = self.episodes[idx]

        s_imgs, s_masks = [], []
        for si in support_indices:
            img, mask = self.reader[si]               # [3,H,W], [H,W]
            img_n, bin_mask = self._process(img, mask, cls_key)
            s_imgs.append(img_n)
            s_masks.append(bin_mask)

        q_img, q_mask = self.reader[query_index]
        q_img_n, q_bin_mask = self._process(q_img, q_mask, cls_key)

        return (
            torch.stack(s_imgs),    # [k_shot, 3, H, W]
            torch.stack(s_masks),   # [k_shot, H, W]
            q_img_n,                # [3, H, W]
            q_bin_mask              # [H, W]
        )


def prepare_pascal5i(data_root, fold=0, k_shot=5, img_size=473,
                     n_train_episodes=2000, n_test_episodes=1000,
                     val_fraction=0.1, batch_size=6, seed=42):
    """
    Returns train_loader, val_loader, test_loader for Pascal-5i fold.

    Args:
        data_root         : parent folder containing BOTH sbd/ and VOCdevkit/
        fold              : 0-3
        k_shot            : 1 or 5
        img_size          : resize all images/masks to this square size
        n_train_episodes  : how many train episodes to pre-generate
        n_test_episodes   : how many test episodes to pre-generate
        val_fraction      : fraction of train episodes held out for val
        batch_size        : episodes per batch
        seed              : random seed

    IMPORTANT — data_root must look like:
        data_root/
        ├── sbd/
        │   ├── train.txt
        │   ├── val.txt
        │   ├── img/
        │   └── cls/
        └── VOCdevkit/
            └── VOC2012/
                ├── JPEGImages/
                ├── SegmentationClass/
                └── ImageSets/Segmentation/
    """
    from data.fss_dataset.pascal5i_reader import Pascal5iReader # copied from RogerQi repo

    torch.manual_seed(seed)

    # train=True  → base classes (15 classes, SBD+VOC combined)
    # train=False → novel classes (5 classes, VOC val only)
    raw_train = Pascal5iReader(data_root, fold=fold, train=True)
    raw_test  = Pascal5iReader(data_root, fold=fold, train=False)

    n_val   = max(1, int(n_train_episodes * val_fraction))
    n_train = n_train_episodes - n_val

    train_ds = Pascal5iEpisodic(raw_train, k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_train, seed=seed)
    val_ds   = Pascal5iEpisodic(raw_train, k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_val,   seed=seed + 1)
    test_ds  = Pascal5iEpisodic(raw_test,  k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_test_episodes, seed=seed + 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=4, pin_memory=True
    )
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 1   # NUM_CLASSES=1 (binary)

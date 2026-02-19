import os
import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSSliceDataset(Dataset):
    """
    Dataset for a SINGLE packed .npy file containing all slices.

    Expected directory layout:
      data_dir/
        <exactly one> *.npy   (packed array)

    Packed file format:
      - dtype: float32 (recommended)
      - shape: (N, H, W)
      - value range: [-1, 1]

    Returns:
      - torch.Tensor of shape (1, H, W), dtype float32
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        mmap: bool = True,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train | val | test")

        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if len(npy_files) == 0:
            raise RuntimeError(
                f"No .npy file found in {data_dir}.\n"
                f"Expected exactly one packed .npy produced by preprocess.py."
            )
        if len(npy_files) != 1:
            raise RuntimeError(
                f"Found {len(npy_files)} .npy files in {data_dir}, but expected exactly 1 packed file.\n"
                f"Files:\n" + "\n".join(npy_files)
            )

        self.data_path = npy_files[0]
        self.split = split
        self.seed = int(seed)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.mmap = bool(mmap)

        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("train_ratio must be in (0,1)")
        if self.val_ratio < 0 or self.val_ratio >= 1:
            raise ValueError("val_ratio must be in [0,1)")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")

        # Memory-map for fast random access + low RAM
        self._packed = np.load(self.data_path, mmap_mode="r" if self.mmap else None)

        if self._packed.ndim != 3:
            raise ValueError(
                f"Packed .npy must have shape (N,H,W). Got shape: {self._packed.shape} in {self.data_path}"
            )

        self.n_total, self.H, self.W = self._packed.shape

        # Deterministic split
        rng = np.random.RandomState(self.seed)
        idx = np.arange(self.n_total)
        rng.shuffle(idx)

        n_train = int(self.train_ratio * self.n_total)
        n_val = int(self.val_ratio * self.n_total)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> torch.Tensor:
        idx = int(self.indices[i])
        arr = self._packed[idx]  # (H, W)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return x

    def get_packed_info(self) -> Tuple[str, Tuple[int, int, int]]:
        return self.data_path, (int(self.n_total), int(self.H), int(self.W))

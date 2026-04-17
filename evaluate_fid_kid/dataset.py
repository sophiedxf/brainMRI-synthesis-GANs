import glob
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSSliceDataset(Dataset):
    """
    Dataset for a single packed .npy file containing all slices.

    Expected directory layout:
      data_dir/
        <exactly one> *.npy         (packed slice array)
        <matching> *_metadata.npz   (slice-to-patient mapping)

    Packed file format:
      - dtype: float32 (recommended)
      - shape: (N, H, W)
      - value range: [-1, 1]

    Metadata format:
      - slice_patient_ids: array of shape (N,)

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
        self.metadata_path = os.path.splitext(self.data_path)[0] + "_metadata.npz"
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

        # Memory-map for fast random access + low RAM.
        self._packed = np.load(self.data_path, mmap_mode="r" if self.mmap else None)

        if self._packed.ndim != 3:
            raise ValueError(
                f"Packed .npy must have shape (N,H,W). Got shape: {self._packed.shape} in {self.data_path}"
            )

        self.n_total, self.H, self.W = self._packed.shape

        if not os.path.isfile(self.metadata_path):
            raise RuntimeError(
                "Patient-level splitting requires the preprocessing metadata sidecar.\n"
                f"Expected metadata file: {self.metadata_path}\n"
                "Please rerun preprocess/preprocess.py to regenerate the packed dataset with metadata."
            )

        with np.load(self.metadata_path, allow_pickle=False) as metadata:
            if "slice_patient_ids" not in metadata:
                raise ValueError(
                    f"Metadata file {self.metadata_path} is missing 'slice_patient_ids'. "
                    "Please regenerate the packed dataset with the updated preprocessing script."
                )
            self.slice_patient_ids = np.asarray(metadata["slice_patient_ids"]).astype(str, copy=False)
        if self.slice_patient_ids.ndim != 1:
            raise ValueError(
                f"'slice_patient_ids' must be a 1D array. Got shape {self.slice_patient_ids.shape} "
                f"in {self.metadata_path}"
            )
        if len(self.slice_patient_ids) != self.n_total:
            raise ValueError(
                "Packed data and metadata length mismatch: "
                f"{self.n_total} slices in {self.data_path} vs {len(self.slice_patient_ids)} patient entries "
                f"in {self.metadata_path}"
            )

        unique_patient_ids = np.unique(self.slice_patient_ids)
        if len(unique_patient_ids) == 0:
            raise ValueError(f"No patient IDs found in metadata file: {self.metadata_path}")

        # Split by unique patients so all slices from one patient stay together.
        rng = np.random.RandomState(self.seed)
        patient_ids = unique_patient_ids.copy()
        rng.shuffle(patient_ids)

        n_train = int(self.train_ratio * len(patient_ids))
        n_val = int(self.val_ratio * len(patient_ids))

        if split == "train":
            selected_patient_ids = patient_ids[:n_train]
        elif split == "val":
            selected_patient_ids = patient_ids[n_train:n_train + n_val]
        else:
            selected_patient_ids = patient_ids[n_train + n_val:]

        self.indices = np.flatnonzero(np.isin(self.slice_patient_ids, selected_patient_ids))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> torch.Tensor:
        idx = int(self.indices[i])
        arr = self._packed[idx]  # (H, W)

        # Make a writable float32 copy so torch.from_numpy does not warn on read-only mmap slices.
        arr = np.array(arr, dtype=np.float32, copy=True)

        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return x

    def get_packed_info(self) -> Tuple[str, Tuple[int, int, int]]:
        return self.data_path, (int(self.n_total), int(self.H), int(self.W))

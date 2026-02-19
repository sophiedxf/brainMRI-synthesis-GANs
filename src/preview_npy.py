import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def to_uint8_0_255(x: np.ndarray) -> np.ndarray:
    """
    x expected in [-1, 1] (your pipeline). Convert to uint8 [0,255].
    If your array is already 0..1 or 0..255, adjust accordingly.
    """
    x01 = (x + 1.0) / 2.0
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).round().astype(np.uint8)


def save_grid(imgs_u8: np.ndarray, out_path: str, nrow: int = 8):
    """
    imgs_u8: (64, H, W) uint8
    Saves an 8x8 grid as a single PNG.
    """
    assert imgs_u8.shape[0] == nrow * nrow, "Need exactly 64 images for an 8x8 grid."

    ncol = nrow
    H, W = imgs_u8.shape[1], imgs_u8.shape[2]

    grid = np.zeros((nrow * H, ncol * W), dtype=np.uint8)

    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            grid[i * H : (i + 1) * H, j * W : (j + 1) * W] = imgs_u8[idx]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npy",
        type=str,
        default=r"data\preprocessed_slices_64\brats2023_t2f_64_packed.npy",
        help="Path to packed .npy file",
    )
    p.add_argument(
        "--out",
        type=str,
        default=r"data\png_samples_64\preview_packed_grid_8x8.png",
        help="Output PNG path",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num", type=int, default=64, help="Number of slices to show (must be 64 for 8x8)")
    args = p.parse_args()

    if args.num != 64:
        raise ValueError("--num must be 64 for an 8x8 grid in this script.")

    if not os.path.isfile(args.npy):
        raise FileNotFoundError(args.npy)

    arr = np.load(args.npy, mmap_mode="r")  # (N,64,64) float32
    if arr.ndim != 3:
        raise ValueError(f"Expected (N,H,W) array, got shape {arr.shape}")
    if arr.shape[1] != 64 or arr.shape[2] != 64:
        raise ValueError(f"Expected slice size 64x64, got {arr.shape[1]}x{arr.shape[2]}")

    N = arr.shape[0]
    if N < 64:
        raise ValueError(f"Need at least 64 slices, but file contains {N}")

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(N, size=64, replace=False)
    samples = np.asarray(arr[idx], dtype=np.float32)  # (64,64,64)

    samples_u8 = to_uint8_0_255(samples)
    save_grid(samples_u8, args.out, nrow=8)

    print(f"Loaded: {args.npy}")
    print(f"Array shape: {arr.shape} | dtype={arr.dtype}")
    print(f"Saved 8x8 grid to: {args.out}")


if __name__ == "__main__":
    main()

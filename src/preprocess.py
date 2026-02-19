# preprocess.py (updated: resized background mask approach)

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image

VALID_IMAGE_SIZES = {64, 128, 256}

MODALITY_SUFFIX_MAP = {
    "t1n": ["-t1n.nii.gz"],
    "t1c": ["-t1c.nii.gz"],
    "t2w": ["-t2w.nii.gz"],
    "t2f": ["-t2f.nii.gz"],
    "t1": ["-t1n.nii.gz"],
    "t1ce": ["-t1c.nii.gz"],
    "t2": ["-t2w.nii.gz"],
    "flair": ["-t2f.nii.gz"],
}


def robust_normalise_to_minus1_1(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    bg_mask = vol == 0
    brain = vol[~bg_mask]

    if brain.size == 0:
        return np.full_like(vol, -1.0, dtype=np.float32)

    mean = brain.mean()
    std = brain.std() + 1e-6
    vol_z = (vol - mean) / std

    brain_z = vol_z[~bg_mask]
    p1, p99 = np.percentile(brain_z, [1, 99])
    vol_clip = np.clip(vol_z, p1, p99)

    vol_scaled = 2.0 * (vol_clip - p1) / (p99 - p1 + 1e-6) - 1.0
    vol_scaled[bg_mask] = -1.0
    return vol_scaled.astype(np.float32)


def simple_xy_crop(vol: np.ndarray, bg_value: float = -1.0) -> np.ndarray:
    mask = vol != bg_value
    coords = np.where(mask)
    if coords[0].size == 0:
        return vol
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    return vol[x_min:x_max + 1, y_min:y_max + 1, :]


def find_modality_files(raw_dir: str, modality: str):
    modality = modality.lower()
    if modality not in MODALITY_SUFFIX_MAP:
        raise ValueError(f"Unknown modality '{modality}'. Choose from {sorted(set(MODALITY_SUFFIX_MAP.keys()))}")

    files = []
    for suf in MODALITY_SUFFIX_MAP[modality]:
        files.extend(glob.glob(os.path.join(raw_dir, f"**/*{suf}"), recursive=True))
    return sorted(set(files))


def slice_foreground_score(slice2d: np.ndarray, thr: float = -0.9) -> int:
    return int(np.count_nonzero(slice2d > thr))


def _patient_rng(seed: int, patient_index: int) -> np.random.RandomState:
    mixed = (seed * 1000003 + patient_index * 9176) % (2**32 - 1)
    return np.random.RandomState(mixed)


def choose_slice_indices(
    vol: np.ndarray,
    min_foreground: int,
    max_slices_per_patient: int | None,
    selection: str,
    rng: np.random.RandomState,
):
    candidates = []
    for z in range(vol.shape[2]):
        sl = vol[:, :, z]
        score = slice_foreground_score(sl)
        if score >= min_foreground:
            candidates.append((z, score))

    if len(candidates) == 0:
        return []

    if max_slices_per_patient is None or max_slices_per_patient <= 0:
        return [z for z, _ in candidates]

    k = min(max_slices_per_patient, len(candidates))

    if selection == "topk_foreground":
        candidates.sort(key=lambda t: t[1], reverse=True)
        chosen = [z for z, _ in candidates[:k]]
        chosen.sort()
        return chosen

    if selection == "uniform":
        candidates.sort(key=lambda t: t[0])
        z_list = [z for z, _ in candidates]
        if k == 1:
            return [z_list[len(z_list) // 2]]
        idx = np.linspace(0, len(z_list) - 1, k).round().astype(int)
        chosen = [z_list[i] for i in idx]
        return sorted(set(chosen))

    if selection == "random":
        candidates.sort(key=lambda t: t[0])
        z_list = [z for z, _ in candidates]
        chosen = rng.choice(z_list, size=k, replace=False).tolist()
        chosen.sort()
        return chosen

    raise ValueError("selection must be one of: topk_foreground | uniform | random")


def save_slice_png(slice2d: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = (slice2d + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(x, mode="L").save(out_path)


def _load_and_prepare_volume(nifti_path: str) -> np.ndarray:
    img = nib.load(nifti_path)
    vol = img.get_fdata().astype(np.float32)
    vol = np.rot90(vol, k=1, axes=(0, 1))
    vol = robust_normalise_to_minus1_1(vol)
    vol = simple_xy_crop(vol, bg_value=-1.0)
    return vol


def count_total_slices(
    paths: list[str],
    target_total_slices: int | None,
    max_slices_per_patient: int | None,
    min_foreground: int,
    selection: str,
    seed: int,
) -> int:
    total = 0
    for patient_index, p in enumerate(tqdm(paths, desc="Pass 1/2: counting slices")):
        if target_total_slices is not None and total >= target_total_slices:
            break

        vol = _load_and_prepare_volume(p)
        rng = _patient_rng(seed, patient_index)
        chosen = choose_slice_indices(vol, min_foreground, max_slices_per_patient, selection, rng)

        if target_total_slices is not None:
            remaining = target_total_slices - total
            chosen = chosen[:max(0, remaining)]

        total += len(chosen)

    return total


def write_packed_slices(
    paths: list[str],
    packed_out_path: str,
    target_size: int,
    total_slices: int,
    target_total_slices: int | None,
    max_slices_per_patient: int | None,
    min_foreground: int,
    selection: str,
    seed: int,
    save_png_samples: bool,
    png_dir: str,
    png_every_n_patients: int,
    png_max_per_patient: int,
):
    os.makedirs(os.path.dirname(packed_out_path) or ".", exist_ok=True)

    arr = np.lib.format.open_memmap(
        packed_out_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_slices, target_size, target_size),
    )

    write_idx = 0
    for patient_index, p in enumerate(tqdm(paths, desc="Pass 2/2: writing packed array")):
        if write_idx >= total_slices:
            break

        vol = _load_and_prepare_volume(p)
        rng = _patient_rng(seed, patient_index)
        chosen_z = choose_slice_indices(vol, min_foreground, max_slices_per_patient, selection, rng)

        if target_total_slices is not None:
            remaining = target_total_slices - write_idx
            chosen_z = chosen_z[:max(0, remaining)]

        if len(chosen_z) == 0:
            continue

        base = os.path.basename(p).replace(".nii.gz", "")
        do_png = save_png_samples and png_every_n_patients > 0 and (patient_index % png_every_n_patients == 0)
        png_saved = 0

        for z in chosen_z:
            if write_idx >= total_slices:
                break

            # Original slice in [-1,1] with exact background == -1 in vol
            sl = vol[:, :, z].astype(np.float32, copy=False)

            # Background mask BEFORE resize (exact background only)
            # Using <= -0.999 is robust to any rare float quirks.
            bg_mask = (sl <= -0.999)

            # Resize image (linear, anti-aliased)
            sl_resized = resize(
                sl,
                (target_size, target_size),
                order=1,
                mode="constant",
                cval=-1.0,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)

            # Resize mask (nearest-neighbour, no anti-aliasing)
            bg_mask_resized = resize(
                bg_mask.astype(np.uint8),
                (target_size, target_size),
                order=0,
                mode="constant",
                cval=1,                # outside original crop treated as background
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.uint8).astype(bool)

            # Enforce perfectly constant background after interpolation
            sl_resized[bg_mask_resized] = -1.0

            arr[write_idx] = sl_resized
            write_idx += 1

            if do_png and png_saved < max(0, png_max_per_patient):
                png_path = os.path.join(png_dir, f"{base}_z{z:03d}.png")
                save_slice_png(sl_resized, png_path)
                png_saved += 1

    arr.flush()
    return write_idx


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS2023 GLI volumes into 2D slices packed into a single .npy.")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--target_size", type=int, required=True)
    parser.add_argument("--min_foreground", type=int, default=500)

    parser.add_argument("--max_slices_per_patient", type=int, default=0)
    parser.add_argument("--target_total_slices", type=int, default=0)
    parser.add_argument("--selection", type=str, default="topk_foreground", choices=["topk_foreground", "uniform", "random"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--packed_out", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--save_png_samples", action="store_true")
    parser.add_argument("--png_dir", type=str, default="")
    parser.add_argument("--png_every_n_patients", type=int, default=50)
    parser.add_argument("--png_max_per_patient", type=int, default=8)

    args = parser.parse_args()

    if args.target_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"--target_size must be one of {sorted(VALID_IMAGE_SIZES)}")

    max_slices_per_patient = args.max_slices_per_patient if args.max_slices_per_patient > 0 else None
    target_total_slices = args.target_total_slices if args.target_total_slices > 0 else None

    os.makedirs(args.out_dir, exist_ok=True)

    if args.png_dir.strip() == "":
        args.png_dir = os.path.join(args.out_dir, "png_samples")

    if args.packed_out.strip() == "":
        args.packed_out = os.path.join(args.out_dir, f"brats2023_{args.modality.lower()}_{args.target_size}_packed.npy")

    paths = find_modality_files(args.raw_dir, args.modality)
    if len(paths) == 0:
        raise RuntimeError(f"No files found for modality={args.modality} under {args.raw_dir}.")

    total_slices = count_total_slices(
        paths=paths,
        target_total_slices=target_total_slices,
        max_slices_per_patient=max_slices_per_patient,
        min_foreground=args.min_foreground,
        selection=args.selection,
        seed=args.seed,
    )

    if total_slices == 0:
        raise RuntimeError("No slices selected. Try lowering --min_foreground or changing --selection.")

    print(f"\nSelected slices to write: {total_slices}")
    print(f"Packed output path: {args.packed_out}")

    written = write_packed_slices(
        paths=paths,
        packed_out_path=args.packed_out,
        target_size=args.target_size,
        total_slices=total_slices,
        target_total_slices=target_total_slices,
        max_slices_per_patient=max_slices_per_patient,
        min_foreground=args.min_foreground,
        selection=args.selection,
        seed=args.seed,
        save_png_samples=args.save_png_samples,
        png_dir=args.png_dir,
        png_every_n_patients=args.png_every_n_patients,
        png_max_per_patient=args.png_max_per_patient,
    )

    print(f"\nDone. Packed slices written: {written}")
    print(f"Packed .npy file: {args.packed_out}")
    if args.save_png_samples:
        print(f"PNG samples directory: {args.png_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import os

def _validate_ratios(train_ratio: float, val_ratio: float):
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("--train_ratio must be in (0,1)")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("--val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("--train_ratio + --val_ratio must be < 1")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _sample_subset_indices(length: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    rng = np.random.RandomState(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    return indices[:limit].tolist()


def _split_seed_offset(split: str) -> int:
    offsets = {
        "train": 11,
        "val": 23,
        "test": 37,
    }
    return offsets[split]


def _load_subset_tensor(
    data_dir: str,
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    limit: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.Tensor:
    dataset = BraTSSliceDataset(
        data_dir,
        split=split,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    subset_indices = _sample_subset_indices(len(dataset), limit, seed=seed + _split_seed_offset(split))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    chunks: list[torch.Tensor] = []
    for batch in loader:
        chunks.append(batch.to(torch.float32))
    if not chunks:
        raise RuntimeError(f"No samples loaded for split '{split}'.")
    return torch.cat(chunks, dim=0)


def _generate_fake_tensor(
    ckpt_path: str,
    use_ema: bool,
    num_fake: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        device = get_device()
        G, image_size, z_dim, chosen = _load_generator_from_ckpt(ckpt_path, device, use_ema)

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        batches: list[torch.Tensor] = []
        seen = 0
        while seen < num_fake:
            bsz = min(batch_size, num_fake - seen)
            z = torch.randn(bsz, z_dim, 1, 1, generator=gen, device=device)
            x = G(z).detach().cpu().to(torch.float32)
            batches.append(x)
            seen += bsz

        return torch.cat(batches, dim=0), {
            "generator_used": chosen,
            "image_size": image_size,
            "z_dim": z_dim,
        }


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def _to_3ch_uint8(x: torch.Tensor) -> torch.Tensor:
    x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    if x01.size(1) == 1:
        x01 = x01.repeat(1, 3, 1, 1)
    return (x01 * 255.0).round().to(torch.uint8)


def _extract_inception_features(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    device = get_device()
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    features = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch = x[start:start + batch_size]
            batch_uint8 = _to_3ch_uint8(batch).to(device, non_blocking=True)
            (batch_features,) = extractor(batch_uint8)
            features.append(batch_features.detach().cpu().to(torch.float32))

    return torch.cat(features, dim=0)


def _nearest_l2(query: torch.Tensor, reference: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    query = _flatten(query)
    reference = _flatten(reference)

    ref_sq = (reference * reference).sum(dim=1).unsqueeze(0)
    ref_t = reference.t()

    min_dist = []
    min_idx = []
    for start in range(0, query.size(0), chunk_size):
        q = query[start:start + chunk_size]
        q_sq = (q * q).sum(dim=1, keepdim=True)
        dist = q_sq + ref_sq - 2.0 * (q @ ref_t)
        dist = dist.clamp_min_(0.0)
        best_dist, best_idx = torch.min(dist, dim=1)
        min_dist.append(best_dist.cpu())
        min_idx.append(best_idx.cpu())
    return torch.cat(min_dist), torch.cat(min_idx)


def _nearest_cosine(query: torch.Tensor, reference: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    query = _flatten(query)
    reference = _flatten(reference)

    query = torch.nn.functional.normalize(query, dim=1)
    reference = torch.nn.functional.normalize(reference, dim=1)
    ref_t = reference.t()

    max_sim = []
    max_idx = []
    for start in range(0, query.size(0), chunk_size):
        q = query[start:start + chunk_size]
        sim = q @ ref_t
        best_sim, best_idx = torch.max(sim, dim=1)
        max_sim.append(best_sim.cpu())
        max_idx.append(best_idx.cpu())
    return torch.cat(max_sim), torch.cat(max_idx)


def _train_self_nn_l2(train_real: torch.Tensor, chunk_size: int) -> torch.Tensor:
    train_flat = _flatten(train_real)
    train_sq = (train_flat * train_flat).sum(dim=1).unsqueeze(0)
    train_t = train_flat.t()

    mins = []
    for start in range(0, train_flat.size(0), chunk_size):
        chunk = train_flat[start:start + chunk_size]
        chunk_sq = (chunk * chunk).sum(dim=1, keepdim=True)
        dist = chunk_sq + train_sq - 2.0 * (chunk @ train_t)
        dist = dist.clamp_min_(0.0)

        row_idx = torch.arange(dist.size(0))
        col_idx = torch.arange(start, start + dist.size(0))
        dist[row_idx, col_idx] = float("inf")

        best_dist, _ = torch.min(dist, dim=1)
        mins.append(best_dist.cpu())
    return torch.cat(mins)


def _to_uint8_image(x: torch.Tensor) -> Image.Image:
    arr = x.squeeze(0).detach().cpu().numpy()
    arr = ((arr + 1.0) / 2.0).clip(0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _save_triplet_image(
    out_path: str,
    fake_img: torch.Tensor,
    train_img: torch.Tensor,
    ref_img: torch.Tensor,
    caption: str,
):
    tiles = [
        ("fake", _to_uint8_image(fake_img)),
        ("nearest_train", _to_uint8_image(train_img)),
        ("nearest_ref", _to_uint8_image(ref_img)),
    ]

    pad = 12
    top_h = 34
    tile_w, tile_h = tiles[0][1].size
    canvas_w = len(tiles) * tile_w + (len(tiles) + 1) * pad
    canvas_h = tile_h + top_h + 2 * pad + 20

    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(tiles):
        x = pad + idx * (tile_w + pad)
        y = top_h
        canvas.paste(image, (x, y))
        draw.text((x, 8), label, fill=0)

    draw.text((pad, canvas_h - 18), caption[:140], fill=0)
    canvas.save(out_path)


def _write_csv(rows: list[dict], out_path: str):
    fieldnames = [
        "fake_index",
        "train_nn_index_l2",
        "reference_nn_index_l2",
        "train_l2",
        "reference_l2",
        "l2_gap_reference_minus_train",
        "train_ref_l2_ratio",
        "train_nn_index_cosine",
        "reference_nn_index_cosine",
        "train_cosine",
        "reference_cosine",
        "cosine_gap_train_minus_reference",
        "suspicious_rank_l2",
        "train_nn_index_feature_l2",
        "reference_nn_index_feature_l2",
        "train_feature_l2",
        "reference_feature_l2",
        "feature_l2_gap_reference_minus_train",
        "train_ref_feature_l2_ratio",
        "suspicious_rank_feature_l2",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_summary(out_path: str, lines: list[str]):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Audit generated images for memorization risk using nearest-neighbor comparisons.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing packed dataset")
    p.add_argument("--out_dir", type=str, default="runs/privacy_audit", help="Output directory for audit results")
    p.add_argument("--reference_split", type=str, default="test", choices=["val", "test"], help="Held-out split used as the non-training reference")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_fake", type=int, default=2000, help="How many fake images to generate for the audit")
    p.add_argument("--num_train_real", type=int, default=2000, help="How many training images to compare against (0 = all)")
    p.add_argument("--num_reference_real", type=int, default=2000, help="How many held-out images to compare against (0 = all)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0, help="Set to 0 on Windows if multiprocessing causes issues")
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", help="Use G_ema if present in checkpoint")
    p.add_argument("--no_ema", action="store_true", help="Force using raw G even if G_ema exists")
    p.add_argument("--chunk_size", type=int, default=512, help="Chunk size for pairwise distance calculations")
    p.add_argument("--save_examples", type=int, default=25, help="How many most suspicious triplets to save")
    return p.parse_args()


def _lazy_imports():
    global np, torch, Image, ImageDraw, DataLoader, Subset
    global BraTSSliceDataset, _load_generator_from_ckpt, get_device
    global FeatureExtractorInceptionV3

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader, Subset
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    from dataset import BraTSSliceDataset
    from eval_fid import _load_generator_from_ckpt, get_device


def main():
    args = parse_args()
    _lazy_imports()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if args.num_fake < 1:
        raise ValueError("--num_fake must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be >= 1")
    if args.save_examples < 0:
        raise ValueError("--save_examples must be >= 0")
    _validate_ratios(args.train_ratio, args.val_ratio)

    use_ema = bool(args.use_ema) and (not args.no_ema)
    _ensure_dir(args.out_dir)
    examples_dir = os.path.join(args.out_dir, "suspicious_examples")
    _ensure_dir(examples_dir)

    print("Device:", get_device())
    print("Checkpoint:", args.ckpt)
    print("Reference split:", args.reference_split)

    train_real = _load_subset_tensor(
        data_dir=args.data_dir,
        split="train",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        limit=args.num_train_real,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    reference_real = _load_subset_tensor(
        data_dir=args.data_dir,
        split=args.reference_split,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        limit=args.num_reference_real,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    fake_images, fake_meta = _generate_fake_tensor(
        ckpt_path=args.ckpt,
        use_ema=use_ema,
        num_fake=args.num_fake,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"Loaded {len(train_real)} train images, {len(reference_real)} {args.reference_split} images, {len(fake_images)} fake images")

    train_l2, train_l2_idx = _nearest_l2(fake_images, train_real, chunk_size=args.chunk_size)
    ref_l2, ref_l2_idx = _nearest_l2(fake_images, reference_real, chunk_size=args.chunk_size)
    train_cos, train_cos_idx = _nearest_cosine(fake_images, train_real, chunk_size=args.chunk_size)
    ref_cos, ref_cos_idx = _nearest_cosine(fake_images, reference_real, chunk_size=args.chunk_size)
    print("Extracting Inception feature-space embeddings...")
    train_features = _extract_inception_features(train_real, batch_size=args.batch_size)
    reference_features = _extract_inception_features(reference_real, batch_size=args.batch_size)
    fake_features = _extract_inception_features(fake_images, batch_size=args.batch_size)
    train_feature_l2, train_feature_l2_idx = _nearest_l2(fake_features, train_features, chunk_size=args.chunk_size)
    ref_feature_l2, ref_feature_l2_idx = _nearest_l2(fake_features, reference_features, chunk_size=args.chunk_size)

    train_self_l2 = _train_self_nn_l2(train_real, chunk_size=args.chunk_size)
    train_self_feature_l2 = _train_self_nn_l2(train_features, chunk_size=args.chunk_size)

    eps = 1e-8
    l2_gap = ref_l2 - train_l2
    l2_ratio = train_l2 / (ref_l2 + eps)
    cos_gap = train_cos - ref_cos
    feature_l2_gap = ref_feature_l2 - train_feature_l2
    feature_l2_ratio = train_feature_l2 / (ref_feature_l2 + eps)

    suspicious_order = torch.argsort(l2_gap, descending=True)
    suspicious_ranks = torch.empty_like(suspicious_order)
    suspicious_ranks[suspicious_order] = torch.arange(1, len(fake_images) + 1)
    suspicious_order_feature = torch.argsort(feature_l2_gap, descending=True)
    suspicious_ranks_feature = torch.empty_like(suspicious_order_feature)
    suspicious_ranks_feature[suspicious_order_feature] = torch.arange(1, len(fake_images) + 1)

    rows = []
    for i in range(len(fake_images)):
        rows.append({
            "fake_index": i,
            "train_nn_index_l2": int(train_l2_idx[i]),
            "reference_nn_index_l2": int(ref_l2_idx[i]),
            "train_l2": float(train_l2[i]),
            "reference_l2": float(ref_l2[i]),
            "l2_gap_reference_minus_train": float(l2_gap[i]),
            "train_ref_l2_ratio": float(l2_ratio[i]),
            "train_nn_index_cosine": int(train_cos_idx[i]),
            "reference_nn_index_cosine": int(ref_cos_idx[i]),
            "train_cosine": float(train_cos[i]),
            "reference_cosine": float(ref_cos[i]),
            "cosine_gap_train_minus_reference": float(cos_gap[i]),
            "suspicious_rank_l2": int(suspicious_ranks[i]),
            "train_nn_index_feature_l2": int(train_feature_l2_idx[i]),
            "reference_nn_index_feature_l2": int(ref_feature_l2_idx[i]),
            "train_feature_l2": float(train_feature_l2[i]),
            "reference_feature_l2": float(ref_feature_l2[i]),
            "feature_l2_gap_reference_minus_train": float(feature_l2_gap[i]),
            "train_ref_feature_l2_ratio": float(feature_l2_ratio[i]),
            "suspicious_rank_feature_l2": int(suspicious_ranks_feature[i]),
        })

    csv_path = os.path.join(args.out_dir, "privacy_audit.csv")
    _write_csv(rows, csv_path)

    save_examples = min(args.save_examples, len(fake_images))
    for rank, fake_idx in enumerate(suspicious_order[:save_examples], start=1):
        fake_idx_int = int(fake_idx)
        train_idx_int = int(train_l2_idx[fake_idx_int])
        ref_idx_int = int(ref_l2_idx[fake_idx_int])
        caption = (
            f"rank={rank} fake={fake_idx_int} "
            f"train_l2={float(train_l2[fake_idx_int]):.4f} "
            f"{args.reference_split}_l2={float(ref_l2[fake_idx_int]):.4f} "
            f"gap={float(l2_gap[fake_idx_int]):.4f}"
        )
        out_path = os.path.join(examples_dir, f"rank_{rank:03d}_fake_{fake_idx_int:04d}.png")
        _save_triplet_image(
            out_path,
            fake_images[fake_idx_int],
            train_real[train_idx_int],
            reference_real[ref_idx_int],
            caption,
        )

    self_p01 = float(torch.quantile(train_self_l2, 0.01))
    self_p05 = float(torch.quantile(train_self_l2, 0.05))
    self_p10 = float(torch.quantile(train_self_l2, 0.10))
    self_feature_p01 = float(torch.quantile(train_self_feature_l2, 0.01))
    self_feature_p05 = float(torch.quantile(train_self_feature_l2, 0.05))
    self_feature_p10 = float(torch.quantile(train_self_feature_l2, 0.10))

    summary_lines = [
        "Privacy audit summary",
        f"checkpoint: {args.ckpt}",
        f"generator_used: {fake_meta['generator_used']}",
        f"reference_split: {args.reference_split}",
        f"image_size: {fake_meta['image_size']}",
        f"num_fake: {len(fake_images)}",
        f"num_train_real: {len(train_real)}",
        f"num_reference_real: {len(reference_real)}",
        "",
        "Nearest-neighbor summary",
        f"mean_train_l2: {float(train_l2.mean()):.6f}",
        f"mean_reference_l2: {float(ref_l2.mean()):.6f}",
        f"median_train_l2: {float(train_l2.median()):.6f}",
        f"median_reference_l2: {float(ref_l2.median()):.6f}",
        f"fraction_train_closer_l2: {float((train_l2 < ref_l2).float().mean()):.6f}",
        f"fraction_train_closer_cosine: {float((train_cos > ref_cos).float().mean()):.6f}",
        f"mean_l2_gap_reference_minus_train: {float(l2_gap.mean()):.6f}",
        f"max_l2_gap_reference_minus_train: {float(l2_gap.max()):.6f}",
        "",
        "Feature-space nearest-neighbor summary (Inception 2048)",
        f"mean_train_feature_l2: {float(train_feature_l2.mean()):.6f}",
        f"mean_reference_feature_l2: {float(ref_feature_l2.mean()):.6f}",
        f"median_train_feature_l2: {float(train_feature_l2.median()):.6f}",
        f"median_reference_feature_l2: {float(ref_feature_l2.median()):.6f}",
        f"fraction_train_closer_feature_l2: {float((train_feature_l2 < ref_feature_l2).float().mean()):.6f}",
        f"mean_feature_l2_gap_reference_minus_train: {float(feature_l2_gap.mean()):.6f}",
        f"max_feature_l2_gap_reference_minus_train: {float(feature_l2_gap.max()):.6f}",
        "",
        "Train-to-train baseline (sampled train pool)",
        f"train_self_nn_l2_p01: {self_p01:.6f}",
        f"train_self_nn_l2_p05: {self_p05:.6f}",
        f"train_self_nn_l2_p10: {self_p10:.6f}",
        f"fraction_fake_below_train_self_p01: {float((train_l2 <= self_p01).float().mean()):.6f}",
        f"fraction_fake_below_train_self_p05: {float((train_l2 <= self_p05).float().mean()):.6f}",
        f"fraction_fake_below_train_self_p10: {float((train_l2 <= self_p10).float().mean()):.6f}",
        "",
        "Feature-space train-to-train baseline",
        f"train_self_feature_l2_p01: {self_feature_p01:.6f}",
        f"train_self_feature_l2_p05: {self_feature_p05:.6f}",
        f"train_self_feature_l2_p10: {self_feature_p10:.6f}",
        f"fraction_fake_below_train_self_feature_p01: {float((train_feature_l2 <= self_feature_p01).float().mean()):.6f}",
        f"fraction_fake_below_train_self_feature_p05: {float((train_feature_l2 <= self_feature_p05).float().mean()):.6f}",
        f"fraction_fake_below_train_self_feature_p10: {float((train_feature_l2 <= self_feature_p10).float().mean()):.6f}",
        "",
        "Interpretation notes",
        "Lower fake-to-train distances can indicate memorization risk, especially when train matches are much closer than held-out matches.",
        "The train-to-train baseline gives context: fake samples below the lowest train-to-train distances are more suspicious.",
        "Feature-space distance is usually more meaningful than raw pixels for structural similarity, so it should carry more weight in your interpretation.",
        "This is an audit, not a formal privacy guarantee.",
    ]

    summary_path = os.path.join(args.out_dir, "summary.txt")
    _write_summary(summary_path, summary_lines)

    print("\n=== Privacy Audit Summary ===")
    for line in summary_lines[:20]:
        print(line)
    if len(summary_lines) > 20:
        print("...")
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved suspicious examples to: {examples_dir}")


if __name__ == "__main__":
    main()

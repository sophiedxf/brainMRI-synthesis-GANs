import argparse
import csv
import os
import re
from pathlib import Path


def _parse_epoch(path: Path) -> int | None:
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _sort_checkpoints(paths: list[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda p: (_parse_epoch(p) is None, _parse_epoch(p) or 0, p.name),
    )


def _discover_all_checkpoints(ckpt_dir: str, include_latest: bool) -> list[Path]:
    root = Path(ckpt_dir)
    if not root.is_dir():
        raise FileNotFoundError(ckpt_dir)

    ckpts = list(root.glob("checkpoint_epoch_*.pt"))
    if include_latest:
        latest = root / "checkpoint_latest.pt"
        if latest.is_file():
            ckpts.append(latest)

    ckpts = _sort_checkpoints(ckpts)
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts


def _resolve_named_checkpoints(ckpt_dir: str, names: list[str]) -> list[Path]:
    root = Path(ckpt_dir)
    resolved: list[Path] = []
    missing: list[str] = []

    for name in names:
        path = root / name
        if path.is_file():
            resolved.append(path)
        else:
            missing.append(name)

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Requested checkpoint file(s) not found in {ckpt_dir}: {missing_text}")

    return _sort_checkpoints(resolved)


def _resolve_epoch_checkpoints(ckpt_dir: str, epochs: list[int], include_latest: bool) -> list[Path]:
    root = Path(ckpt_dir)
    resolved: list[Path] = []
    missing: list[int] = []

    for epoch in epochs:
        name = f"checkpoint_epoch_{epoch:04d}.pt"
        path = root / name
        if path.is_file():
            resolved.append(path)
        else:
            missing.append(epoch)

    if include_latest:
        latest = root / "checkpoint_latest.pt"
        if latest.is_file():
            resolved.append(latest)

    if missing:
        missing_text = ", ".join(str(epoch) for epoch in missing)
        raise FileNotFoundError(f"Requested epoch checkpoint(s) not found in {ckpt_dir}: {missing_text}")

    return _sort_checkpoints(resolved)


def _select_checkpoints(
    ckpt_dir: str,
    include_latest: bool,
    checkpoint_names: list[str],
    epochs: list[int],
) -> list[Path]:
    if checkpoint_names and epochs:
        raise ValueError("Use either --checkpoint_names or --epochs, not both.")
    if checkpoint_names:
        return _resolve_named_checkpoints(ckpt_dir, checkpoint_names)
    if epochs:
        return _resolve_epoch_checkpoints(ckpt_dir, epochs, include_latest=include_latest)
    return _discover_all_checkpoints(ckpt_dir, include_latest=include_latest)


def _write_csv(rows: list[dict], out_path: str):
    fieldnames = [
        "rank",
        "epoch",
        "checkpoint",
        "split",
        "generator_used",
        "fid",
        "kid_mean",
        "kid_std",
        "num_real",
        "num_fake",
        "kid_subset_size",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and rank GAN checkpoints by FID/KID.")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing checkpoint files")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing packed dataset")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Real-data split used for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_real", type=int, default=2000)
    p.add_argument("--num_fake", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", help="Use G_ema if present in checkpoint")
    p.add_argument("--no_ema", action="store_true", help="Force using raw G even if G_ema exists")
    p.add_argument("--kid_subset_size", type=int, default=1000)
    p.add_argument("--sort_by", type=str, default="fid", choices=["fid", "kid_mean"], help="Metric used for ranking (lower is better)")
    p.add_argument("--top_k", type=int, default=5, help="How many top checkpoints to print in the summary")
    p.add_argument("--max_checkpoints", type=int, default=0, help="Limit how many checkpoints to evaluate after filtering (0 = all)")
    p.add_argument("--include_latest", action="store_true", help="Also include checkpoint_latest.pt if present")
    p.add_argument("--epochs", type=int, nargs="+", default=None, help="Only evaluate the specified checkpoint epochs, for example: --epochs 50 80 100")
    p.add_argument("--checkpoint_names", type=str, nargs="+", default=None, help="Only evaluate the specified checkpoint filenames, for example: --checkpoint_names checkpoint_epoch_0050.pt checkpoint_latest.pt")
    p.add_argument("--csv_out", type=str, default="", help="Optional CSV file to save the full ranking")
    return p.parse_args()


def main():
    args = parse_args()
    from eval_fid import compute_fid_kid, get_device

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("--train_ratio must be in (0,1)")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train_ratio + --val_ratio must be < 1")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.max_checkpoints < 0:
        raise ValueError("--max_checkpoints must be >= 0")

    use_ema = bool(args.use_ema) and (not args.no_ema)
    checkpoints = _select_checkpoints(
        ckpt_dir=args.ckpt_dir,
        include_latest=args.include_latest,
        checkpoint_names=args.checkpoint_names or [],
        epochs=args.epochs or [],
    )
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[:args.max_checkpoints]

    print("Device:", get_device())
    print(f"Evaluation split: {args.split}")
    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate in {args.ckpt_dir}")
    print(f"Ranking metric: {args.sort_by} (lower is better)")

    results: list[dict] = []
    for idx, ckpt_path in enumerate(checkpoints, start=1):
        print(f"\n[{idx}/{len(checkpoints)}] Evaluating: {ckpt_path.name}")
        out = compute_fid_kid(
            ckpt_path=str(ckpt_path),
            data_dir=args.data_dir,
            split=args.split,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_real=args.num_real,
            num_fake=args.num_fake,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            use_ema=use_ema,
            kid_subset_size=args.kid_subset_size,
        )
        out["checkpoint"] = ckpt_path.name
        out["epoch"] = _parse_epoch(ckpt_path)
        results.append(out)
        print(f"  FID={out['fid']:.4f} | KID={out['kid_mean']:.6f} +/- {out['kid_std']:.6f}")

    sort_keys = {
        "fid": lambda row: (row["fid"], row["kid_mean"], row["epoch"] if row["epoch"] is not None else 10**9),
        "kid_mean": lambda row: (row["kid_mean"], row["fid"], row["epoch"] if row["epoch"] is not None else 10**9),
    }
    ranked = sorted(results, key=sort_keys[args.sort_by])

    print("\n=== Ranking ===")
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        epoch_str = "latest" if row["epoch"] is None else str(row["epoch"])
        print(
            f"{rank:>2}. split={row['split']:>4} | "
            f"epoch={epoch_str:>6} | "
            f"FID={row['fid']:.4f} | "
            f"KID={row['kid_mean']:.6f} +/- {row['kid_std']:.6f} | "
            f"{row['checkpoint']}"
        )

    best = ranked[0]
    best_epoch = "latest" if best["epoch"] is None else best["epoch"]
    print("\nBest checkpoint:")
    print(
        f"split={best['split']} | "
        f"epoch={best_epoch} | "
        f"FID={best['fid']:.4f} | "
        f"KID={best['kid_mean']:.6f} +/- {best['kid_std']:.6f} | "
        f"{best['checkpoint']}"
    )

    top_k = min(args.top_k, len(ranked))
    print(f"\nTop {top_k}:")
    for row in ranked[:top_k]:
        print(f"- {row['checkpoint']}: split={row['split']}, FID={row['fid']:.4f}, KID={row['kid_mean']:.6f}")

    if args.csv_out.strip():
        _write_csv(ranked, args.csv_out)
        print(f"\nSaved CSV ranking to: {args.csv_out}")


if __name__ == "__main__":
    main()

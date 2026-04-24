
import argparse
import os
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from dataset import BraTSSliceDataset
from models_dcgan import DCGANGenerator

_KID_BUFFER_WARNING_RE = (
    r"Metric `Kernel Inception Distance` will save all extracted features in buffer\..*"
)

# TorchMetrics emits this informational warning every time KID is constructed.
# Keep the filter narrow so unrelated UserWarnings still surface normally.
warnings.filterwarnings(
    "ignore",
    message=_KID_BUFFER_WARNING_RE,
    category=UserWarning,
    module=r"torchmetrics\.utilities\.prints",
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _infer_model_kwargs_from_ckpt(state: dict) -> Dict:
    """
    Reconstruct DCGANGenerator kwargs from checkpoint.
    Tries top-level keys first, then state["config"] if present.
    """
    cfg = state.get("config", {}) or {}

    image_size = int(state.get("image_size", cfg.get("image_size", 256)))
    z_dim = int(state.get("z_dim", cfg.get("z_dim", 128)))

    ngf = state.get("ngf", None)
    if ngf is None:
        ngf = cfg.get("ngf", 64)
    ngf = int(ngf)

    in_channels = int(cfg.get("in_channels", 1))
    return dict(image_size=image_size, z_dim=z_dim, ngf=ngf, out_channels=in_channels)


def _load_generator_from_ckpt(
    ckpt_path: str, device: str, use_ema: bool
) -> Tuple[nn.Module, int, int, str]:
    state = torch.load(ckpt_path, map_location=device)
    model_kwargs = _infer_model_kwargs_from_ckpt(state)

    image_size = int(model_kwargs["image_size"])
    z_dim = int(model_kwargs["z_dim"])

    G = DCGANGenerator(**model_kwargs).to(device)

    chosen = "G"
    if use_ema and (state.get("G_ema") is not None):
        G.load_state_dict(state["G_ema"])
        chosen = "G_ema"
    else:
        G.load_state_dict(state["G"])

    G.eval()
    return G, image_size, z_dim, chosen


def _to_3ch_0_1(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W) in [-1,1] -> (B,3,H,W) in [0,1]
    TorchMetrics FID/KID with normalize=True expects float in [0,1].
    """
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def _build_kid_metric(kid_subset_size: int, device: str) -> KernelInceptionDistance:
    return KernelInceptionDistance(
        subset_size=kid_subset_size,
        normalize=True,
    ).to(device)


def _build_real_dataloader(
    data_dir: str,
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = BraTSSliceDataset(
        data_dir,
        split=split,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def _update_metrics_from_real_loader(
    dl: DataLoader,
    num_samples: int,
    fid: FrechetInceptionDistance,
    kid: KernelInceptionDistance,
    *,
    metric_real_flag: bool,
    device: str,
) -> int:
    seen = 0
    for x in dl:
        remaining = num_samples - seen
        if remaining <= 0:
            break

        x = x[:remaining].to(device, non_blocking=True)
        x = _to_3ch_0_1(x)
        fid.update(x, real=metric_real_flag)
        kid.update(x, real=metric_real_flag)
        seen += x.size(0)

    return seen


@torch.no_grad()
def compute_fid_kid(
    ckpt_path: str,
    data_dir: str,
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    num_real: int,
    num_fake: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_ema: bool,
    kid_subset_size: int,
) -> Dict:
    device = get_device()

    dl_real = _build_real_dataloader(
        data_dir=data_dir,
        split=split,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    G, image_size, z_dim, chosen = _load_generator_from_ckpt(ckpt_path, device, use_ema)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid_subset_size = int(min(kid_subset_size, num_real, num_fake))
    kid = _build_kid_metric(kid_subset_size, device)

    real_seen = _update_metrics_from_real_loader(
        dl_real,
        num_real,
        fid,
        kid,
        metric_real_flag=True,
        device=device,
    )

    seen = 0
    while seen < num_fake:
        bsz = min(batch_size, num_fake - seen)
        z = torch.randn(bsz, z_dim, 1, 1, device=device)
        fake = G(z)
        fake = _to_3ch_0_1(fake)
        fid.update(fake, real=False)
        kid.update(fake, real=False)
        seen += bsz

    fid_score = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()

    return {
        "ckpt": ckpt_path,
        "split": split,
        "generator_used": chosen,
        "image_size": image_size,
        "z_dim": z_dim,
        "num_real": real_seen,
        "num_fake": num_fake,
        "batch_size": batch_size,
        "fid": fid_score,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "kid_subset_size": kid_subset_size,
    }


@torch.no_grad()
def compute_fid_kid_real_vs_real(
    data_dir: str,
    split_a: str,
    split_b: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    num_a: int,
    num_b: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    kid_subset_size: int,
) -> Dict:
    device = get_device()

    dl_a = _build_real_dataloader(
        data_dir=data_dir,
        split=split_a,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dl_b = _build_real_dataloader(
        data_dir=data_dir,
        split=split_b,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid_subset_size = int(min(kid_subset_size, num_a, num_b))
    kid = _build_kid_metric(kid_subset_size, device)

    seen_a = _update_metrics_from_real_loader(
        dl_a,
        num_a,
        fid,
        kid,
        metric_real_flag=True,
        device=device,
    )
    seen_b = _update_metrics_from_real_loader(
        dl_b,
        num_b,
        fid,
        kid,
        metric_real_flag=False,
        device=device,
    )

    fid_score = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()

    return {
        "split_a": split_a,
        "split_b": split_b,
        "num_a": seen_a,
        "num_b": seen_b,
        "batch_size": batch_size,
        "fid": fid_score,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "kid_subset_size": kid_subset_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt)")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing packed dataset")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Real-data split used for evaluation")
    p.add_argument(
        "--real_vs_real",
        action="store_true",
        help="Compare two real dataset splits directly for a baseline instead of using a generator checkpoint",
    )
    p.add_argument("--split_a", type=str, default="train", choices=["train", "val", "test"], help="First real split for --real_vs_real")
    p.add_argument("--split_b", type=str, default="test", choices=["train", "val", "test"], help="Second real split for --real_vs_real")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_real", type=int, default=2000)
    p.add_argument("--num_fake", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", help="Use G_ema if present in checkpoint")
    p.add_argument("--no_ema", action="store_true", help="Force using raw G even if G_ema exists")
    p.add_argument("--kid_subset_size", type=int, default=1000, help="Subset size used by TorchMetrics KID")

    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("--train_ratio must be in (0,1)")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train_ratio + --val_ratio must be < 1")
    if args.real_vs_real and args.split_a == args.split_b:
        raise ValueError("--split_a and --split_b must be different for --real_vs_real")
    if (not args.real_vs_real) and (not args.ckpt):
        raise ValueError("--ckpt is required unless --real_vs_real is set")
    if args.ckpt and (not os.path.exists(args.ckpt)):
        raise FileNotFoundError(args.ckpt)

    use_ema = bool(args.use_ema) and (not args.no_ema)

    print("Device:", get_device())
    if args.real_vs_real:
        print("Mode: real-vs-real baseline")
        print(f"Split A: {args.split_a}")
        print(f"Split B: {args.split_b}")

        out = compute_fid_kid_real_vs_real(
            data_dir=args.data_dir,
            split_a=args.split_a,
            split_b=args.split_b,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_a=args.num_real,
            num_b=args.num_fake,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            kid_subset_size=args.kid_subset_size,
        )

        print(f"Split A used: {out['split_a']} | samples: {out['num_a']}")
        print(f"Split B used: {out['split_b']} | samples: {out['num_b']}")
        print(f"KID subset_size: {out['kid_subset_size']}")
        print("\n=== TorchMetrics Inception-v3 (ImageNet) ===")
        print(f"FID: {out['fid']:.4f}")
        print(f"KID: {out['kid_mean']:.6f} +/- {out['kid_std']:.6f}")
    else:
        print("Mode: generator-vs-real")
        print("Checkpoint:", args.ckpt)
        print("Split:", args.split)

        out = compute_fid_kid(
            ckpt_path=args.ckpt,
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

        print(f"Generator used: {out['generator_used']}")
        print(f"Evaluation split: {out['split']}")
        print(f"Image size: {out['image_size']} | z_dim: {out['z_dim']}")
        print(f"Real samples used: {out['num_real']} | Fake samples used: {out['num_fake']}")
        print(f"KID subset_size: {out['kid_subset_size']}")
        print("\n=== TorchMetrics Inception-v3 (ImageNet) ===")
        print(f"FID: {out['fid']:.4f}")
        print(f"KID: {out['kid_mean']:.6f} +/- {out['kid_std']:.6f}")


if __name__ == "__main__":
    main()

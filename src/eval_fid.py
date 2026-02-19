# eval_fid_kid.py
#
# Clean evaluation script:
# - Uses ONLY TorchMetrics Inception-v3 (ImageNet) features
# - Reports BOTH FID and KID
# - Works with your packed .npy dataset + DCGANGenerator checkpoints
# - Supports EMA generator via --use_ema
#
# Requirements:
#   pip install torchmetrics[image] torch-fidelity
#
# Usage:
#   python src\eval_fid_kid.py --data_dir data\preprocessed_slices_64 --ckpt runs\dcgan_64\checkpoint_latest.pt --num_real 2000 --num_fake 2000 --batch_size 32 --use_ema
#

import os
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from dataset import BraTSSliceDataset
from models_dcgan import DCGANGenerator


# -------------------------
# helpers
# -------------------------
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

    # out_channels is 1 for your MRI slices
    in_channels = cfg.get("in_channels", 1)
    in_channels = int(in_channels)

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
        chosen = "G"

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


# -------------------------
# main evaluation
# -------------------------
@torch.no_grad()
def compute_fid_kid(
    ckpt_path: str,
    data_dir: str,
    num_real: int,
    num_fake: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_ema: bool,
    kid_subset_size: int,
) -> Dict:
    device = get_device()

    # dataset (test split)
    ds_real = BraTSSliceDataset(data_dir, split="test")
    dl_real = DataLoader(
        ds_real,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # generator
    G, image_size, z_dim, chosen = _load_generator_from_ckpt(ckpt_path, device, use_ema)

    # metrics
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # KID buffers features; subset_size must be <= both real and fake counts used
    kid_subset_size = int(min(kid_subset_size, num_real, num_fake))
    kid = KernelInceptionDistance(subset_size=kid_subset_size, normalize=True).to(device)

    # ---- real updates ----
    seen = 0
    for x in dl_real:
        x = x.to(device, non_blocking=True)
        x = _to_3ch_0_1(x)
        fid.update(x, real=True)
        kid.update(x, real=True)
        seen += x.size(0)
        if seen >= num_real:
            break

    # ---- fake updates ----
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
    kid_mean = float(kid_mean.item())
    kid_std = float(kid_std.item())

    return {
        "ckpt": ckpt_path,
        "generator_used": chosen,
        "image_size": image_size,
        "z_dim": z_dim,
        "num_real": num_real,
        "num_fake": num_fake,
        "batch_size": batch_size,
        "fid": fid_score,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "kid_subset_size": kid_subset_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing packed dataset")
    p.add_argument("--num_real", type=int, default=2000)
    p.add_argument("--num_fake", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)

    # performance knobs
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true", default=True)

    # EMA
    p.add_argument("--use_ema", action="store_true", help="Use G_ema if present in checkpoint")
    p.add_argument("--no_ema", action="store_true", help="Force using raw G even if G_ema exists")

    # KID
    p.add_argument("--kid_subset_size", type=int, default=1000, help="Subset size used by TorchMetrics KID")

    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)

    use_ema = bool(args.use_ema) and (not args.no_ema)

    print("Device:", get_device())
    print("Checkpoint:", args.ckpt)

    out = compute_fid_kid(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        num_real=args.num_real,
        num_fake=args.num_fake,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_ema=use_ema,
        kid_subset_size=args.kid_subset_size,
    )

    print(f"Generator used: {out['generator_used']}")
    print(f"Image size: {out['image_size']} | z_dim: {out['z_dim']}")
    print(f"Real samples used: {out['num_real']} | Fake samples used: {out['num_fake']}")
    print(f"KID subset_size: {out['kid_subset_size']}")
    print("\n=== TorchMetrics Inception-v3 (ImageNet) ===")
    print(f"FID: {out['fid']:.4f}")
    print(f"KID: {out['kid_mean']:.6f} ± {out['kid_std']:.6f}")
    print("\nNote: Inception-based metrics are imperfect for MRIs; treat as relative metrics only.")


if __name__ == "__main__":
    main()

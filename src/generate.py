import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image, make_grid

from models_dcgan import DCGANGenerator


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _infer_model_kwargs_from_ckpt(state: dict):
    """
    Reconstruct DCGANGenerator kwargs from checkpoint.
    Works for both DCGAN and WGAN-GP in this project because both use DCGANGenerator.

    Tries:
      - top-level keys: image_size, z_dim, ngf
      - state["config"] fallback: image_size, z_dim, ngf, in_channels
    """
    cfg = state.get("config", {}) or {}

    image_size = int(state.get("image_size", cfg.get("image_size", 256)))
    z_dim = int(state.get("z_dim", cfg.get("z_dim", 128)))

    ngf = state.get("ngf", None)
    if ngf is None:
        ngf = cfg.get("ngf", 64)
    ngf = int(ngf)

    # Your training scripts treat this as 1 for MRI; stored in config in your code
    in_channels = int(cfg.get("in_channels", 1))

    return dict(image_size=image_size, z_dim=z_dim, ngf=ngf, out_channels=in_channels)

def save_grid_exact_px(grid_tensor: torch.Tensor, out_path: str, grid_px: int):
    """
    grid_tensor: (C,H,W) in [0,1]
    Saves image as EXACTLY grid_px x grid_px pixels.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grid_np = grid_tensor.detach().cpu().permute(1, 2, 0).numpy()  # (H,W,C)
    if grid_np.shape[2] == 1:
        grid_np = grid_np[:, :, 0]  # (H,W) for grayscale imshow

    # Pick a DPI and derive inches so pixels match exactly.
    # Using 100 keeps things simple and stable.
    dpi = 100
    fig = plt.figure(figsize=(grid_px / dpi, grid_px / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])  # fill canvas
    ax.axis("off")

    if grid_np.ndim == 2:
        ax.imshow(grid_np, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        ax.imshow(grid_np, vmin=0.0, vmax=1.0, interpolation="nearest")

    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic brain MRI slices from a GAN checkpoint (DCGAN or WGAN-GP)."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt (DCGAN or WGAN-GP)")
    p.add_argument("--out_dir", type=str, default="runs/generated", help="Output directory")
    p.add_argument("--num", type=int, default=256, help="Number of images to generate")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # EMA control
    p.add_argument("--use_ema", action="store_true", help="Use G_ema if available (recommended)")
    p.add_argument("--no_ema", action="store_true", help="Force using raw G even if G_ema exists")

    # Output modes
    p.add_argument("--save_grid", action="store_true", help="Save a single grid image")
    p.add_argument("--grid_nrow", type=int, default=8, help="Grid columns when --save_grid is set")
    p.add_argument("--grid_px", type=int, default=1600, help="Pixel size of saved grid image (saved as grid_px x grid_px). Default: 1600")

    p.add_argument("--save_individual", action="store_true", help="Save individual PNG files per sample")
    p.add_argument("--save_npy", action="store_true", help="Also save packed samples as a single .npy (N,H,W)")

    # Optional: tag for filenames (e.g. dcgan / wgangp)
    p.add_argument("--tag", type=str, default="", help="Optional tag added to output filenames (e.g. dcgan_64)")

    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    state = torch.load(args.ckpt, map_location=device)

    model_kwargs = _infer_model_kwargs_from_ckpt(state)
    image_size = model_kwargs["image_size"]
    z_dim = model_kwargs["z_dim"]

    G = DCGANGenerator(**model_kwargs).to(device)

    use_ema = bool(args.use_ema) and (not args.no_ema)
    if use_ema and (state.get("G_ema") is not None):
        G.load_state_dict(state["G_ema"])
        which = "G_ema"
    else:
        G.load_state_dict(state["G"])
        which = "G"

    G.eval()

    # default behaviour: if nothing specified, save grid + individual (beginner-friendly)
    if (not args.save_grid) and (not args.save_individual) and (not args.save_npy):
        args.save_grid = True
        args.save_individual = True

    tag = args.tag.strip()
    tag_part = f"_{tag}" if tag else ""
    meta_part = f"{tag_part}_{which}_{image_size}"

    all_samples = []  # for npy saving
    remaining = args.num
    idx_global = 0

    while remaining > 0:
        bsz = min(args.batch_size, remaining)
        z = torch.randn(bsz, z_dim, 1, 1, device=device)
        x = G(z)  # (B,1,H,W) in [-1,1]

        # map to [0,1] for PNG
        x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)

        if args.save_individual:
            for i in range(bsz):
                out_path = os.path.join(args.out_dir, f"sample{tag_part}_{idx_global:06d}.png")
                save_image(x01[i], out_path)
                idx_global += 1
        else:
            idx_global += bsz

        if args.save_npy:
            # store as (B,H,W) float32 in [-1,1]
            all_samples.append(x.squeeze(1).detach().cpu().numpy().astype(np.float32))

        remaining -= bsz

    if args.save_grid:
        # Make a nice square-ish grid (up to nrow*nrow images)
        n_grid = min(args.num, args.grid_nrow * args.grid_nrow)
        z = torch.randn(n_grid, z_dim, 1, 1, device=device)
        x = G(z)
        x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        grid = make_grid(x01, nrow=args.grid_nrow, padding=2)
        grid_path = os.path.join(args.out_dir, f"grid{meta_part}_{args.grid_px}px.png")
        save_grid_exact_px(grid, grid_path, grid_px=args.grid_px)

    if args.save_npy:
        arr = np.concatenate(all_samples, axis=0)  # (N,H,W)
        npy_path = os.path.join(args.out_dir, f"samples{meta_part}.npy")
        np.save(npy_path, arr)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Generator used: {which}")
    print(f"Image size: {image_size}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

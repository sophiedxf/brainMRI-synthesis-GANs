import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image

from dataset import BraTSSliceDataset
from models_dcgan import DCGANGenerator, DCGANDiscriminator
from utils_training import set_seed, save_sample_grid, save_checkpoint


# =========================
# CONFIG (edit defaults here)
# =========================
CONFIG = {
    # Data / run
    "data_dir": "data/preprocessed_slices_64",
    "out_dir": "runs/dcgan_64",
    "image_size": 64,              # 64 | 128 | 256
    "seed": 42,

    # Dataloader
    "num_workers": 8,
    "pin_memory": True,

    # Model
    "z_dim": 128,
    "ngf": 64,
    "ndf": 64,
    "in_channels": 1,

    # Optimisation
    "epochs": 100,
    "batch_size": 128,
    "lrG": 4e-4,   # TTUR
    "lrD": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,

    # AMP (recommended for DCGAN)
    "use_amp": True,

    # EMA (for cleaner samples / more stable eval)
    "ema_enabled": True,
    "ema_beta": 0.999,         # 0.999 for 64/128; try 0.9995 for 256
    "ema_start_epoch": 1,      # start updating EMA from this epoch

    # Logging / saving cadence
    "sample_grid_n": 16,
    "sample_grid_nrow": 4,
    "save_samples_every": 5,
    "save_ckpt_every": 10,
    "save_progress_every": 1,
    "progress_use_ema": True,
    "progress_gif_filename": "generator_progression.gif",

    # Output files
    "loss_curve_filename": "loss_curves.png",
    "checkpoint_filename": "checkpoint_epoch_{epoch:04d}.pt",
    "checkpoint_latest_filename": "checkpoint_latest.pt",
}

VALID_IMAGE_SIZES = {64, 128, 256}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_loss_curves(out_dir: str, epochs: list[int], loss_d: list[float], loss_g: list[float], filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(epochs, loss_d, label="Discriminator loss")
    plt.plot(epochs, loss_g, label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def ema_init(ema_model: torch.nn.Module, model: torch.nn.Module):
    """Initialise EMA weights to exactly match model weights."""
    ema_model.load_state_dict(model.state_dict(), strict=True)


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, beta: float):
    """
    Update EMA weights from model weights:
      ema = beta * ema + (1-beta) * model
    Buffers (e.g., BatchNorm running stats) are copied directly from model.
    """
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(beta).add_(p.data, alpha=1.0 - beta)

    # Important for DCGAN because BatchNorm uses buffers (running_mean/var)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def save_progress_animation_gif(frames_dir: str, out_path: str, duration_ms: int = 800):
    frame_paths = sorted(Path(frames_dir).glob("epoch_*.png"))
    if len(frame_paths) == 0:
        return

    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    first, rest = frames[0], frames[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()


def parse_args():
    p = argparse.ArgumentParser()

    # Data / run
    p.add_argument("--data_dir", type=str, default=CONFIG["data_dir"])
    p.add_argument("--out_dir", type=str, default=CONFIG["out_dir"])
    p.add_argument("--image_size", type=int, default=CONFIG["image_size"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])

    # Model
    p.add_argument("--z_dim", type=int, default=CONFIG["z_dim"])
    p.add_argument("--ngf", type=int, default=CONFIG["ngf"])
    p.add_argument("--ndf", type=int, default=CONFIG["ndf"])

    # Optimisation
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--lrG", type=float, default=CONFIG["lrG"])
    p.add_argument("--lrD", type=float, default=CONFIG["lrD"])
    p.add_argument("--beta1", type=float, default=CONFIG["beta1"])
    p.add_argument("--beta2", type=float, default=CONFIG["beta2"])

    # Dataloader
    p.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--pin_memory", action="store_true", default=CONFIG["pin_memory"])

    # AMP
    p.add_argument("--use_amp", action="store_true", default=CONFIG["use_amp"])
    p.add_argument("--no_amp", action="store_true", default=False, help="Disable AMP regardless of --use_amp")

    # EMA
    p.add_argument("--ema", action="store_true", default=CONFIG["ema_enabled"], help="Enable EMA for generator")
    p.add_argument("--ema_beta", type=float, default=CONFIG["ema_beta"])
    p.add_argument("--ema_start_epoch", type=int, default=CONFIG["ema_start_epoch"])

    # Saving cadence
    p.add_argument("--save_samples_every", type=int, default=CONFIG["save_samples_every"])
    p.add_argument("--save_ckpt_every", type=int, default=CONFIG["save_ckpt_every"])
    p.add_argument("--sample_grid_n", type=int, default=CONFIG["sample_grid_n"])
    p.add_argument("--sample_grid_nrow", type=int, default=CONFIG["sample_grid_nrow"])
    p.add_argument("--save_progress_every", type=int, default=CONFIG["save_progress_every"])
    p.add_argument("--progress_use_ema", action="store_true", default=CONFIG["progress_use_ema"])
    p.add_argument("--no_progress_use_ema", action="store_true", default=False, help="Use raw G instead of EMA for progression animation")

    # Resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint_epoch_XXXX.pt to resume from")

    return p.parse_args()


def _warn_if_mismatch(args, ckpt: dict):
    def _cmp(key: str, current_val, ckpt_val):
        if ckpt_val is None:
            return
        if current_val != ckpt_val:
            print(f"[WARN] Mismatch for '{key}': checkpoint={ckpt_val} vs current={current_val}")

    _cmp("image_size", args.image_size, ckpt.get("image_size", None))
    _cmp("z_dim", args.z_dim, ckpt.get("z_dim", None))
    _cmp("ngf", args.ngf, ckpt.get("ngf", None))
    _cmp("ndf", args.ndf, ckpt.get("ndf", None))


def train(args):
    if args.image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"--image_size must be one of {sorted(VALID_IMAGE_SIZES)}")
    if args.save_progress_every < 1:
        raise ValueError("--save_progress_every must be >= 1")

    device = get_device()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    use_amp = (args.use_amp and (not args.no_amp) and device == "cuda")
    print("AMP enabled:", use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    progress_frames_dir = os.path.join(args.out_dir, "progress_frames")
    progress_gif_path = os.path.join(args.out_dir, CONFIG["progress_gif_filename"])

    # Dataset / loader
    ds = BraTSSliceDataset(args.data_dir, split="train", seed=args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    # Models
    G = DCGANGenerator(
        image_size=args.image_size,
        z_dim=args.z_dim,
        ngf=args.ngf,
        out_channels=CONFIG["in_channels"],
    ).to(device)

    D = DCGANDiscriminator(
        image_size=args.image_size,
        ndf=args.ndf,
        in_channels=CONFIG["in_channels"],
    ).to(device)

    # EMA model (shadow copy of G)
    G_ema = None
    if args.ema:
        G_ema = DCGANGenerator(
            image_size=args.image_size,
            z_dim=args.z_dim,
            ngf=args.ngf,
            out_channels=CONFIG["in_channels"],
        ).to(device)
        G_ema.eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
        ema_init(G_ema, G)

    # Loss: logits + BCEWithLogitsLoss (D must NOT have Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    optG = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    optD = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fixed_noise = torch.randn(args.sample_grid_n, args.z_dim, 1, 1, device=device)

    epoch_list: list[int] = []
    lossD_hist: list[float] = []
    lossG_hist: list[float] = []
    start_epoch = 1

    # -----------------------
    # Resume from checkpoint
    # -----------------------
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)
        _warn_if_mismatch(args, ckpt)

        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])

        if "optG" in ckpt:
            optG.load_state_dict(ckpt["optG"])
        if "optD" in ckpt:
            optD.load_state_dict(ckpt["optD"])

        if "lossD_hist" in ckpt:
            lossD_hist = list(ckpt["lossD_hist"])
        if "lossG_hist" in ckpt:
            lossG_hist = list(ckpt["lossG_hist"])

        # Restore EMA if present; otherwise re-init from current G
        if args.ema and G_ema is not None:
            if ckpt.get("G_ema") is not None:
                G_ema.load_state_dict(ckpt["G_ema"])
                print("Loaded G_ema from checkpoint.")
            else:
                ema_init(G_ema, G)
                print("Checkpoint has no G_ema; initialised EMA from current G.")

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = last_epoch + 1

        if len(lossD_hist) == len(lossG_hist) and len(lossD_hist) > 0:
            epoch_list = list(range(1, len(lossD_hist) + 1))

        print(f"Resuming from: {args.resume}")
        print(f"Checkpoint epoch: {last_epoch} -> starting from epoch {start_epoch}")

        if len(epoch_list) > 0:
            save_loss_curves(args.out_dir, epoch_list, lossD_hist, lossG_hist, CONFIG["loss_curve_filename"])

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(start_epoch, args.epochs + 1):
        start_t = time.time()
        G.train()
        D.train()

        t0 = time.time()

        lossD_running = 0.0
        lossG_running = 0.0
        n_batches = 0

        for real in dl:
            real = real.to(device, non_blocking=True)
            bsz = real.size(0)

            # one-sided label smoothing
            labels_real = torch.full((bsz, 1), 0.9, device=device)
            labels_fake = torch.zeros((bsz, 1), device=device)

            # -------------------
            # Train Discriminator
            # -------------------
            optD.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_real = D(real)
                lossD_real = criterion(out_real, labels_real)

                noise = torch.randn(bsz, args.z_dim, 1, 1, device=device)
                fake = G(noise).detach()
                out_fake = D(fake)
                lossD_fake = criterion(out_fake, labels_fake)

                lossD = lossD_real + lossD_fake

            scaler.scale(lossD).backward()
            scaler.step(optD)

            # --------------
            # Train Generator
            # --------------
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                noise = torch.randn(bsz, args.z_dim, 1, 1, device=device)
                fake = G(noise)
                out_fake = D(fake)
                lossG = criterion(out_fake, labels_real)

            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            # EMA update (after G update)
            if args.ema and (G_ema is not None) and (epoch >= args.ema_start_epoch):
                ema_update(G_ema, G, beta=float(args.ema_beta))

            lossD_running += float(lossD.item())
            lossG_running += float(lossG.item())
            n_batches += 1

        t_train = time.time()

        lossD_epoch = lossD_running / max(1, n_batches)
        lossG_epoch = lossG_running / max(1, n_batches)

        epoch_list.append(epoch)
        lossD_hist.append(lossD_epoch)
        lossG_hist.append(lossG_epoch)

        # Save loss curves every 10 epochs (and at start_epoch)
        if (epoch % 10 == 0) or (epoch == start_epoch):
            save_loss_curves(args.out_dir, epoch_list, lossD_hist, lossG_hist, CONFIG["loss_curve_filename"])

        t_plot = time.time()

        if (epoch % args.save_progress_every) == 0:
            G_progress = G
            if (not args.no_progress_use_ema) and args.progress_use_ema and args.ema and (G_ema is not None):
                G_progress = G_ema
            G_progress.eval()
            with torch.no_grad():
                progress_samples = G_progress(fixed_noise).cpu()
            progress_frame_path = os.path.join(progress_frames_dir, f"epoch_{epoch:04d}.png")
            save_sample_grid(progress_samples, progress_frame_path, nrow=args.sample_grid_nrow)
            save_progress_animation_gif(progress_frames_dir, progress_gif_path)

        # Save samples every N epochs (use EMA generator if enabled)
        if epoch % args.save_samples_every == 0:
            G_sample = G_ema if (args.ema and G_ema is not None) else G
            G_sample.eval()
            with torch.no_grad():
                samples = G_sample(fixed_noise).cpu()
            tag = "ema" if (args.ema and G_ema is not None) else "raw"
            sample_path = os.path.join(args.out_dir, f"samples_{tag}_epoch_{epoch:04d}.png")
            save_sample_grid(samples, sample_path, nrow=args.sample_grid_nrow)

        # Save checkpoint every M epochs
        if epoch % args.save_ckpt_every == 0:
            payload = {
                "epoch": epoch,
                "image_size": args.image_size,
                "z_dim": args.z_dim,
                "ngf": args.ngf,
                "ndf": args.ndf,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "G_ema": (G_ema.state_dict() if (args.ema and G_ema is not None) else None),
                "ema_beta": float(args.ema_beta),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "lossD_hist": lossD_hist,
                "lossG_hist": lossG_hist,
                "config": vars(args),
            }

            ckpt_path = os.path.join(args.out_dir, CONFIG["checkpoint_filename"].format(epoch=epoch))
            save_checkpoint(payload, ckpt_path)

            # Update latest only on checkpoint epochs
            latest_path = os.path.join(args.out_dir, CONFIG["checkpoint_latest_filename"])
            save_checkpoint(payload, latest_path)

        t_ckpt = time.time()
        print(f"time breakdown: train={t_train - t0:.2f}s | plot={t_plot - t_train:.2f}s | ckpt={t_ckpt - t_plot:.2f}s")

        elapsed = time.time() - start_t
        print(
            f"[DCGAN {args.image_size}] Epoch {epoch:03d}/{args.epochs} | "
            f"lossD={lossD_epoch:.4f} | lossG={lossG_epoch:.4f} | "
            f"time={elapsed:.1f}s"
        )

    print("DCGAN training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)

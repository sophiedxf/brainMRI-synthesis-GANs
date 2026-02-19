# train_wgangp.py
import os
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dataset import BraTSSliceDataset
from models_dcgan import DCGANGenerator
from models_wgangp import WGANGPCritic, gradient_penalty
from utils_training import set_seed, save_sample_grid, save_checkpoint


# =========================
# CONFIG (edit defaults here)
# =========================
CONFIG = {
    # Data / run
    "data_dir": "data/preprocessed_slices_64",
    "out_dir": "runs/wgangp_64",
    "image_size": 64,              # 64 | 128 | 256
    "seed": 42,

    # Dataloader
    "num_workers": 2,
    "pin_memory": True,

    # Model
    "z_dim": 128,
    "ngf": 64,
    "ndf": 64,
    "in_channels": 1,

    # Optimisation (WGAN-GP)
    "epochs": 200,
    "batch_size": 64,
    "lr": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9,
    "n_critic": 3,          # (Change #2) default reduced from 5 -> 2
    "lambda_gp": 10.0,
    "gp_every": 2,          # (Change #1) compute GP every N critic steps (>=1)

    # AMP
    # For WGAN-GP + gradient penalty, it's usually safer to keep AMP OFF.
    "use_amp": False,

    # EMA (recommended for cleaner samples / stabler eval)
    "ema_enabled": True,
    "ema_beta": 0.999,
    "ema_start_epoch": 1,

    # Logging / saving cadence
    "sample_grid_n": 64,
    "sample_grid_nrow": 8,
    "save_samples_every": 10,
    "save_ckpt_every": 10,

    # Output files
    "loss_curve_filename": "loss_curves.png",
    "checkpoint_filename": "checkpoint_epoch_{epoch:04d}.pt",
    "checkpoint_latest_filename": "checkpoint_latest.pt",
}

VALID_IMAGE_SIZES = {64, 128, 256}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_loss_curves(out_dir: str, epochs: list[int], loss_c: list[float], loss_g: list[float], filename: str):
    """Saves a PNG plot of critic & generator losses vs epoch."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(epochs, loss_c, label="Critic loss")
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

    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


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
    p.add_argument("--lr", type=float, default=CONFIG["lr"])
    p.add_argument("--beta1", type=float, default=CONFIG["beta1"])
    p.add_argument("--beta2", type=float, default=CONFIG["beta2"])
    p.add_argument("--n_critic", type=int, default=CONFIG["n_critic"])      # (Change #2)
    p.add_argument("--lambda_gp", type=float, default=CONFIG["lambda_gp"])
    p.add_argument("--gp_every", type=int, default=CONFIG["gp_every"],      # (Change #1)
                   help="Compute gradient penalty every N critic steps (1 = every step)")

    # Dataloader
    p.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--pin_memory", action="store_true", default=CONFIG["pin_memory"])

    # AMP
    p.add_argument("--use_amp", action="store_true", default=CONFIG["use_amp"])
    p.add_argument("--no_amp", action="store_true", default=False)

    # EMA
    p.add_argument("--ema", action="store_true", default=CONFIG["ema_enabled"], help="Enable EMA for generator")
    p.add_argument("--ema_beta", type=float, default=CONFIG["ema_beta"])
    p.add_argument("--ema_start_epoch", type=int, default=CONFIG["ema_start_epoch"])

    # Saving cadence
    p.add_argument("--save_samples_every", type=int, default=CONFIG["save_samples_every"])
    p.add_argument("--save_ckpt_every", type=int, default=CONFIG["save_ckpt_every"])
    p.add_argument("--sample_grid_n", type=int, default=CONFIG["sample_grid_n"])
    p.add_argument("--sample_grid_nrow", type=int, default=CONFIG["sample_grid_nrow"])

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
    if args.gp_every < 1:
        raise ValueError("--gp_every must be >= 1")

    device = get_device()

    print("Device:", device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    use_amp = (args.use_amp and (not args.no_amp) and device == "cuda")
    print("AMP enabled:", use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Dataset / loader
    ds = BraTSSliceDataset(args.data_dir, split="train", seed=args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # (Change #3) pin_memory_device + robust worker settings
        pin_memory_device="cuda" if (args.pin_memory and torch.cuda.is_available()) else "",
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Models
    G = DCGANGenerator(
        image_size=args.image_size,
        z_dim=args.z_dim,
        ngf=args.ngf,
        out_channels=CONFIG["in_channels"],
    ).to(device)

    C = WGANGPCritic(
        image_size=args.image_size,
        ndf=args.ndf,
        in_channels=CONFIG["in_channels"],
    ).to(device)

    # EMA shadow generator
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

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optC = optim.Adam(C.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(args.sample_grid_n, args.z_dim, 1, 1, device=device)

    epoch_list: list[int] = []
    lossC_hist: list[float] = []
    lossG_hist: list[float] = []
    start_epoch = 1

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -----------------------
    # Resume from checkpoint
    # -----------------------
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)
        _warn_if_mismatch(args, ckpt)

        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])

        if "optG" in ckpt:
            optG.load_state_dict(ckpt["optG"])
        if "optC" in ckpt:
            optC.load_state_dict(ckpt["optC"])

        if "lossC_hist" in ckpt:
            lossC_hist = list(ckpt["lossC_hist"])
        if "lossG_hist" in ckpt:
            lossG_hist = list(ckpt["lossG_hist"])

        if args.ema and G_ema is not None:
            if ckpt.get("G_ema") is not None:
                G_ema.load_state_dict(ckpt["G_ema"])
                print("Loaded G_ema from checkpoint.")
            else:
                ema_init(G_ema, G)
                print("Checkpoint has no G_ema; initialised EMA from current G.")

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = last_epoch + 1

        if len(lossC_hist) == len(lossG_hist) and len(lossC_hist) > 0:
            epoch_list = list(range(1, len(lossC_hist) + 1))

        print(f"Resuming from: {args.resume}")
        print(f"Checkpoint epoch: {last_epoch} -> starting from epoch {start_epoch}")

        if len(epoch_list) > 0:
            save_loss_curves(args.out_dir, epoch_list, lossC_hist, lossG_hist, CONFIG["loss_curve_filename"])

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(start_epoch, args.epochs + 1):
        start_t = time.time()
        G.train()
        C.train()

        lossC_running = 0.0
        lossG_running = 0.0
        n_batches = 0

        for real in dl:
            real = real.to(device, non_blocking=True)
            bsz = real.size(0)

            # -----------------------
            # Train Critic n_critic times
            # -----------------------
            for c_step in range(args.n_critic):
                z = torch.randn(bsz, args.z_dim, 1, 1, device=device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake = G(z).detach()
                    C_real = C(real)
                    C_fake = C(fake)

                    # (Change #1) GP less frequently
                    if (c_step % args.gp_every) == 0:
                        gp = gradient_penalty(C, real, fake, device=device, lambda_gp=args.lambda_gp)
                    else:
                        gp = 0.0

                    lossC = (C_fake.mean() - C_real.mean()) + gp

                optC.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(lossC).backward()
                    scaler.step(optC)
                    scaler.update()
                else:
                    lossC.backward()
                    optC.step()

            # -----------------------
            # Train Generator
            # -----------------------
            z = torch.randn(bsz, args.z_dim, 1, 1, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = G(z)
                lossG = -C(fake).mean()

            optG.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(lossG).backward()
                scaler.step(optG)
                scaler.update()
            else:
                lossG.backward()
                optG.step()

            # EMA update (after generator update)
            if args.ema and (G_ema is not None) and (epoch >= args.ema_start_epoch):
                ema_update(G_ema, G, beta=float(args.ema_beta))

            lossC_running += float(lossC.item())
            lossG_running += float(lossG.item())
            n_batches += 1

        lossC_epoch = lossC_running / max(1, n_batches)
        lossG_epoch = lossG_running / max(1, n_batches)

        epoch_list.append(epoch)
        lossC_hist.append(lossC_epoch)
        lossG_hist.append(lossG_epoch)

        # Save loss curves every 10 epochs (and at start_epoch)
        if (epoch % 10 == 0) or (epoch == start_epoch):
            save_loss_curves(args.out_dir, epoch_list, lossC_hist, lossG_hist, CONFIG["loss_curve_filename"])

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
                "C": C.state_dict(),
                "G_ema": (G_ema.state_dict() if (args.ema and G_ema is not None) else None),
                "ema_beta": float(args.ema_beta),
                "optG": optG.state_dict(),
                "optC": optC.state_dict(),
                "lossC_hist": lossC_hist,
                "lossG_hist": lossG_hist,
                "config": vars(args),
            }

            ckpt_path = os.path.join(args.out_dir, CONFIG["checkpoint_filename"].format(epoch=epoch))
            save_checkpoint(payload, ckpt_path)

            latest_path = os.path.join(args.out_dir, CONFIG["checkpoint_latest_filename"])
            save_checkpoint(payload, latest_path)

        elapsed = time.time() - start_t
        print(
            f"[WGAN-GP {args.image_size}] Epoch {epoch:03d}/{args.epochs} | "
            f"lossC={lossC_epoch:.4f} | lossG={lossG_epoch:.4f} | "
            f"time={elapsed:.1f}s"
        )

    print("WGAN-GP training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)

import os
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    Make experiments more reproducible.
    Note: full determinism is difficult with GPU ops; this improves consistency.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# EMA utilities
# =========================
@torch.no_grad()
def init_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Create an EMA copy of a model.
    - Same architecture (deep copy)
    - Parameters are copied from model
    - Put in eval mode
    - EMA params do not require gradients

    Typical use:
        G_ema = init_ema_model(G)
    """
    import copy

    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    """
    Update EMA weights:
        ema = decay * ema + (1 - decay) * model

    Notes:
    - Call AFTER you update the live model weights (e.g. after optG.step()).
    - Use decay in [0,1). Typical: 0.999 to 0.9999.
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError("EMA decay must be in [0, 1).")

    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for k, v in ema_state.items():
        src = model_state[k]
        if torch.is_floating_point(v):
            v.mul_(decay).add_(src, alpha=(1.0 - decay))
        else:
            # For non-float buffers (e.g. num_batches_tracked), just copy
            ema_state[k] = src

    ema_model.load_state_dict(ema_state, strict=True)


@torch.no_grad()
def copy_to_ema(ema_model: torch.nn.Module, model: torch.nn.Module):
    """
    Hard copy model -> ema_model (useful at init or if you want to reset EMA).
    """
    ema_model.load_state_dict(model.state_dict(), strict=True)


# =========================
# Visualisation / checkpoint
# =========================
@torch.no_grad()
def save_sample_grid(samples: torch.Tensor, out_path: str, nrow: int = 8):
    """
    Save a grid of images to disk.

    Args:
        samples: Tensor (B, C, H, W) in [-1, 1]
        out_path: file path to save PNG
        nrow: number of images per row in the grid
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Map from [-1, 1] to [0, 1] for saving/viewing
    x = (samples + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    # Make a grid image (C, H_grid, W_grid)
    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=2)
    grid = grid.cpu()

    # Convert to numpy for matplotlib
    npimg = grid.numpy()

    plt.figure(figsize=(8, 8))
    plt.axis("off")

    # If grayscale (C=1), show as gray; else show as RGB
    if npimg.shape[0] == 1:
        plt.imshow(npimg[0], cmap="gray")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_checkpoint(state: dict, out_path: str):
    """
    Save training state (model weights, epoch, etc.)

    EMA integration:
    - If you include EMA in training, just add it to the state dict:
        state["G_ema"] = G_ema.state_dict()
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)


def load_checkpoint(path: str, device: str = "cuda"):
    """
    Load checkpoint to the specified device.
    """
    return torch.load(path, map_location=device)

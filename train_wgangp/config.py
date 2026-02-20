from dataclasses import dataclass

@dataclass
class Config:
    # Data (default)
    data_dir: str = "data/preprocessed_slices_256"
    modality: str = "t1ce"
    image_size: int = 256
    in_channels: int = 1

    # Train common
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True

    # Latent
    z_dim: int = 128
    ngf: int = 64
    ndf: int = 64

    # DCGAN hyperparams
    dcgan_batch_size: int = 16
    dcgan_epochs: int = 100
    dcgan_lr: float = 2e-4
    dcgan_beta1: float = 0.5
    dcgan_beta2: float = 0.999

    # WGAN-GP hyperparams
    wgangp_batch_size: int = 8
    wgangp_epochs: int = 100
    wgangp_lr: float = 1e-4
    wgangp_beta1: float = 0.0
    wgangp_beta2: float = 0.9
    n_critic: int = 5
    lambda_gp: float = 10.0

    # Logging/checkpoints
    sample_grid_n: int = 64
    save_every_epochs: int = 1

VALID_IMAGE_SIZES = {64, 128, 256}

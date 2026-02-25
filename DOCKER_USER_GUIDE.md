# 2D Brain MRI Slice Generation - Docker User Guide

This guide explains how to use the Docker images for this project **independently** (without using `docker-compose`).

## 1. Prerequisites & General Usage

- **Docker Desktop**: Must be installed and running.
- **Data Folders**: You must create `data/` and `runs/` in your current host directory, matching the underlying repository layout. Put your BraTS2023 raw data inside `data/raw/`.

Every module's script reads from `data/` and writes outputs (checkpoints, generated images, logs) to either `data/` or `runs/`. Because Docker runs an isolated environment, you **must use volume mounts** to link your host's folders into the container's `/app` folder.

### Device Support (CPU or GPU)

The code automatically detects whether a GPU is available. 
- To use a **GPU**, you must add `--gpus all` to your Docker run command. Ensure you have NVIDIA drivers installed (if on Windows/WSL2, Docker Desktop v4.31.0+ is recommended).
- To run on **CPU only** (which is supported but much slower, especially for training), simply omit the `--gpus all` flag from the command.

### General `docker run` Command Pattern
```bash
docker run --rm [ --gpus all ] \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/runs:/app/runs \
  <image_name> \
  [python_script] [arguments...]
```

* `--rm`: Automatically remove the container after it exits to keep your system clean.
* `--gpus all` *(optional)*: Passes your NVIDIA GPU hardware directly into the container. Omit this flag to use CPU instead.
* `-v ${PWD}/data:/app/data`: Mounts your local `./data` folder to the container's `/app/data` path. (On Windows CMD/PowerShell, replace `${PWD}` with `%cd%` or `$PWD` accordingly, or provide full absolute paths like `C:\path\to\data`).

---

## 2. Docker Image Catalog

These are the 5 independent images available for the modules:

* `ghcr.io/sydgep/docker-images/brats_preprocess:latest`
* `ghcr.io/sydgep/docker-images/brats_train_dcgan:latest`
* `ghcr.io/sydgep/docker-images/brats_train_wgangp:latest`
* `ghcr.io/sydgep/docker-images/brats_evaluate:latest`
* `ghcr.io/sydgep/docker-images/brats_generate:latest`

**Note:** If you run any of these images *without* any arguments, they will run a default example CMD (configured in the Dockerfile), but you can override the arguments by appending them to the inner `docker run` command.

---

## 3. Preprocessing (`preprocess`)

Converts 3D BraTS NIfTI volumes into a single packed `.npy` file of slices.

**Image**: `ghcr.io/sydgep/docker-images/brats_preprocess`

### Example Command (64x64, T2f, ~10k slices)
```bash
docker run --rm --gpus all \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  ghcr.io/sydgep/docker-images/brats_preprocess \
  python preprocess/preprocess.py \
    --raw_dir data/raw \
    --out_dir data/preprocessed_slices_64 \
    --modality t2f \
    --target_size 64 \
    --min_foreground 500 \
    --target_total_slices 10000 \
    --selection topk_foreground \
    --seed 42 \
    --save_png_samples
```

---

## 4. Train DCGAN (`train_dcgan`)

Trains the Deep Convolutional GAN model.

**Image**: `ghcr.io/sydgep/docker-images/brats_train_dcgan`

### Example Command (DCGAN 64x64)
```bash
docker run --rm --gpus all \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  ghcr.io/sydgep/docker-images/brats_train_dcgan \
  python train_dcgan/train_dcgan.py \
    --data_dir data/preprocessed_slices_64 \
    --out_dir runs/dcgan_64 \
    --image_size 64 \
    --batch_size 128 \
    --epochs 100 \
    --lrG 4e-4 \
    --lrD 2e-4 \
    --use_amp \
    --ema \
    --ema_beta 0.999
```

---

## 5. Train WGAN-GP (`train_wgangp`)

Trains the Wasserstein GAN with Gradient Penalty.

**Image**: `ghcr.io/sydgep/docker-images/brats_train_wgangp`

### Example Command (WGAN-GP 64x64, faster GP)
```bash
docker run --rm --gpus all \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  ghcr.io/sydgep/docker-images/brats_train_wgangp \
  python train_wgangp/train_wgangp.py \
    --data_dir data/preprocessed_slices_64 \
    --out_dir runs/wgangp_64 \
    --image_size 64 \
    --batch_size 64 \
    --epochs 100 \
    --n_critic 5 \
    --lambda_gp 10 \
    --gp_every 2 \
    --ema \
    --ema_beta 0.999
```

---

## 6. Evaluate FID/KID (`evaluate`)

Computes FID + KID using TorchMetrics Inception-v3 features.

**Image**: `ghcr.io/sydgep/docker-images/brats_evaluate`

### Example Command
To evaluate the latest checkpoint using 2000 real and 2000 fake samples (using EMA weights):
```bash
docker run -it --rm \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  ghcr.io/sydgep/docker-images/brats_evaluate:latest \
  python evaluate/eval_fid.py \
    --data_dir data/preprocessed_slices_64 \
    --ckpt runs/wgangp_64/checkpoint_latest.pt \
    --num_real 2000 \
    --num_fake 2000 \
    --batch_size 32 \
    --kid_subset_size 1000
```

---

## 7. Generate Images (`generate`)

Generates synthetic images from a checkpoint and optionally saves a grid PNG, individual PNGs, or a packed `.npy`.

**Image**: `ghcr.io/sydgep/docker-images/brats_generate`

### Example Command (1600x1600 grid)
```bash
docker run --rm --gpus all \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  ghcr.io/sydgep/docker-images/brats_generate \
  python generate/generate.py \
    --ckpt runs/wgangp_64/checkpoint_latest.pt \
    --out_dir runs/generated/wgangp_64 \
    --num 64 \
    --batch_size 64 \
    --save_grid \
    --grid_nrow 8 \
    --grid_px 1600 \
    --use_ema \
    --tag wgangp64
```

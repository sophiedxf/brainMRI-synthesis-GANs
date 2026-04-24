# 2D Brain MRI Generation with GANs

This project trains **DCGAN** and **WGAN-GP** models to generate **2D brain MRI slices** from the **BraTS 2023** dataset.

It includes:
- preprocessing from 3D NIfTI volumes to packed 2D slices
- patient-level train/val/test splitting
- DCGAN and WGAN-GP training
- FID/KID evaluation
- Nearest neighbour test for privacy audit
- image generation from trained checkpoints

Supported image sizes:
- `64 x 64`
- `128 x 128`
- `256 x 256`

Supported MRI modalities:
- `t1n`
- `t1c`
- `t2w`
- `t2f`

<p align="center">
  <img src="grid_G_ema_64_1600px.png" width="600" alt="Example generated MRI slices">
</p>

## Overview

The typical workflow is:

1. Put BraTS data under `data/raw/`
2. Run preprocessing to create packed slices and metadata
3. Train `DCGAN` or `WGAN-GP`
4. Evaluate checkpoints with `FID` and `KID`
5. Generate sample images from a trained model

## Setup

Recommended environment:

```bat
conda create -n dcgan_wgan python=3.10.11 -y
conda activate dcgan_wgan
pip install -r requirements_all.txt
```

The scripts use CUDA automatically if available. CPU also works, but training and evaluation will be much slower.

## Repository Layout

```text
data/
  raw/                     BraTS data goes here
  preprocessed_slices_64/  Created by preprocessing

preprocess/
  preprocess.py

train_dcgan/
  train_dcgan.py
  dataset.py
  models_dcgan.py

train_wgangp/
  train_wgangp.py
  dataset.py
  models_dcgan.py
  models_wgangp.py

evaluate_fid_kid/
  eval_fid.py
  rank_checkpoints.py
  Dockerfile

evaluate_privacy/
  privacy_audit.py
  Dockerfile

evaluate_progression_animation/
  make_progress_animation.py
  Dockerfile

generate/
  generate.py

runs/
  checkpoints, samples, plots, generated outputs
```

## Data

This repository does **not** include BraTS data.

Place the extracted BraTS folders under:

```text
data/raw/
```

The preprocessing script searches recursively. Expected BraTS modality suffixes are:
- `-t1n.nii.gz`
- `-t1c.nii.gz`
- `-t2w.nii.gz`
- `-t2f.nii.gz`

## Preprocessing

Preprocessing converts BraTS volumes into:
- a packed slice array: `*_packed.npy`
- a metadata file: `*_packed_metadata.npz`

The metadata stores the patient ID for each slice, which allows **patient-level splitting** during training and evaluation.

Example:

```bat
python preprocess\preprocess.py ^
  --raw_dir data\raw ^
  --out_dir data\preprocessed_slices_64 ^
  --modality t2f ^
  --target_size 64 ^
  --min_foreground 500 ^
  --target_total_slices 10000 ^
  --selection topk_foreground ^
  --seed 42 ^
  --save_png_samples
```

Good starting defaults:
- `--target_size 64`
- `--modality t2f`
- `--min_foreground 500`
- `--target_total_slices 10000`

## Train DCGAN

Example:

```bat
python train_dcgan\train_dcgan.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\dcgan_64 ^
  --image_size 64 ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --batch_size 128 ^
  --epochs 100 ^
  --lrG 4e-4 ^
  --lrD 2e-4 ^
  --beta1 0.5 ^
  --beta2 0.999 ^
  --use_amp ^
  --ema ^
  --ema_beta 0.999 ^
  --save_progress_every 1
```

Notes:
- `test_ratio` is the remainder: `1 - train_ratio - val_ratio`
- DCGAN saves progression frames only
- use `evaluate_progression_animation\make_progress_animation.py` to turn those frames into a GIF with adjustable speed and epoch labels

Create the animation after training:

```bat
python evaluate_progression_animation\make_progress_animation.py ^
  --frames_dir runs\dcgan_64\progress_frames ^
  --duration_ms 800
```

## Train WGAN-GP

Example:

```bat
python train_wgangp\train_wgangp.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\wgangp_64 ^
  --image_size 64 ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --batch_size 64 ^
  --epochs 100 ^
  --lr 1e-4 ^
  --beta1 0.0 ^
  --beta2 0.9 ^
  --n_critic 5 ^
  --lambda_gp 10 ^
  --gp_every 2 ^
  --ema ^
  --ema_beta 0.999 ^
  --save_progress_every 1
```

You can create a WGAN-GP animation the same way:

```bat
python evaluate_progression_animation\make_progress_animation.py ^
  --frames_dir runs\wgangp_64\progress_frames ^
  --duration_ms 800
```

## Evaluate

`evaluate_fid_kid/eval_fid.py` computes **FID** and **KID** using TorchMetrics Inception-v3 features.

Use the **same** `seed`, `train_ratio`, and `val_ratio` as training so the held-out test set matches.

Example:

```bat
python evaluate_fid_kid\eval_fid.py ^
  --data_dir data\preprocessed_slices_64 ^
  --ckpt runs\dcgan_64\checkpoint_latest.pt ^
  --split test ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --use_ema ^
  --kid_subset_size 1000
```

To compare multiple checkpoints and rank them:

```bat
python evaluate_fid_kid\rank_checkpoints.py ^
  --ckpt_dir runs\dcgan_64 ^
  --data_dir data\preprocessed_slices_64 ^
  --split test ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --use_ema ^
  --sort_by fid
```

Useful options:
- use `--split val` if you want to rank checkpoints on the validation split instead of the test split
- use `--epochs 50 80 100` to evaluate only selected epoch checkpoints
- use `--checkpoint_names checkpoint_epoch_0050.pt checkpoint_latest.pt` to evaluate an explicit file list

To compute a real-data baseline between dataset splits instead of generator-vs-real:

```bat
python evaluate_fid_kid\eval_fid.py ^
  --real_vs_real ^
  --data_dir data\preprocessed_slices_64 ^
  --split_a train ^
  --split_b test ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --kid_subset_size 1000
```

## Privacy Evaluation

`evaluate_privacy/privacy_audit.py` performs a **memorisation-risk audit** using nearest-neighbor comparisons.

It does three comparisons for each generated image:
- nearest training image
- nearest held-out real image from `val` or `test`
- cross-patient train-to-train baseline for context

Example:

```bat
python evaluate_privacy\privacy_audit.py ^
  --ckpt runs\dcgan_64\t2f_image20k_epoch100\checkpoint_epoch_0080.pt ^
  --data_dir data\preprocessed_slices_64 ^
  --reference_split test ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --num_fake 2000 ^
  --num_train_real 2000 ^
  --num_reference_real 2000 ^
  --batch_size 32 ^
  --num_workers 0 ^
  --use_ema
```

Outputs:
- `privacy_audit.csv` with one row per generated image
- `summary.txt` with aggregate statistics and interpretation notes
- `suspicious_examples/` containing fake / nearest-train / nearest-reference triplets for manual inspection

The audit reports:
- raw-pixel `L2` nearest-neighbor distance
- raw-pixel cosine similarity
- Inception feature-space `L2` nearest-neighbor distance

Useful summary fields:
- `fraction_train_closer_l2`: fraction of fake images whose nearest training image is closer than their nearest held-out image
- `fraction_train_closer_feature_l2`: same idea in Inception feature space
- `train_self_nn_l2_p01`, `p05`, `p10`: the 1st, 5th, and 10th percentiles of the cross-patient train-to-train nearest-neighbor distance distribution
- `fraction_fake_below_train_self_p01`, `p05`, `p10`: fraction of fake images whose nearest-train distance is at or below those baseline thresholds

Interpretation:
- if train matches are consistently much closer than held-out matches, memorization risk is higher
- feature-space distance is usually more informative than raw-pixel distance for structural similarity
- this audit is a useful warning signal, not a formal privacy guarantee

## Docker-Ready Evaluation Folders

The evaluation utilities are split into separate self-contained folders so each can be built into its own Docker image:
- `evaluate_fid_kid`
- `evaluate_privacy`
- `evaluate_progression_animation`

Examples:

```bat
docker build -t dcgan-fid evaluate_fid_kid
docker build -t dcgan-privacy evaluate_privacy
docker build -t dcgan-animation evaluate_progression_animation
```

## Generate Images

Example:

```bat
python generate\generate.py ^
  --ckpt runs\wgangp_64\checkpoint_latest.pt ^
  --out_dir runs\generated\wgangp_64 ^
  --num 64 ^
  --batch_size 64 ^
  --save_grid ^
  --grid_nrow 8 ^
  --grid_px 1600 ^
  --use_ema ^
  --tag wgangp64
```

## Patient-Level Split

The project uses **patient-level** rather than slice-level splitting.

That means all slices from one patient go entirely into:
- train
- val
- or test

This avoids leakage between splits and makes evaluation more realistic.

## Outputs

Typical outputs in `runs/...` include:
- checkpoints
- loss curves
- sample grids
- DCGAN progression frames
- `generator_progression.gif`

## License

This repository includes a `LICENSE` file. BraTS data is not distributed with this project and must be obtained separately under its own terms.

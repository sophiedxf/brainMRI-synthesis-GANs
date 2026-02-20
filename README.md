# 2D Brain MRI Slice Generation (DCGAN & WGAN-GP; multiple resolutions)

This project trains and compares **DCGAN** and **WGAN-GP** to generate **unconditional 2D brain MRI slices** from **BraTS 2023**.

Supported resolutions: **64×64**, **128×128**, **256×256**.

Workflow:

1) Preprocess BraTS volumes → packed `.npy` slices  
2) Train **DCGAN** and/or **WGAN-GP** (optionally with **EMA**)  
3) Evaluate with **FID + KID** (TorchMetrics Inception-v3 features)  
4) Generate synthetic images + grids  

> **Important:** BraTS data is not included in this repo.  
> **Important:** Inception-based FID/KID are imperfect for MRIs; treat them as **relative** metrics only.

---

## Device (GPU vs CPU)

- All scripts automatically use **GPU (CUDA)** if available; otherwise they run on **CPU**.
- You can confirm the device from the console output (`Device: cuda` or `Device: cpu`).
- CPU is supported but will be **much slower**, especially for training and evaluation.

---

## 0) Environment setup (Windows + Conda)

### 0.1 Create environment (Python 3.10.11)

```bat
conda create -n dcgan_wgan python=3.10.11 -y
conda activate dcgan_wgan
```

### 0.2 Install dependencies

You have two options:

#### Option A (recommended for local dev): install everything
```bat
pip install -r requirements_all.txt
```

#### Option B (per-step installs / per Docker image): install only what you need
Each step folder has its own `requirements.txt`.

Example:
```bat
pip install -r train_wgangp\requirements.txt
```

> If you use CUDA PyTorch wheels, ensure the requirements file includes the correct PyTorch index (e.g. `--extra-index-url https://download.pytorch.org/whl/cu118`).

---

## 1) Repository structure (new)

```
repo_root/
├─ data/                       # NOT committed
│  ├─ raw/                     # put BraTS2023 here
│  └─ preprocessed_slices_64/  # produced by preprocessing (or _128/_256)
├─ runs/                       # NOT committed (checkpoints, samples, logs)
│
├─ preprocess/
│  ├─ preprocess.py
│  └─ requirements.txt
│
├─ train_dcgan/
│  ├─ train_dcgan.py
│  ├─ dataset.py
│  ├─ models_dcgan.py
│  ├─ utils_training.py
│  ├─ (optional) config.py
│  └─ requirements.txt
│
├─ train_wgangp/
│  ├─ train_wgangp.py
│  ├─ dataset.py
│  ├─ models_dcgan.py
│  ├─ models_wgangp.py
│  ├─ utils_training.py
│  ├─ (optional) config.py
│  └─ requirements.txt
│
├─ generate/
│  ├─ generate.py
│  ├─ models_dcgan.py
│  ├─ (optional) utils_training.py / config.py
│  └─ requirements.txt
│
├─ evaluate/
│  ├─ eval_fid_kid.py
│  ├─ dataset.py
│  ├─ models_dcgan.py
│  └─ requirements.txt
│
└─ requirements_all.txt
```

---

## 2) Download BraTS 2023 data and place it in `data/raw/`

This repository **does not include BraTS data** (and you should not commit it to GitHub).

1. Download BraTS 2023 (licence required).
2. Extract/unzip locally.
3. Put the extracted folders under:

```
data/raw/
```

The preprocessing script searches **recursively**, so nested folders are fine as long as the NIfTI files exist somewhere under `data/raw/`.

Expected modality filename suffixes (BraTS GLI naming):
- `...-t1n.nii.gz`
- `...-t1c.nii.gz`
- `...-t2w.nii.gz`
- `...-t2f.nii.gz`

If preprocessing reports “No files found”, check:
- you used `--raw_dir data/raw`
- your dataset naming matches the expected suffixes

---

## 3) Preprocessing — `preprocess/preprocess.py`

Converts 3D BraTS NIfTI volumes into a **single packed `.npy`** file of slices.

**Outputs**
- `data/preprocessed_slices_<size>/brats2023_<modality>_<size>_packed.npy`
- slices are float32 in **[-1, 1]**, background ≈ **-1**

### Key parameters (what they do + suggestions)

**Core**
- `--raw_dir` *(required)*: root folder containing BraTS, e.g. `data/raw`
- `--out_dir`: output folder, e.g. `data/preprocessed_slices_64`
- `--modality`: `t1n | t1c | t2w | t2f` (aliases: `t1`, `t1ce`, `t2`, `flair`)
- `--target_size`: `64 | 128 | 256`  
  Suggestion: start with **64** first.

**Slice selection**
- `--min_foreground`: filters nearly-empty slices  
  Suggestion: start at **500**, then tune.
- `--max_slices_per_patient`: cap per patient (0 = no cap)
- `--target_total_slices`: stop after N slices total (0 = no cap)  
  Suggestion: **10000** for fast experiments.
- `--selection`: `topk_foreground | uniform | random`  
  Suggestion: `topk_foreground` for higher-quality slices.
- `--seed`: makes selection reproducible

**Optional PNG previews**
- `--save_png_samples`
- `--png_every_n_patients`
- `--png_max_per_patient`

### Example command (64×64, T2f, ~10k slices)

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

---

## 4) Train DCGAN — `train_dcgan/train_dcgan.py`

DCGAN training supports:
- TTUR (separate LR for G and D)
- AMP (faster on CUDA)
- EMA (cleaner samples / often better FID/KID)

### Key parameters (what they do + suggestions)

**Data**
- `--data_dir`: directory containing the packed dataset file
- `--out_dir`: output run folder for checkpoints/samples
- `--image_size`: must match preprocessing size (`64/128/256`)
- `--seed`

**Model**
- `--z_dim`: latent dim (typical: **128**)
- `--ngf`, `--ndf`: channel multipliers (typical: **64**)  
  Increase for quality, decrease for speed/VRAM.

**Optimisation**
- `--epochs`
- `--batch_size`
- `--lrG`, `--lrD`
- `--beta1`, `--beta2`

**AMP**
- `--use_amp` / `--no_amp`  
  Suggestion: enable AMP on GPU unless you see instability.

**EMA**
- `--ema`
- `--ema_beta` (try **0.999**; for 256 try **0.9995**)
- `--ema_start_epoch` (usually **1**)

**Saving / resume**
- `--save_samples_every`
- `--save_ckpt_every`
- `--resume <checkpoint_path>`

### Example command (DCGAN 64×64)

```bat
python train_dcgan\train_dcgan.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\dcgan_64 ^
  --image_size 64 ^
  --batch_size 128 ^
  --epochs 100 ^
  --lrG 4e-4 ^
  --lrD 2e-4 ^
  --use_amp ^
  --ema ^
  --ema_beta 0.999
```

---

## 5) Train WGAN-GP — `train_wgangp/train_wgangp.py`

WGAN-GP is usually more stable than DCGAN and often produces better samples.

### Key parameters (what they do + suggestions)

**Optimisation**
- `--lr` (common: **1e-4**)
- `--beta1 0.0 --beta2 0.9` (common WGAN-GP setting)
- `--n_critic`: critic updates per generator update  
  Typical: **5** (higher = slower).
- `--lambda_gp`: gradient penalty coefficient  
  Typical: **10**.

**Speed knob**
- `--gp_every`: compute gradient penalty every N critic steps  
  - `1` = every critic step (most accurate, slowest)  
  - `2` or `4` = faster  
  Suggestion: start with **1**, then try **2** if training is too slow.

**EMA**
- `--ema`, `--ema_beta`, `--ema_start_epoch` (same idea as DCGAN)

### Example command (WGAN-GP 64×64, faster GP)

```bat
python train_wgangp\train_wgangp.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\wgangp_64 ^
  --image_size 64 ^
  --batch_size 64 ^
  --epochs 100 ^
  --n_critic 5 ^
  --lambda_gp 10 ^
  --gp_every 2 ^
  --ema ^
  --ema_beta 0.999
```

---

## 6) Evaluate FID/KID — `evaluate/eval_fid_kid.py`

Computes FID + KID using **TorchMetrics Inception-v3 (ImageNet)** features.

### Key parameters (what they do + suggestions)

- `--ckpt` *(required)*
- `--data_dir` *(required)*
- `--num_real`, `--num_fake`  
  Suggestion: **2000** for quick comparisons; increase to 5k/10k for more stable estimates.
- `--batch_size`
- `--use_ema` / `--no_ema`  
  Suggestion: report both raw and EMA, or use EMA as primary.
- `--kid_subset_size`  
  Suggestion: **1000** (must be ≤ num_real and num_fake).

### Example command (EMA, 2k/2k)

```bat
python evaluate\eval_fid_kid.py ^
  --data_dir data\preprocessed_slices_64 ^
  --ckpt runs\wgangp_64\checkpoint_latest.pt ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --use_ema ^
  --kid_subset_size 1000
```

---

## 7) Generate images — `generate/generate.py`

Generates synthetic images from a checkpoint and optionally saves:
- a grid PNG
- individual PNGs
- a packed `.npy`

### Key parameters (what they do + suggestions)

- `--ckpt` *(required)*
- `--out_dir`
- `--num`, `--batch_size`, `--seed`
- `--use_ema` / `--no_ema`
- `--save_grid` + `--grid_nrow`
- `--grid_px` (sets final grid resolution, e.g. **1600×1600**)
- `--save_individual`
- `--save_npy`
- `--tag` (adds a label into filenames)

### Example command (1600×1600 grid)

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

---

## Notes / Good practice

- Keep `data/` and `runs/` out of GitHub (use `.gitignore`).
- Always match preprocessing size ↔ training `--image_size`.
- For reporting:
  - compare DCGAN vs WGAN-GP under identical preprocessing + evaluation settings
  - consider reporting both raw `G` and `G_ema`

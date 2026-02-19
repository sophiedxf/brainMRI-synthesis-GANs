# README — 2D Brain MRI Slice Generation (DCGAN & WGAN-GP; multiple resolutions)

This project trains **DCGAN** and **WGAN-GP** to generate **unconditional 2D brain MRI slices** from **BraTS 2023**.

Supported resolutions: **64×64**, **128×128**, **256×256**.

Workflow:

1) Preprocess BraTS volumes → packed `.npy` slices  
2) Train DCGAN and/or WGAN-GP (optionally with EMA)  
3) Evaluate with **FID + KID** (TorchMetrics Inception-v3 features)  
4) Generate synthetic images (`generate.py`)  

---

## 0) Environment setup (Windows + Conda)

### 0.1 Create environment (Python 3.10.11)

```bat
conda create -n dcgan_wgan python=3.10.11 -y
conda activate dcgan_wgan
```

### 0.2 Install dependencies from `requirements.txt`

From the repo root (where `requirements.txt` is):

```bat
pip install -r requirements.txt
```

**Verify versions (optional):**

```bat
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

---

## 1) Repository structure

```
dcgan_wgan/
├─ data/
│  ├─ raw/                       # BraTS 2023 goes here (renamed)
│  └─ preprocessed_slices_64/     # example output (or _128 / _256)
├─ runs/
│  ├─ dcgan_64/
│  └─ wgangp_64/
└─ src/
   ├─ preprocess.py
   ├─ dataset.py
   ├─ models_dcgan.py
   ├─ models_wgangp.py
   ├─ utils_training.py
   ├─ train_dcgan.py
   ├─ train_wgangp.py
   ├─ eval_fid_kid.py            # your current eval script
   └─ generate.py
```

---

## 2) Preprocessing — `src/preprocess.py`

Converts BraTS NIfTI volumes into a **single packed `.npy`** file of slices.

### Key outputs
- Packed slices: `.../brats2023_<modality>_<size>_packed.npy`
- Values are in **[-1, 1]**
- Background is set to **-1** (black when visualised)

### Parameters

**Core**
- `--raw_dir` *(required)*: where BraTS is stored (use `data/raw`)
- `--out_dir` *(required)*: output directory for packed data (e.g. `data/preprocessed_slices_64`)
- `--modality` *(required)*: which modality to extract (`t2f` (FLAIR), `t2w` (T2), `t1c` (T1ce), `t1n` (T1))
- `--target_size` *(required)*: `64 | 128 | 256`  
  - **64**: fastest iteration  
  - **128**: better detail, still manageable  
  - **256**: hardest, most compute

**Slice selection**
- `--min_foreground`: minimum foreground pixel count (filters near-empty slices)  
  - Suggestion: start with **500**, adjust if too many empty slices remain.
- `--max_slices_per_patient`: cap slices per patient (0 = no cap)  
  - Use this to reduce bias from patients with many valid slices.
- `--target_total_slices`: stop early after writing N slices (0 = no cap)  
  - Great for quick experiments (e.g. **10k** slices).
- `--selection`: `topk_foreground | uniform | random`  
  - `topk_foreground`: best quality slices, least background  
  - `uniform`: evenly distributed anatomy  
  - `random`: best diversity but includes more low-signal slices
- `--seed`: makes sampling reproducible

**Debug/preview**
- `--save_png_samples`: save occasional PNG previews (recommended early on)
- `--png_every_n_patients`: e.g. 50
- `--png_max_per_patient`: e.g. 8

### Example: preprocess 64×64 T2f (FLAIR) to ~10k slices

```bat
python src\preprocess.py ^
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

## 3) Training DCGAN — `src/train_dcgan.py`

DCGAN with:
- TTUR (different LR for G and D)
- optional AMP (recommended)
- optional EMA (recommended for cleaner samples + stabler evaluation)

### Parameters

**Data/run**
- `--data_dir`: must match the size you preprocessed
- `--out_dir`: where checkpoints/samples go
- `--image_size`: must match dataset (`64/128/256`)
- `--seed`: reproducibility

**Model**
- `--z_dim`: latent size (typical: **128**)  
  - Larger can help diversity but may slow training slightly.
- `--ngf`: generator base channels (typical: **64**)  
  - Increase for quality, decrease for speed/VRAM.
- `--ndf`: discriminator base channels (typical: **64**)  
  - Similar tradeoff to `ngf`.

**Optimisation**
- `--epochs`: training length  
- `--batch_size`: try as large as VRAM allows  
- `--lrG`, `--lrD`: TTUR rates  
  - Often `lrG > lrD` works well.
- `--beta1`, `--beta2`: Adam betas  
  - Standard DCGAN often uses `beta1=0.5, beta2=0.999`.

**AMP**
- `--use_amp`: faster training on RTX GPUs  
- `--no_amp`: disable if you see instability

**EMA**
- `--ema`: enable EMA tracking  
- `--ema_beta`: smoothing factor  
  - **0.999** for 64/128; **0.9995** can help at 256.
- `--ema_start_epoch`: when EMA starts  
  - Usually **1** is fine.

**Saving**
- `--save_samples_every`: sample grid frequency (e.g. 5 or 10)
- `--save_ckpt_every`: checkpoint frequency (e.g. 10)
- `--sample_grid_n`: number of samples in the training grid
- `--sample_grid_nrow`: grid columns

**Resume**
- `--resume`: checkpoint path

### Example: DCGAN 64×64

```bat
python src\train_dcgan.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\dcgan_64 ^
  --image_size 64 ^
  --batch_size 128 ^
  --epochs 100 ^
  --lrG 4e-4 ^
  --lrD 2e-4 ^
  --use_amp ^
  --ema ^
  --ema_beta 0.999 ^
  --ema_start_epoch 1
```

---

## 4) Training WGAN-GP — `src/train_wgangp.py`

### Parameters

**Core**
- `--data_dir`, `--out_dir`, `--image_size`, `--seed` as above

**Model**
- `--z_dim`, `--ngf`, `--ndf` as above

**Optimisation**
- `--epochs`, `--batch_size`
- `--lr`: WGAN-GP commonly uses **1e-4**
- `--beta1`, `--beta2`: commonly `0.0, 0.9`
- `--n_critic`: critic steps per generator step  
  - Typical: **5**  
  - Increasing improves critic strength but slows training linearly.
- `--lambda_gp`: gradient penalty coefficient  
  - Typical: **10**  
  - Too high can over-regularise; too low can destabilise.

**Gradient penalty frequency**
- `--gp_every`: compute GP every N critic steps  
  - `1` = every critic step (most accurate, slowest)  
  - `2` or `4` = faster with small quality tradeoff  
  - Suggestion: start with **1**, then try **2** if training is too slow.

**AMP**
- Usually **OFF** for WGAN-GP (GP can be numerically sensitive)
- `--use_amp` exists but only enable if you know it’s stable.

**EMA**
- same idea as DCGAN: cleaner samples + better evaluation stability

### Example: WGAN-GP 64×64 (faster GP)

```bat
python src\train_wgangp.py ^
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

## 5) Evaluation (FID + KID) — `src/eval_fid_kid.py`

- Uses **TorchMetrics Inception-v3 (ImageNet) features**  
- Reports **FID and KID**  
- Works with checkpoints containing `G` and optionally `G_ema`

### Parameters

**Inputs**
- `--ckpt` *(required)*: checkpoint `.pt`
- `--data_dir` *(required)*: preprocessed slice directory (contains test split)

**Counts**
- `--num_real`: number of real images from test set  
  - Typical: **2000**
  - Higher = more stable estimates but slower.
- `--num_fake`: generated samples  
  - Match `num_real` for fairness.
- `--batch_size`: evaluation batch size  
  - Use as high as VRAM allows (32 is safe).

**Performance**
- `--num_workers`: dataloader workers  
  - 2–8 depending on CPU.
- `--pin_memory`: usually helps on CUDA

**EMA**
- `--use_ema`: evaluate with EMA generator if available  
  - Suggested for reporting if you trained EMA.
- `--no_ema`: force raw generator

**KID**
- `--kid_subset_size`: subset size used internally by TorchMetrics KID  
  - Must be ≤ `num_real` and `num_fake`
  - Suggestion: use **1000** for speed; increase if you increase `num_real/num_fake`.

### Example: evaluate WGAN-GP EMA with 2k/2k

```bat
python src\eval_fid_kid.py ^
  --data_dir data\preprocessed_slices_64 ^
  --ckpt runs\wgangp_64\checkpoint_latest.pt ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --use_ema ^
  --kid_subset_size 1000
```

---

## 6) Generation — `src/generate.py`

Generates images from a checkpoint (DCGAN or WGAN-GP).

### Parameters (with guidance)

**Inputs**
- `--ckpt` *(required)*
- `--out_dir`
- `--seed`

**How many**
- `--num`: number of samples to generate
- `--batch_size`: generation batch size

**EMA**
- `--use_ema`: recommended if you trained EMA
- `--no_ema`: force raw generator

**Outputs**
- `--save_grid`: save a single grid image
- `--grid_nrow`: columns in grid (e.g. 8)
- `--grid_px`: set final grid image to exact pixel size (e.g. 1600 → 1600×1600)
- `--save_individual`: save individual PNGs
- `--save_npy`: save a packed `.npy` of generated slices
- `--tag`: optional name tag for output filenames

### Example: save 8×8 grid at 1600×1600 using EMA

```bat
python src\generate.py ^
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

## 7) Practical tips

### Choosing resolution
- Start with **64×64** to validate everything quickly.
- Move to **128×128** once your pipeline is stable.
- Use **256×256** last (hardest).

### Using EMA
- EMA tends to give **cleaner samples** and **better FID/KID stability**.
- If you report results, it’s reasonable to report **both**:
  - “raw G” metrics
  - “EMA G” metrics (often better)

## Device (GPU vs CPU)
- All scripts automatically choose **GPU (CUDA)** if available; otherwise they fall back to **CPU**.
- You can verify which device is used by checking the printed line `Device: cuda` or `Device: cpu`.
- Running on CPU is supported but will be **significantly slower** for training and evaluation.


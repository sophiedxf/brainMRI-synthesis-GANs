FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# The context should be the root of the repo
COPY generate/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command from README
# Mappings will be expected as volumes: -v ./data:/app/data -v ./runs:/app/runs
CMD ["python", "generate/generate.py", "--ckpt", "runs/wgangp_64/checkpoint_latest.pt", "--out_dir", "runs/generated/wgangp_64", "--num", "64", "--batch_size", "64", "--save_grid", "--grid_nrow", "8", "--grid_px", "1600", "--use_ema", "--tag", "wgangp64"]

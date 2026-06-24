FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/generate

COPY generate/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate/ .

CMD ["python", "generate.py", "--ckpt", "/app/runs/wgangp_64/checkpoint_latest.pt", "--out_dir", "/app/runs/generated/wgangp_64", "--num", "64", "--batch_size", "64", "--save_grid", "--grid_nrow", "8", "--grid_px", "1600", "--use_ema", "--tag", "wgangp64"]

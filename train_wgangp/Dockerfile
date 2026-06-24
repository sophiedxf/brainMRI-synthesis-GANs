FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/train_wgangp

COPY train_wgangp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_wgangp/ .

CMD ["python", "train_wgangp.py", "--data_dir", "/app/data/preprocessed_slices_64", "--out_dir", "/app/runs/wgangp_64", "--image_size", "64", "--batch_size", "64", "--epochs", "100", "--n_critic", "5", "--lambda_gp", "10", "--gp_every", "2", "--ema", "--ema_beta", "0.999"]

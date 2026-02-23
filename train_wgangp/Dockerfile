FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# The context should be the root of the repo
COPY train_wgangp/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command from README
# Mappings will be expected as volumes: -v ./data:/app/data -v ./runs:/app/runs
CMD ["python", "train_wgangp/train_wgangp.py", "--data_dir", "data/preprocessed_slices_64", "--out_dir", "runs/wgangp_64", "--image_size", "64", "--batch_size", "64", "--epochs", "100", "--n_critic", "5", "--lambda_gp", "10", "--gp_every", "2", "--ema", "--ema_beta", "0.999"]

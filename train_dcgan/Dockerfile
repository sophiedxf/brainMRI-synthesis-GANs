FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# The context should be the root of the repo
COPY train_dcgan/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command from README
# Mappings will be expected as volumes: -v ./data:/app/data -v ./runs:/app/runs
CMD ["python", "train_dcgan/train_dcgan.py", "--data_dir", "data/preprocessed_slices_64", "--out_dir", "runs/dcgan_64", "--image_size", "64", "--batch_size", "128", "--epochs", "100", "--lrG", "4e-4", "--lrD", "2e-4", "--use_amp", "--ema", "--ema_beta", "0.999"]

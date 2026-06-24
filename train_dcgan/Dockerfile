FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/train_dcgan

COPY train_dcgan/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_dcgan/ .

CMD ["python", "train_dcgan.py", "--data_dir", "/app/data/preprocessed_slices_64", "--out_dir", "/app/runs/dcgan_64", "--image_size", "64", "--batch_size", "128", "--epochs", "100", "--lrG", "4e-4", "--lrD", "2e-4", "--use_amp", "--ema", "--ema_beta", "0.999"]

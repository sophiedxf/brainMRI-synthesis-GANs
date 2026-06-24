FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/evaluate_privacy

COPY evaluate_privacy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate_privacy/ .

CMD ["python", "privacy_audit.py", "--ckpt", "/app/runs/dcgan_64/checkpoint_latest.pt", "--data_dir", "/app/data/preprocessed_slices_64", "--reference_split", "test", "--num_fake", "2000", "--num_train_real", "2000", "--num_reference_real", "2000", "--batch_size", "32", "--num_workers", "0", "--use_ema"]

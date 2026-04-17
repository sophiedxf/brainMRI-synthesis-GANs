FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python", "privacy_audit.py", "--ckpt", "/runs/checkpoint_latest.pt", "--data_dir", "/data/preprocessed_slices_64", "--reference_split", "test", "--num_fake", "2000", "--num_train_real", "2000", "--num_reference_real", "2000", "--batch_size", "32", "--num_workers", "0", "--use_ema"]

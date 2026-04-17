FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python", "eval_fid.py", "--data_dir", "/data/preprocessed_slices_64", "--ckpt", "/runs/checkpoint_latest.pt", "--num_real", "2000", "--num_fake", "2000", "--batch_size", "32", "--use_ema", "--kid_subset_size", "1000"]

FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/evaluate_fid_kid

COPY evaluate_fid_kid/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate_fid_kid/ .

CMD ["python", "eval_fid.py", "--data_dir", "/app/data/preprocessed_slices_64", "--ckpt", "/app/runs/wgangp_64/checkpoint_latest.pt", "--num_real", "2000", "--num_fake", "2000", "--batch_size", "32", "--use_ema", "--kid_subset_size", "1000"]

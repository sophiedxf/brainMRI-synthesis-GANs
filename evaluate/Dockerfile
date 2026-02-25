FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# The context should be the root of the repo
COPY evaluate/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command from README
# Mappings will be expected as volumes: -v ./data:/app/data -v ./runs:/app/runs
CMD ["python", "evaluate/eval_fid.py", "--data_dir", "data/preprocessed_slices_64", "--ckpt", "runs/wgangp_64/checkpoint_latest.pt", "--num_real", "2000", "--num_fake", "2000", "--batch_size", "32", "--use_ema", "--kid_subset_size", "1000"]

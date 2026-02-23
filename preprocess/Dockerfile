FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# The context should be the root of the repo (e.g. docker build -f preprocess/Dockerfile .)
COPY preprocess/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command from README
# Mappings will be expected as volumes: -v ./data:/app/data -v ./runs:/app/runs
CMD ["python", "preprocess/preprocess.py", "--raw_dir", "data/raw", "--out_dir", "data/preprocessed_slices_64", "--modality", "t2f", "--target_size", "64", "--min_foreground", "500", "--target_total_slices", "10000", "--selection", "topk_foreground", "--seed", "42", "--save_png_samples"]

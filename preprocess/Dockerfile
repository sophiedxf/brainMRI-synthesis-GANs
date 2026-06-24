FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/preprocess

COPY preprocess/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess/ .

CMD ["python", "preprocess.py", "--raw_dir", "/app/data/raw", "--out_dir", "/app/data/preprocessed_slices_64", "--modality", "t2f", "--target_size", "64", "--min_foreground", "500", "--target_total_slices", "10000", "--selection", "topk_foreground", "--seed", "42", "--save_png_samples"]

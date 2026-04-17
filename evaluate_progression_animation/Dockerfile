FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python", "make_progress_animation.py", "--frames_dir", "/runs/progress_frames", "--duration_ms", "800"]

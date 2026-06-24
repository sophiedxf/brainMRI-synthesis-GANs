FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/evaluate_progression_animation

COPY evaluate_progression_animation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate_progression_animation/ .

CMD ["python", "make_progress_animation.py", "--frames_dir", "/app/runs/progress_frames", "--duration_ms", "800"]

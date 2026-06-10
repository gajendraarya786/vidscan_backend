# ── Stage 1: Build ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install system build dependencies for OpenCV + numpy + ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (includes ultralytics/YOLOv8)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system libraries required by OpenCV headless + ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Create temp dir (used by /convert and /preview-frames)
RUN mkdir -p /tmp

# ── Model bootstrap ─────────────────────────────────────────────────────────
# If a trained document.pt exists in models/, it will be copied via COPY . .
# above and used automatically at runtime.
#
# If NO custom model exists yet, the server starts with OpenCV-only detection
# (zero-failure fallback). To add the model later:
#   docker cp document.pt <container>:/app/models/document.pt
#   docker restart <container>
#
# For HuggingFace Spaces: add document.pt to your Space's files directly via
# the HF web UI or git LFS.
RUN mkdir -p /app/models

# Create ultralytics config dir to avoid permission errors in container
RUN mkdir -p /root/.config/Ultralytics

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

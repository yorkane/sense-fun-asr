FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install minimal system dependencies & clean apt cache
# We use python slim because PyTorch wheel ships self-contained CUDA/cuDNN libraries,
# avoiding the 2.5GB+ duplication overhead of using nvidia/cuda runtime base images.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements to maximize Docker layer cache hits
COPY requirements.txt .

# 3. Install heavy dependencies (PyTorch + Core deps)
# Any application code changes won't trigger pip re-downloads
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# 4. Prepare directories for volume mapping
RUN mkdir -p /app/data/models /app/data/trt_cache

# 5. Copy the actual application code AT THE VERY END
# This ensures that code updates only rebuild this tiny final layer without invalidating Python library layers.
COPY ./app /app/app
COPY ./gunicorn_conf.py /app/

EXPOSE 8000

# Start server
CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]

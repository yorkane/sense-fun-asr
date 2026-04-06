FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Step 1: Install PyTorch GPU (cu121). 
# Due to ONNX Runtime fatal bugs with SenseVoice node types, we must use PyTorch natively!
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install core Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    gunicorn \
    python-multipart \
    funasr \
    ffmpeg-python \
    modelscope \
    numpy \
    librosa \
    soundfile

# The models and the trt cache will be mounted from the host
RUN mkdir -p /app/data/models
RUN mkdir -p /app/data/trt_cache

COPY ./app /app/app
COPY ./gunicorn_conf.py /app/

EXPOSE 8000

# Start server
CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]

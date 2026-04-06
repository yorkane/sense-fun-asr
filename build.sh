#!/bin/bash
set -e

echo "====================================="
echo "   FunASR Model Builder Script       "
echo "====================================="

export MODELS_DIR="$(pwd)/app_data/models"
mkdir -p "$MODELS_DIR"

echo "1. Installing export dependencies natively (if not present)..."
pip3 install --user -q modelscope funasr torch torchaudio soundfile

echo "2. Copying SenseVoiceSmall Native PyTorch Model..."
python3 -c "
import shutil
import os
from modelscope import snapshot_download
print('Downloading SenseVoiceSmall...')
cache_dir = snapshot_download('iic/SenseVoiceSmall')
out_dir = '$MODELS_DIR/SenseVoiceSmall'
if not os.path.exists(os.path.join(out_dir, 'model.pt')):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(cache_dir, out_dir)
else:
    print('SenseVoiceSmall already downloaded.')
print('SenseVoiceSmall copied successfully.')
"

echo "3. Copying FSMN-VAD Native PyTorch Model..."
python3 -c "
import shutil
import os
from modelscope import snapshot_download
print('Downloading FSMN-VAD...')
cache_dir = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
out_dir = '$MODELS_DIR/Fsmn-Vad'
if not os.path.exists(os.path.join(out_dir, 'model.pt')):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(cache_dir, out_dir)
else:
    print('FSMN-VAD already downloaded.')
print('FSMN-VAD copied successfully.')
"

echo "====================================="
echo "4. Building Docker Image (ghcr.io/yorkane/sense-fun-asr:latest)..."

# Support running with or without sudo wrapper for Docker depending on the user group
if command -v docker &> /dev/null; then
    if groups | grep -q '\bdocker\b' || [ "$(id -u)" -eq 0 ]; then
        docker build -t ghcr.io/yorkane/sense-fun-asr:latest .
    else
        sudo docker build -t ghcr.io/yorkane/sense-fun-asr:latest .
    fi
else
    echo "Docker is not installed! Please install Docker first."
    exit 1
fi

echo "====================================="
echo "Build complete. Models and Docker Image are ready."
echo "You can now run 'sudo docker compose up -d'."

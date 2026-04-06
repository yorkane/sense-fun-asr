import shutil
import os
from modelscope import snapshot_download

# Download SenseVoiceSmall
print('Downloading SenseVoiceSmall from ModelScope...')
cache_dir = snapshot_download('iic/SenseVoiceSmall')
out_dir = '/app/data/models/SenseVoiceSmall'
if not os.path.exists(out_dir):
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    shutil.copytree(cache_dir, out_dir)

# Download FSMN-VAD
print('Downloading FSMN-VAD from ModelScope...')
cache_dir = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
out_dir = '/app/data/models/Fsmn-Vad'
if not os.path.exists(out_dir):
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    shutil.copytree(cache_dir, out_dir)

print('Models successfully baked into the image!')

from huggingface_hub import snapshot_download

# Download SenseVoiceSmall directly to target directory
print('Downloading SenseVoiceSmall from HuggingFace (FunAudioLLM)...')
snapshot_download(
    repo_id='FunAudioLLM/SenseVoiceSmall', 
    local_dir='/app/data/models/SenseVoiceSmall',
    local_dir_use_symlinks=False
)

# Download FSMN-VAD directly to target directory
print('Downloading FSMN-VAD from HuggingFace (alextomcat fork)...')
snapshot_download(
    repo_id='alextomcat/speech_fsmn_vad_zh-cn-16k-common-pytorch', 
    local_dir='/app/data/models/Fsmn-Vad',
    local_dir_use_symlinks=False
)

print('Models successfully fetched from HuggingFace and baked into the image!')

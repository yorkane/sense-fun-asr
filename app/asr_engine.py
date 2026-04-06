import os
import io
import datetime
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import librosa
import numpy as np

# Load parameters from ENV
ASR_MODEL_DIR = os.getenv("ASR_MODEL_DIR", "/app/data/models/SenseVoiceSmall") 
VAD_MODEL_DIR = os.getenv("VAD_MODEL_DIR", "/app/data/models/Fsmn-Vad")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

print(f"Initializing VAD Model (PyTorch): {VAD_MODEL_DIR}")
vad_model = AutoModel(model=VAD_MODEL_DIR, disable_update=True)

print(f"Initializing ASR Engine (PyTorch GPU)... Model: {ASR_MODEL_DIR}")
try:
    asr_model = AutoModel(
        model=ASR_MODEL_DIR,
        device="cuda:0",
        disable_update=True
    )
    print("ASR Engine (PyTorch) Initialized Successfully!")
except Exception as e:
    print(f"Failed to initialize ASR model: {e}")

# SenseVoice models inherently predict punctuation, we don't need a separate punc model.

def format_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT time format: HH:MM:SS,mmm"""
    td = datetime.timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    hours += td.days * 24
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def process_audio_file(audio_path: str) -> str:
    """
    Process an audio file explicitly:
    1. Extract VAD segments
    2. Slice Audio
    3. Generate text using SenseVoice batched
    4. Format SRT
    """
    # 1. Provide audio to VAD
    vad_res = vad_model.generate(input=audio_path)
    
    # 2. Extract waveforms
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    
    if len(vad_res) == 0 or 'value' not in vad_res[0]:
        return "1\n00:00:00,000 --> 00:00:10,000\n[No Speech Detected]\n"
        
    segments = vad_res[0]['value'] # List of [start_ms, end_ms]
    
    audio_chunks = []
    chunk_timestamps = []
    
    for seg in segments:
        start_ms, end_ms = seg[0], seg[1]
        start_frame = int((start_ms / 1000.0) * sample_rate)
        end_frame = int((end_ms / 1000.0) * sample_rate)
        chunk = waveform[start_frame:end_frame]
            
        audio_chunks.append(chunk)
        chunk_timestamps.append((start_ms, end_ms))
        
    # 3. Batch prediction using ASR model
    srt_content = io.StringIO()
    
    all_texts = []
    for i in range(0, len(audio_chunks), BATCH_SIZE):
        batch = audio_chunks[i:i+BATCH_SIZE]
        # Native PyTorch AutoModel generation
        asr_res = asr_model.generate(input=batch)
        
        for r in asr_res:
            raw_text = r.get('text', '')
            clean_text = rich_transcription_postprocess(raw_text)
            all_texts.append(clean_text)
            
    # 4. Generate SRT
    for i, (t_ms, text) in enumerate(zip(chunk_timestamps, all_texts)):
        start_time = format_timestamp(t_ms[0])
        end_time = format_timestamp(t_ms[1])
        
        srt_content.write(f"{i+1}\n")
        srt_content.write(f"{start_time} --> {end_time}\n")
        srt_content.write(f"{text}\n\n")

    return srt_content.getvalue()

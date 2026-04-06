from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import tempfile
import os
import shutil

# Import the engine logic
from app.asr_engine import process_audio_file

app = FastAPI(
    title="High-Performance Audio Processing Service",
    description="""
    这是基于 FastAPI 和 Gunicorn 构建的极速音频处理服务端。
    主要能力：
    - **音频打点与切分**：集成离线 FSMN-VAD 获取纳秒级音频静音片段。
    - **高并发识别**：使用 SenseVoiceSmall ONNX 格式集成 TensorRT 进行全并发混合多语种识别。
    - **SRT 支持**：通过 HTTP 同步长连接，流程式返回符合工业标准的 `.srt` 结果。
    """,
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.post(
    "/transcribe", 
    response_class=PlainTextResponse,
    summary="提取媒体音频并转写为纯净带时间戳的 SRT 字幕",
    description="""
    接受上传任意常见格式音频文件（WAV, MP3, MP4 等），由服务直接处理。
    由于长音频需耗散数十秒运算时间，请确保请求侧 HTTP Client 或前置 Nginx 等代理设置了足够的 `proxy_read_timeout`。
    
    返回：纯文本编码的 `.srt` 格式字幕数据。
    """,
    tags=["Audio Processing"]
)
async def transcribe_audio(file: UploadFile = File(..., description="需要转录音视频文件，支持任意ffmpeg兼容格式")):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    print(f"Received file: {file.filename}")
    
    # Save uploaded file to a temporary location
    try:
        # Create a temp file with the same extension if possible
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name
            
        print(f"Saved temporary file to {temp_path}")
        
        # Process audio asynchronously in threadpool to avoid blocking event loop
        import asyncio
        loop = asyncio.get_event_loop()
        srt_result = await loop.run_in_executor(None, process_audio_file, temp_path)
        
        return srt_result
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

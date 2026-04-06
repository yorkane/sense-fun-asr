import multiprocessing

# Gunicorn config variables
loglevel = "info"
workers = 4  # adjust based on 4090 24G, usually 4 is a good balance for ASR to avoid OOM
bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 6000  # High timeout for synchronous HTTP processing of long audio files

# Preload the application to save memory and startup time
preload_app = False # FunASR models being loaded on GPU might have issues if forked after initialization. It's better to load inside the worker.

def post_worker_init(worker):
    pass
    

import requests
import time
import concurrent.futures
import sys
import mimetypes

def upload_task(file_path, url="http://localhost:7880/transcribe"):
    start_time = time.time()
    try:
        with open(file_path, 'rb') as f:
            mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            files = {'file': (file_path, f, mime_type)}
            response = requests.post(url, files=files, timeout=600)  # 10 minutes timeout
            
        latency = time.time() - start_time
        if response.status_code == 200:
            srt_len = len(response.text)
            return True, latency, srt_len
        else:
            return False, latency, f"Status: {response.status_code}, Error: {response.text}"
    except Exception as e:
         latency = time.time() - start_time
         return False, latency, str(e)

def run_performance_test(concurrent_users=2, file_path="test/audio.wav"):
    print(f"--- Starting Performance Test ---")
    print(f"Concurrent Users: {concurrent_users}")
    print(f"File: {file_path}")
    
    total_start = time.time()
    
    success_count = 0
    fail_count = 0
    latencies = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(upload_task, file_path) for _ in range(concurrent_users)]
        
        for future in concurrent.futures.as_completed(futures):
            res, latency, details = future.result()
            latencies.append(latency)
            if res:
                success_count += 1
                print(f"[SUCCESS] Latency: {latency:.2f}s, SRT Length: {details} chars")
            else:
                fail_count += 1
                print(f"[FAILED] Latency: {latency:.2f}s, Reason: {details}")

    total_time = time.time() - total_start
    
    print("\n--- Summary ---")
    print(f"Total Time taken: {total_time:.2f}s")
    print(f"Successful Requests: {success_count}/{concurrent_users}")
    if latencies:
        print(f"Min Latency: {min(latencies):.2f}s")
        print(f"Max Latency: {max(latencies):.2f}s")
        print(f"Avg Latency: {(sum(latencies)/len(latencies)):.2f}s")

if __name__ == "__main__":
    users = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_performance_test(users)

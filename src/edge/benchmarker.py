import torch
import torch.nn as nn
import time
import os
import sys
import yaml
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP

def load_config():
    with open("configs/edge.yaml", "r") as f:
        return yaml.safe_load(f)["simulation"]

def run_edge_benchmark(model_path, is_quantized=False, X_test=None, thread_limit=None):
    """
    Profiles model file size, startup time, peak memory usage, inference latency,
    and throughput under simulated CPU thread constraints.
    """
    config = load_config()
    threads = thread_limit if thread_limit is not None else config["cpu_threads"]
    
    # 1. Simulate CPU Core/Thread Limits
    torch.set_num_threads(threads)
    print(f"Simulating edge deployment with CPU threads = {threads}")
    
    # 2. Measure Startup and Loading Time
    start_time = time.time()
    
    # Measure memory before load
    mem_before = 0.0
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024.0 * 1024.0)
        
    if is_quantized:
        # Load whole quantized model object
        model = torch.load(model_path, weights_only=False)
    else:
        # Load state dict
        # We assume input_dim=44 for this project, let's load it dynamically if we can
        # For testing, we default to 44
        model = LightweightMLP(input_dim=44)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        
    model.eval()
    startup_time_sec = time.time() - start_time
    
    # Measure memory after load
    mem_after = 0.0
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_after = process.memory_info().rss / (1024.0 * 1024.0)
        
    peak_ram_mb = max(0.0, mem_after - mem_before)
    
    # Model size
    model_size_kb = os.path.getsize(model_path) / 1024.0
    
    # 3. Predict latency and throughput
    if X_test is None:
        X_test = np.random.randn(1000, 44)
        
    X_test_t = torch.FloatTensor(X_test)
    num_samples = len(X_test)
    
    # Warmup
    with torch.no_grad():
        _ = model(X_test_t[:10])
        
    # Latency evaluation (batch prediction)
    start_eval = time.time()
    with torch.no_grad():
        _ = model(X_test_t)
    batch_latency_ms = (time.time() - start_eval) * 1000.0
    
    # Latency evaluation (single-sample prediction)
    single_latencies = []
    with torch.no_grad():
        for i in range(min(100, num_samples)):
            sample = X_test_t[i].unsqueeze(0)
            start_single = time.time()
            _ = model(sample)
            single_latencies.append((time.time() - start_single) * 1000.0)
            
    avg_single_latency_ms = float(np.mean(single_latencies))
    throughput_fps = 1000.0 / avg_single_latency_ms if avg_single_latency_ms > 0 else 0.0
    
    results = {
        "model_file_size_kb": model_size_kb,
        "startup_time_sec": startup_time_sec,
        "peak_ram_diff_mb": peak_ram_mb,
        "total_ram_usage_mb": mem_after,
        "batch_latency_ms": batch_latency_ms,
        "single_sample_latency_ms": avg_single_latency_ms,
        "throughput_fps": throughput_fps,
        "threads_simulated": threads
    }
    
    return results

if __name__ == "__main__":
    # Test benchmarker
    model_fp32 = "artifacts/checkpoints/central/mlp.pt"
    if os.path.exists(model_fp32):
        print("Benchmarking FP32 Model...")
        res_fp32 = run_edge_benchmark(model_fp32, is_quantized=False)
        print("FP32 Benchmark Results:", res_fp32)
        
        model_int8 = "artifacts/checkpoints/final/mlp_quantized.pt"
        if os.path.exists(model_int8):
            print("\nBenchmarking INT8 Model...")
            res_int8 = run_edge_benchmark(model_int8, is_quantized=True)
            print("INT8 Benchmark Results:", res_int8)
    else:
        print("No models found for edge benchmarking. Run baseline training first.")

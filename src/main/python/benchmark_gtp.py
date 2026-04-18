
import time
import torch
import sys
import os

# Add the current directory to sys.path to allow importing pgl
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pgl.ops.gtp import gtp_gpu

def benchmark():
    # Data size
    N = 1000000  # 1 million rows
    print(f"Benchmarking with N={N} rows...")

    # Generate random data on CPU
    # Counts are non-negative, let's say between 0 and 100
    X1 = torch.randint(0, 100, (N,), dtype=torch.float32)
    X2 = torch.randint(0, 100, (N,), dtype=torch.float32)
    X3 = torch.randint(0, 100, (N,), dtype=torch.float32)

    # --- CPU Benchmark ---
    print("\nRunning on CPU...")
    start_time = time.time()
    # Force CPU execution
    _ = gtp_gpu(X1, X2, X3, prefer_cuda=False, out_device=torch.device('cpu'))
    end_time = time.time()
    cpu_time = end_time - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")

    # --- GPU Benchmark ---
    if torch.cuda.is_available():
        print("\nRunning on GPU...")
        
        # Move data to GPU first to measure pure computation time (optional, 
        # but gtp_gpu handles transfer if passed CPU tensors. 
        # Let's pass CPU tensors to measure the full 'end-to-end' function call as used in practice,
        # OR move them first to see kernel speed. 
        # The user asked about "using this method", which implies the function call.
        # However, usually we want to see if the calculation itself is faster.
        # Let's do a warmup first.
        
        # Warmup
        print("Warming up GPU...")
        _ = gtp_gpu(X1[:100], X2[:100], X3[:100], prefer_cuda=True)
        torch.cuda.synchronize() # Wait for warmup to finish

        start_time = time.time()
        # The function gtp_gpu handles moving X to device if needed.
        # If we want to be fair about "processing speed", we might want inputs already on GPU,
        # but often data starts on CPU. Let's stick to the function signature's behavior.
        # But to show potential speedup, let's also measure with inputs already on GPU.
        
        # Case 1: Inputs on CPU, function moves them
        # res_gpu = gtp_gpu(X1, X2, X3, prefer_cuda=True)
        # torch.cuda.synchronize()
        
        # Case 2: Inputs already on GPU (Pure computation speed)
        X1_cuda = X1.cuda()
        X2_cuda = X2.cuda()
        X3_cuda = X3.cuda()
        torch.cuda.synchronize()
        
        start_time = time.time()
        _ = gtp_gpu(X1_cuda, X2_cuda, X3_cuda, prefer_cuda=True)
        torch.cuda.synchronize() # Wait for all kernels to finish
        end_time = time.time()
        
        gpu_time = end_time - start_time
        print(f"GPU Time (inputs already on GPU): {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
    else:
        print("\nCUDA is not available. Cannot run GPU benchmark.")

if __name__ == "__main__":
    benchmark()

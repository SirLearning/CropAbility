
import time
import torch
import sys
import os
import gc
import matplotlib.pyplot as plt

# Add the current directory to sys.path to allow importing pgl
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pgl.ops.gtp import gtp_gpu

def benchmark_sizes():
    # Define data sizes to test (logarithmic scale)
    sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 200000000, 500000000, 1000000000]
    cpu_times = []
    gpu_times = []
    speedups = []

    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU benchmark.")
        return

    print(f"{'Size (N)':<15} | {'CPU Time (s)':<15} | {'GPU Time (s)':<15} | {'Speedup':<15}")
    print("-" * 70)

    # Warmup GPU
    warmup_data = torch.zeros(100, dtype=torch.float32).cuda()
    
    for N in sizes:
        # Generate random data on CPU
        X1 = torch.randint(0, 100, (N,), dtype=torch.float32)
        X2 = torch.randint(0, 100, (N,), dtype=torch.float32)
        X3 = torch.randint(0, 100, (N,), dtype=torch.float32)

        # --- CPU Benchmark ---
        start_time = time.time()
        _ = gtp_gpu(X1, X2, X3, prefer_cuda=False, out_device=torch.device('cpu'))
        cpu_dur = time.time() - start_time
        cpu_times.append(cpu_dur)

        # --- GPU Benchmark (Inputs pre-loaded to GPU for pure compute comparison) ---
        X1_cuda = X1.cuda()
        X2_cuda = X2.cuda()
        X3_cuda = X3.cuda()
        torch.cuda.synchronize()
        
        start_time = time.time()
        _ = gtp_gpu(X1_cuda, X2_cuda, X3_cuda, prefer_cuda=True)
        torch.cuda.synchronize()
        gpu_dur = time.time() - start_time
        gpu_times.append(gpu_dur)

        # Calculate speedup
        speedup = cpu_dur / gpu_dur if gpu_dur > 0 else 0
        speedups.append(speedup)
        
        torch.cuda.empty_cache()
        gc.collect()

        print(f"{N:<15} | {cpu_dur:<15.4f} | {gpu_dur:<15.4f} | {speedup:<15.2f}x")

    # --- Plotting ---
    # Set scientific style parameters for larger fonts and better aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })

    plt.figure(figsize=(16, 8))
    
    # Plot Speedup
    plt.subplot(1, 2, 1)
    # Using a scientific color (e.g., deep blue)
    plt.plot(sizes, speedups, marker='o', color='#1f77b4', label='Speedup')
    plt.xscale('log')
    plt.xlabel('Sample Number (N)', fontweight='bold')
    plt.ylabel('Speedup Factor (CPU / GPU)', fontweight='bold')
    plt.title('GPU Speedup vs Data Size', fontweight='bold')
    plt.grid(True, which="major", ls="-", alpha=0.8, color='#dddddd')
    plt.grid(True, which="minor", ls=":", alpha=0.5, color='#eeeeee')

    # Plot Raw Times (Log-Log scale for better visibility)
    plt.subplot(1, 2, 2)
    # Using scientific colors (e.g., brick red and forest green)
    plt.plot(sizes, cpu_times, marker='s', color='#d62728', label='CPU Time')
    plt.plot(sizes, gpu_times, marker='^', color='#2ca02c', label='GPU Time')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Number (N)', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')
    plt.title('Execution Time vs Data Size', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, which="major", ls="-", alpha=0.8, color='#dddddd')
    plt.grid(True, which="minor", ls=":", alpha=0.5, color='#eeeeee')

    plt.tight_layout()
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speedup_analysis.png')
    plt.savefig(output_file)
    print(f"\nAnalysis plot saved to: {output_file}")

if __name__ == "__main__":
    benchmark_sizes()

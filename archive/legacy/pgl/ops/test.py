"""
Triton operator performance testing and validation module.
"""

import os
import time
import logging
from typing import Tuple, Dict, Any

import torch
import numpy as np
from .add import add, triton_add, pytorch_add

# Triton profiling output
os.environ["TRITON_PROFILE_PATH"] = "./triton_profile.json"

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Reduce Triton log noise
    logging.getLogger("triton").setLevel(logging.WARNING)

def benchmark_add_operations(sizes = None, num_runs: int = 10):
    """
    Benchmark addition for tensors of various sizes.
    
    Args:
        sizes: List of tensor sizes to test
        num_runs: Runs per size
        
    Returns:
        Dict: Benchmark results
    """
    if sizes is None:
        sizes = [1000, 10000, 100000, 1000000]
    
    results = {
        'sizes': sizes,
        'triton_times': [],
        'pytorch_times': [],
        'speedup_ratios': [],
        'accuracy_checks': []
    }
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; skipping GPU tests")
        return results
    
    logger.info(f"Starting performance benchmark, sizes: {sizes}")
    
    for size in sizes:
        logger.info(f"Testing size: {size}")
        
        # Test data
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # GPU warmup
        for _ in range(3):
            _ = triton_add(x, y)
            _ = pytorch_add(x, y)
        torch.cuda.synchronize()
        
        # Triton timing
        triton_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            triton_result = triton_add(x, y)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            triton_times.append(end_time - start_time)
        
        # PyTorch timing
        pytorch_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pytorch_result = pytorch_add(x, y)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            pytorch_times.append(end_time - start_time)
        
        # Statistics
        avg_triton_time = np.mean(triton_times) * 1000  # convert to ms
        avg_pytorch_time = np.mean(pytorch_times) * 1000
        speedup = avg_pytorch_time / avg_triton_time if avg_triton_time > 0 else 0
        
        # Accuracy check
        try:
            accuracy_check = torch.allclose(triton_result, pytorch_result, atol=1e-6)
            if isinstance(triton_result, torch.Tensor) and isinstance(pytorch_result, torch.Tensor):
                max_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
            else:
                max_diff = 0.0
        except Exception:
            accuracy_check = False
            max_diff = float('inf')
        
        results['triton_times'].append(avg_triton_time)
        results['pytorch_times'].append(avg_pytorch_time)
        results['speedup_ratios'].append(speedup)
        results['accuracy_checks'].append(accuracy_check)
        
        logger.info(f"  Triton time: {avg_triton_time:.3f}ms")
        logger.info(f"  PyTorch time: {avg_pytorch_time:.3f}ms")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Accuracy check: {'PASS' if accuracy_check else 'FAIL'}")
        logger.info(f"  Max diff: {max_diff:.2e}")
    
    return results

def validate_correctness(test_sizes = None) -> bool:
    """
    Validate operator correctness.
    
    Args:
        test_sizes: Tensor sizes to test
        
    Returns:
        bool: True if all tests pass
    """
    if test_sizes is None:
        test_sizes = [100, 1000, 10000]
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; skipping GPU validation")
        return False
    
    all_passed = True
    
    for size in test_sizes:
        logger.info(f"Validating tensors of size {size}...")
        
        # Various input patterns
        test_cases = [
            torch.randn(size, device='cuda', dtype=torch.float32),
            torch.zeros(size, device='cuda', dtype=torch.float32),
            torch.ones(size, device='cuda', dtype=torch.float32),
            torch.full((size,), 0.5, device='cuda', dtype=torch.float32)
        ]
        
        for i, x in enumerate(test_cases):
            y = torch.randn(size, device='cuda', dtype=torch.float32)
            
            try:
                triton_result = triton_add(x, y)
                pytorch_result = pytorch_add(x, y)
                expected = x + y
                
                # Compare with PyTorch
                triton_match = torch.allclose(triton_result, pytorch_result, atol=1e-6)
                expected_match = torch.allclose(triton_result, expected, atol=1e-6)
                
                if not (triton_match and expected_match):
                    logger.error(f"Validation failed - size: {size}, test case: {i}")
                    all_passed = False
                else:
                    logger.debug(f"Validation passed - size: {size}, test case: {i}")
                    
            except Exception as e:
                logger.error(f"Validation error - size: {size}, test case: {i}, error: {e}")
                all_passed = False
    
    return all_passed

def print_system_info():
    """Print system information."""
    import torch
    logger = logging.getLogger(__name__)
    
    logger.info("=== System information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        try:
            import torch.version
            logger.info(f"CUDA version: {torch.version.cuda}")
        except (AttributeError, ImportError):
            logger.info("CUDA version: unavailable")
    
    try:
        import triton
        logger.info(f"Triton version: {triton.__version__}")
    except ImportError:
        logger.warning("Triton not installed")

if __name__ == "__main__":
    # System info
    print_system_info()
    
    # Correctness validation
    print("\n=== Correctness validation ===")
    correctness_passed = validate_correctness()
    print(f"Correctness validation: {'PASS' if correctness_passed else 'FAIL'}")
    
    # Performance benchmark
    print("\n=== Performance benchmark ===")
    benchmark_results = benchmark_add_operations()
    
    # Results summary
    if benchmark_results['sizes']:
        print("\n=== Benchmark summary ===")
        for i, size in enumerate(benchmark_results['sizes']):
            print(f"Size {size:>8}: "
                  f"Triton {benchmark_results['triton_times'][i]:>6.3f}ms, "
                  f"PyTorch {benchmark_results['pytorch_times'][i]:>6.3f}ms, "
                  f"speedup {benchmark_results['speedup_ratios'][i]:>5.2f}x")
    
    print(f"\nTriton profile saved to: {os.environ.get('TRITON_PROFILE_PATH', 'not set')}")
    print("Use this file to analyze SM/core utilization")

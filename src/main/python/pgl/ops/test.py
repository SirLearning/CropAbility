"""
Triton算子性能测试和验证模块
"""

import os
import time
import logging
from typing import Tuple, Dict, Any

import torch
import numpy as np
from .add import add, triton_add, pytorch_add

# 设置Triton性能分析
os.environ["TRITON_PROFILE_PATH"] = "./triton_profile.json"

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 减少Triton日志噪音
    logging.getLogger("triton").setLevel(logging.WARNING)

def benchmark_add_operations(sizes = None, num_runs: int = 10):
    """
    对不同大小的张量进行加法操作基准测试
    
    Args:
        sizes: 要测试的张量大小列表
        num_runs: 每个测试的运行次数
        
    Returns:
        Dict: 包含测试结果的字典
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
        logger.warning("CUDA不可用，跳过GPU测试")
        return results
    
    logger.info(f"开始性能基准测试，测试大小: {sizes}")
    
    for size in sizes:
        logger.info(f"测试大小: {size}")
        
        # 创建测试数据
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # 预热GPU
        for _ in range(3):
            _ = triton_add(x, y)
            _ = pytorch_add(x, y)
        torch.cuda.synchronize()
        
        # 测试Triton实现
        triton_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            triton_result = triton_add(x, y)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            triton_times.append(end_time - start_time)
        
        # 测试PyTorch实现
        pytorch_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pytorch_result = pytorch_add(x, y)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            pytorch_times.append(end_time - start_time)
        
        # 计算统计信息
        avg_triton_time = np.mean(triton_times) * 1000  # 转换为毫秒
        avg_pytorch_time = np.mean(pytorch_times) * 1000
        speedup = avg_pytorch_time / avg_triton_time if avg_triton_time > 0 else 0
        
        # 验证结果准确性
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
        
        logger.info(f"  Triton时间: {avg_triton_time:.3f}ms")
        logger.info(f"  PyTorch时间: {avg_pytorch_time:.3f}ms")
        logger.info(f"  加速比: {speedup:.2f}x")
        logger.info(f"  准确性检查: {'通过' if accuracy_check else '失败'}")
        logger.info(f"  最大差异: {max_diff:.2e}")
    
    return results

def validate_correctness(test_sizes = None) -> bool:
    """
    验证算子正确性
    
    Args:
        test_sizes: 要测试的张量大小列表
        
    Returns:
        bool: 所有测试是否通过
    """
    if test_sizes is None:
        test_sizes = [100, 1000, 10000]
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，跳过GPU验证")
        return False
    
    all_passed = True
    
    for size in test_sizes:
        logger.info(f"验证大小 {size} 的张量...")
        
        # 测试不同的输入模式
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
                
                # 检查与PyTorch的一致性
                triton_match = torch.allclose(triton_result, pytorch_result, atol=1e-6)
                expected_match = torch.allclose(triton_result, expected, atol=1e-6)
                
                if not (triton_match and expected_match):
                    logger.error(f"验证失败 - 大小: {size}, 测试用例: {i}")
                    all_passed = False
                else:
                    logger.debug(f"验证通过 - 大小: {size}, 测试用例: {i}")
                    
            except Exception as e:
                logger.error(f"验证出错 - 大小: {size}, 测试用例: {i}, 错误: {e}")
                all_passed = False
    
    return all_passed

def print_system_info():
    """打印系统信息"""
    import torch
    logger = logging.getLogger(__name__)
    
    logger.info("=== 系统信息 ===")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logger.info(f"CUDA设备名称: {torch.cuda.get_device_name()}")
        try:
            import torch.version
            logger.info(f"CUDA版本: {torch.version.cuda}")
        except (AttributeError, ImportError):
            logger.info("CUDA版本: 无法获取")
    
    try:
        import triton
        logger.info(f"Triton版本: {triton.__version__}")
    except ImportError:
        logger.warning("Triton未安装")

if __name__ == "__main__":
    # 打印系统信息
    print_system_info()
    
    # 运行正确性验证
    print("\n=== 正确性验证 ===")
    correctness_passed = validate_correctness()
    print(f"正确性验证: {'通过' if correctness_passed else '失败'}")
    
    # 运行性能基准测试
    print("\n=== 性能基准测试 ===")
    benchmark_results = benchmark_add_operations()
    
    # 打印结果摘要
    if benchmark_results['sizes']:
        print("\n=== 测试结果摘要 ===")
        for i, size in enumerate(benchmark_results['sizes']):
            print(f"大小 {size:>8}: "
                  f"Triton {benchmark_results['triton_times'][i]:>6.3f}ms, "
                  f"PyTorch {benchmark_results['pytorch_times'][i]:>6.3f}ms, "
                  f"加速比 {benchmark_results['speedup_ratios'][i]:>5.2f}x")
    
    print(f"\nTriton性能分析已保存到: {os.environ.get('TRITON_PROFILE_PATH', '未设置')}")
    print("你可以使用此文件分析SM/core使用情况")
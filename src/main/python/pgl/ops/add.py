"""
Triton加法算子实现
提供高性能的GPU加法计算
"""

import logging

import torch
import triton
import triton.language as tl

# 设置日志级别
logging.getLogger("triton").setLevel(logging.WARNING)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton加法kernel
    
    Args:
        x_ptr: 输入张量x的指针
        y_ptr: 输入张量y的指针  
        output_ptr: 输出张量的指针
        n_elements: 元素总数
        BLOCK_SIZE: 块大小（编译时常量）
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # 执行计算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    使用Triton kernel执行张量加法
    
    Args:
        x: 输入张量1
        y: 输入张量2
        
    Returns:
        torch.Tensor: 计算结果
        
    Raises:
        AssertionError: 当输入张量形状不匹配或不在CUDA设备上时
    """
    assert x.shape == y.shape, "输入张量形状必须相同"
    assert x.is_cuda and y.is_cuda, "输入张量必须在CUDA设备上"
    assert x.dtype == y.dtype, "输入张量数据类型必须相同"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 计算网格大小
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    # 启动kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def pytorch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    使用PyTorch原生操作执行张量加法（用于对比和fallback）
    
    Args:
        x: 输入张量1
        y: 输入张量2
        
    Returns:
        torch.Tensor: 计算结果
    """
    return torch.add(x, y)

def add(x: torch.Tensor, y: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """
    智能加法函数，自动选择最优实现
    
    Args:
        x: 输入张量1
        y: 输入张量2
        use_triton: 是否优先使用Triton（需要CUDA支持）
        
    Returns:
        torch.Tensor: 计算结果
    """
    if use_triton and x.is_cuda and y.is_cuda and torch.cuda.is_available():
        try:
            return triton_add(x, y)
        except Exception as e:
            logging.warning(f"Triton加法失败，回退到PyTorch: {e}")
            return pytorch_add(x, y)
    else:
        return pytorch_add(x, y)


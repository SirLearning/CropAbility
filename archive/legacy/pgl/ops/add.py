"""
Triton element-wise add operator implementation.
Provides high-performance GPU addition.
"""

import logging

import torch
import triton
import triton.language as tl

# Reduce Triton log noise
logging.getLogger("triton").setLevel(logging.WARNING)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton addition kernel.
    
    Args:
        x_ptr: Pointer to input tensor x
        y_ptr: Pointer to input tensor y
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size (compile-time constant)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Run tensor addition with the Triton kernel.
    
    Args:
        x: Input tensor 1
        y: Input tensor 2
        
    Returns:
        torch.Tensor: Result tensor
        
    Raises:
        AssertionError: If shapes differ or tensors are not on CUDA
    """
    assert x.shape == y.shape, "Input tensor shapes must match"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA"
    assert x.dtype == y.dtype, "Input tensor dtypes must match"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def pytorch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Tensor addition via native PyTorch (for comparison and fallback).
    
    Args:
        x: Input tensor 1
        y: Input tensor 2
        
    Returns:
        torch.Tensor: Result tensor
    """
    return torch.add(x, y)

def add(x: torch.Tensor, y: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """
    Smart add: pick the best available implementation.
    
    Args:
        x: Input tensor 1
        y: Input tensor 2
        use_triton: Prefer Triton when CUDA is available
        
    Returns:
        torch.Tensor: Result tensor
    """
    if use_triton and x.is_cuda and y.is_cuda and torch.cuda.is_available():
        try:
            return triton_add(x, y)
        except Exception as e:
            logging.warning(f"Triton add failed, falling back to PyTorch: {e}")
            return pytorch_add(x, y)
    else:
        return pytorch_add(x, y)


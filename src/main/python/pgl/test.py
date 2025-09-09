import triton
import triton.language as tl
import torch

@triton.jit
def kernel(X, Y, Z, N: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * 128
    offsets = block_start + tl.arange(0, 128)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)

def add_tensors(X, Y):
    assert X.shape == Y.shape
    N = X.numel()
    Z = torch.empty_like(X)
    grid = (triton.cdiv(N, 128),)
    kernel[grid](X, Y, Z, N)
    return Z

if __name__ == "__main__":
    X = torch.randn(1000, device='cuda')
    Y = torch.randn(1000, device='cuda')
    Z = add_tensors(X, Y)
    print(Z)



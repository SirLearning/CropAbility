import sys
sys.path.insert(0, '/data/dazheng/git/CropAbility/src/main/python')
import torch
from pgl.ops.gtp import gtp_gpu

# Simple input: three columns for three count vectors
X1 = torch.tensor([10, 5, 0, 2], device='cuda')
X2 = torch.tensor([0, 5, 10, 2], device='cuda')
X3 = torch.tensor([0, 0, 0, 6], device='cuda')

# Run gtp_gpu on GPU
Y = gtp_gpu(X1, X2, X3, prefer_cuda=True)
print('Output:')
print(Y)
print('Output device:', Y.device)
print('CUDA available:', torch.cuda.is_available())

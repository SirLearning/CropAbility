import sys
sys.path.append('src/main/python')
import torch
from pgl.ops.gtp import gtp_gpu


def main():
    # Three non-negative count vectors of length 10000 on GPU
    X1 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    X2 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    X3 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    Y = gtp_gpu(X1, X2, X3)
    print('Y.shape:', Y.shape)
    print('Y.device:', Y.device)
    # Last column should be best genotype index
    print('Best genotype indices (first 10):', Y[:10, -1].long().tolist())
    # Six likelihood columns per row
    print('First row likelihoods:', [round(float(v), 3) for v in Y[0, :6]])

if __name__ == "__main__":
    main()

import sys
sys.path.append('src/main/python')
import torch
from pgl.ops.gtp import gtp_gpu


def main():
    # 构造三个长度为10000的非负计数向量，放在GPU上
    X1 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    X2 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    X3 = torch.poisson(10 * torch.ones(10000, device='cuda'))
    Y = gtp_gpu(X1, X2, X3)
    print('Y.shape:', Y.shape)
    print('Y.device:', Y.device)
    # 检查最后一列是否为最佳基因型索引
    print('Best genotype indices (前10):', Y[:10, -1].long().tolist())
    # 检查每行6列似然值
    print('First row likelihoods:', [round(float(v), 3) for v in Y[0, :6]])

if __name__ == "__main__":
    main()

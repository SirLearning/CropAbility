import sys
sys.path.insert(0, '/data/dazheng/git/CropAbility/src/main/python')
import torch
from pgl.ops.gtp import gtp_gpu

# 构造一个简单的输入，三列分别代表三种计数
X1 = torch.tensor([10, 5, 0, 2], device='cuda')
X2 = torch.tensor([0, 5, 10, 2], device='cuda')
X3 = torch.tensor([0, 0, 0, 6], device='cuda')

# 调用gtp_gpu进行GPU运算
Y = gtp_gpu(X1, X2, X3, prefer_cuda=True)
print('输出结果:')
print(Y)
print('输出设备:', Y.device)
print('CUDA可用:', torch.cuda.is_available())

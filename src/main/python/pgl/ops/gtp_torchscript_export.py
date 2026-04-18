import sys
sys.path.append('src/main/python')
import torch
from pgl.ops.gtp import gtp_gpu


class GtpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X1, X2, X3):
        return gtp_gpu(X1, X2, X3)


def main():
    model = GtpModule()
    # 生成三个示例输入（100,），float32
    X1 = torch.ones(100, dtype=torch.float32)
    X2 = torch.ones(100, dtype=torch.float32)
    X3 = torch.ones(100, dtype=torch.float32)
    # 导出TorchScript
    traced = torch.jit.trace(model, (X1, X2, X3))
    traced.save('gtp_gpu_model.pt')
    print('TorchScript saved as gtp_gpu_model.pt')
    # 测试加载和推理
    loaded = torch.jit.load('gtp_gpu_model.pt')
    out = loaded(torch.rand(5), torch.rand(5), torch.rand(5))
    print('Test output shape:', out.shape)

if __name__ == '__main__':
    main()

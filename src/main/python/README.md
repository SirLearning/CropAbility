# PGL - Performance GPU Library

PGL是一个高性能GPU计算库，基于Triton提供GPU加速算子，并支持导出TorchScript模型供Java/C++使用。

## ✨ 特性

- 🚀 **高性能**: 基于Triton的GPU加速算子
- 🔧 **易集成**: 支持导出TorchScript模型供Java/C++使用  
- 📊 **完整测试**: 内置性能基准测试和正确性验证
- 🎯 **简单易用**: 提供简洁的Python API和命令行工具

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装PGL包（开发模式）
pip install -e .
```

## 🚀 快速开始

### Python API使用

```python
import torch
from pgl import add

# 创建GPU张量
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')

# 使用PGL加法算子（自动选择最优实现）
result = add(x, y)

print(f"结果形状: {result.shape}")
```

### 命令行使用

```bash
# 查看系统信息
python pgl_main.py info

# 运行正确性验证
python pgl_main.py test --correctness

# 运行性能基准测试  
python pgl_main.py test --benchmark

# 导出TorchScript模型
python pgl_main.py export --output model.pt

# 测试导出的模型
python pgl_main.py test --model model.pt
```

## 📁 项目结构

```
pgl/
├── __init__.py          # 主包入口
├── ops/                 # 算子实现包
│   ├── __init__.py     # 导出所有算子
│   ├── add.py          # 加法算子实现
│   ├── export.py       # 模型导出功能
│   └── test.py         # 测试和基准测试
├── pgl_main.py         # 命令行入口
├── requirements.txt    # 依赖文件
└── setup.py           # 包配置文件
```

## 🔧 算子API

### add函数

```python
add(x: torch.Tensor, y: torch.Tensor, use_triton: bool = True) -> torch.Tensor
```

智能加法函数，自动选择最优实现:
- 当`use_triton=True`且输入在CUDA设备上时，使用Triton加速
- 否则使用PyTorch原生实现
- 自动处理异常并回退到PyTorch实现

### triton_add函数

```python
triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

直接使用Triton kernel的加法实现:
- 要求输入张量在CUDA设备上
- 要求输入张量形状和数据类型相同
- 提供最佳GPU性能

### pytorch_add函数

```python
pytorch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

PyTorch原生加法实现，用于对比和fallback。

## 🏃 模型导出

### 导出TorchScript模型

```python
from pgl.ops import export_torchscript_model

# 导出模型
model = export_torchscript_model("model.pt", use_trace=True)

# 测试模型
from pgl.ops import test_exported_model
success = test_exported_model("model.pt")
```

### Java集成

导出的模型可以在Java中使用:

```java
// 加载模型
Module model = Module.load("model.pt");

// 创建输入张量
Tensor input1 = Tensor.fromBlob(data1, shape);
Tensor input2 = Tensor.fromBlob(data2, shape);

// 运行推理
Tensor result = model.forward(IValue.from(input1), IValue.from(input2)).toTensor();
```

## 📊 性能测试

### 运行基准测试

```python
from pgl.ops import benchmark_add_operations

results = benchmark_add_operations(
    sizes=[1000, 10000, 100000], 
    num_runs=10
)

# 查看结果
for i, size in enumerate(results['sizes']):
    print(f"大小 {size}: Triton {results['triton_times'][i]:.3f}ms, "
          f"PyTorch {results['pytorch_times'][i]:.3f}ms, "
          f"加速比 {results['speedup_ratios'][i]:.2f}x")
```

### 验证正确性

```python
from pgl.ops import validate_correctness

# 验证算子正确性
success = validate_correctness([100, 1000, 10000])
print(f"正确性验证: {'通过' if success else '失败'}")
```

## 🐛 故障排除

### 常见问题

1. **CUDA不可用**
   - 检查CUDA安装和GPU驱动
   - 确认PyTorch CUDA支持: `torch.cuda.is_available()`

2. **Triton导入错误**
   - 安装Triton: `pip install triton>=2.0.0`
   - 确认CUDA兼容性

3. **性能不如预期**
   - 确保数据在GPU上: `tensor.is_cuda`
   - 使用足够大的张量以充分利用GPU
   - 检查GPU内存使用情况

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在会显示详细的调试信息
result = add(x, y)
```

## 📈 性能分析

PGL自动生成Triton性能分析文件`triton_profile.json`，可用于分析GPU使用情况。

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支: `git checkout -b feature/new-op`
3. 添加测试用例
4. 提交更改: `git commit -am 'Add new operator'`
5. 推送到分支: `git push origin feature/new-op`
6. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](../../../LICENSE)文件。

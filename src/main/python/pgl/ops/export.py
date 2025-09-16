"""
模型导出模块 - 统一的TorchScript模型导出功能
"""

import logging
import torch
from .add import pytorch_add

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class TritonAddModule(torch.nn.Module):
    """
    可导出为TorchScript的Triton加法模块
    注意：由于Triton kernel无法直接序列化，这里使用PyTorch原生操作作为替代
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量1
            y: 输入张量2
            
        Returns:
            torch.Tensor: 计算结果
        """
        return pytorch_add(x, y)

def export_torchscript_model(output_path: str = "triton_add_model.pt", 
                           use_trace: bool = True) -> torch.jit.ScriptModule:
    """
    导出TorchScript模型供Java/C++使用
    
    Args:
        output_path: 输出模型文件路径
        use_trace: 是否使用trace模式（否则使用script模式）
        
    Returns:
        torch.jit.ScriptModule: 导出的模型
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("开始导出TorchScript模型...")
    
    # 创建模型实例
    model = TritonAddModule()
    model.eval()
    
    try:
        if use_trace:
            # 使用trace模式
            logger.info("使用trace模式导出...")
            example_x = torch.randn(1000, dtype=torch.float32)
            example_y = torch.randn(1000, dtype=torch.float32)
            traced_model = torch.jit.trace(model, (example_x, example_y))
            scripted_model = traced_model
        else:
            # 使用script模式
            logger.info("使用script模式导出...")
            scripted_model = torch.jit.script(model)
        
        # 保存模型
        logger.info(f"保存模型到: {output_path}")
        scripted_model.save(output_path)
        
        # 验证模型
        logger.info("验证模型...")
        loaded_model = torch.jit.load(output_path)
        
        # 创建测试数据
        test_x = torch.randn(100, dtype=torch.float32)
        test_y = torch.randn(100, dtype=torch.float32)
        
        # 测试模型
        with torch.no_grad():
            original_result = model(test_x, test_y)
            loaded_result = loaded_model(test_x, test_y)
            
        # 验证结果一致性
        if torch.allclose(original_result, loaded_result, atol=1e-6):
            logger.info("✓ 模型验证成功！")
        else:
            logger.warning("✗ 模型验证失败 - 结果不一致")
            
        logger.info("✓ TorchScript模型导出完成")
        return scripted_model
        
    except Exception as e:
        logger.error(f"模型导出失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_exported_model(model_path: str = "triton_add_model.pt") -> bool:
    """
    测试导出的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        bool: 测试是否成功
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"加载模型: {model_path}")
        model = torch.jit.load(model_path)
        
        # 创建测试数据
        test_cases = [
            (torch.randn(100), torch.randn(100)),
            (torch.zeros(50), torch.ones(50)),
            (torch.full((200,), 0.5), torch.full((200,), -0.5))
        ]
        
        for i, (x, y) in enumerate(test_cases):
            logger.info(f"测试用例 {i+1}...")
            result = model(x, y)
            expected = x + y
            
            if torch.allclose(result, expected, atol=1e-6):
                logger.info(f"✓ 测试用例 {i+1} 通过")
            else:
                logger.error(f"✗ 测试用例 {i+1} 失败")
                return False
        
        logger.info("✓ 所有测试用例通过")
        return True
    
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

if __name__ == "__main__":
    # 导出模型
    try:
        model = export_torchscript_model()
        
        # 测试导出的模型
        success = test_exported_model()
        
        if success:
            print("✓ 模型导出和验证成功！")
            print("模型已准备好供Java应用使用")
        else:
            print("✗ 模型验证失败")
            
    except Exception as e:
        print(f"导出过程出错: {e}")

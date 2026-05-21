"""
Model export module - unified TorchScript model export.
"""

import logging
import torch
from .add import pytorch_add

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class TritonAddModule(torch.nn.Module):
    """
    Triton add module exportable as TorchScript.
    Note: Triton kernels cannot be serialized directly; PyTorch ops are used here instead.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor 1
            y: Input tensor 2
            
        Returns:
            torch.Tensor: Result tensor
        """
        return pytorch_add(x, y)

def export_torchscript_model(output_path: str = "triton_add_model.pt", 
                           use_trace: bool = True) -> torch.jit.ScriptModule:
    """
    Export a TorchScript model for Java/C++ use.
    
    Args:
        output_path: Output model file path
        use_trace: Use trace mode (otherwise script mode)
        
    Returns:
        torch.jit.ScriptModule: Exported model
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting TorchScript model export...")
    
    # Create model instance
    model = TritonAddModule()
    model.eval()
    
    try:
        if use_trace:
            # Trace mode
            logger.info("Exporting with trace mode...")
            example_x = torch.randn(1000, dtype=torch.float32)
            example_y = torch.randn(1000, dtype=torch.float32)
            traced_model = torch.jit.trace(model, (example_x, example_y))
            scripted_model = traced_model
        else:
            # Script mode
            logger.info("Exporting with script mode...")
            scripted_model = torch.jit.script(model)
        
        # Save model
        logger.info(f"Saving model to: {output_path}")
        scripted_model.save(output_path)
        
        # Validate model
        logger.info("Validating model...")
        loaded_model = torch.jit.load(output_path)
        
        # Test data
        test_x = torch.randn(100, dtype=torch.float32)
        test_y = torch.randn(100, dtype=torch.float32)
        
        # Run model
        with torch.no_grad():
            original_result = model(test_x, test_y)
            loaded_result = loaded_model(test_x, test_y)
            
        # Check consistency
        if torch.allclose(original_result, loaded_result, atol=1e-6):
            logger.info("✓ Model validation succeeded!")
        else:
            logger.warning("✗ Model validation failed - results differ")
            
        logger.info("✓ TorchScript model export complete")
        return scripted_model
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_exported_model(model_path: str = "triton_add_model.pt") -> bool:
    """
    Test an exported model.
    
    Args:
        model_path: Model file path
        
    Returns:
        bool: True if all tests pass
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading model: {model_path}")
        model = torch.jit.load(model_path)
        
        # Test cases
        test_cases = [
            (torch.randn(100), torch.randn(100)),
            (torch.zeros(50), torch.ones(50)),
            (torch.full((200,), 0.5), torch.full((200,), -0.5))
        ]
        
        for i, (x, y) in enumerate(test_cases):
            logger.info(f"Test case {i+1}...")
            result = model(x, y)
            expected = x + y
            
            if torch.allclose(result, expected, atol=1e-6):
                logger.info(f"✓ Test case {i+1} passed")
            else:
                logger.error(f"✗ Test case {i+1} failed")
                return False
        
        logger.info("✓ All test cases passed")
        return True
    
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Export model
    try:
        model = export_torchscript_model()
        
        # Test exported model
        success = test_exported_model()
        
        if success:
            print("✓ Model export and validation succeeded!")
            print("Model is ready for Java applications")
        else:
            print("✗ Model validation failed")
            
    except Exception as e:
        print(f"Export error: {e}")

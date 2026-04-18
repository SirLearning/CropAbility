
import torch

class CleanupModule(torch.nn.Module):
    def forward(self):
        """
        This function attempts to allocate an impossibly large tensor (100 TB).
        This will trigger PyTorch's CachingAllocator to call empty_cache() 
        in a desperate attempt to find memory, before raising a RuntimeError (OOM).
        
        Usage in Java:
        try {
            model.forward();
        } catch (Exception e) {
            // Expected OOM, ignore it. The cache is now empty.
        }
        """
        # 25 * 10^12 * 4 bytes = 100 TB
        return torch.empty(25000000000000, dtype=torch.float32, device='cuda')

if __name__ == "__main__":
    try:
        model = CleanupModule()
        scripted = torch.jit.script(model)
        scripted.save("cleanup_model.pt")
        print("Successfully exported cleanup_model.pt")
    except Exception as e:
        print(f"Export failed: {e}")

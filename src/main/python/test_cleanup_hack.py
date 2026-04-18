
import torch

@torch.jit.script
def force_cleanup():
    try:
        # Attempt to allocate a massive tensor (e.g., ~100 TB) to force the allocator to clear cache
        # 25 * 10^12 floats * 4 bytes = 100 TB
        t = torch.empty(25000000000000, dtype=torch.float32, device='cuda')
    except RuntimeError:
        # Catch the OOM error. The allocator should have cleared the cache before raising this.
        pass
    return 0

class CleanupModule(torch.nn.Module):
    def forward(self):
        return force_cleanup()

if __name__ == "__main__":
    try:
        model = CleanupModule()
        scripted = torch.jit.script(model)
        scripted.save("cleanup_model.pt")
        print("Successfully exported cleanup_model.pt")
        
        # Test it if cuda is available
        if torch.cuda.is_available():
            print("Testing cleanup on GPU...")
            # Allocate some memory
            x = torch.empty(1024*1024*100, device='cuda') # 400MB
            print(f"Allocated: {torch.cuda.memory_allocated()}")
            print(f"Reserved: {torch.cuda.memory_reserved()}")
            del x
            print(f"After del - Reserved (should be high): {torch.cuda.memory_reserved()}")
            
            # Run cleanup
            scripted()
            
            print(f"After cleanup - Reserved (should be lower): {torch.cuda.memory_reserved()}")
    except Exception as e:
        print(f"Export/Run failed: {e}")


import torch

def cleanup_func(x):
    torch.cuda.empty_cache()
    return x

if __name__ == "__main__":
    try:
        x = torch.rand(1)
        traced = torch.jit.trace(cleanup_func, x)
        print("Successfully traced cleanup_func")
        print("Graph code:")
        print(traced.code)
        traced.save("cleanup_model.pt")
    except Exception as e:
        print(f"Trace failed: {e}")


import torch

def find_ops():
    all_ops = torch.jit.supported_ops()
    for op in all_ops:
        if 'cache' in op or 'cuda' in op:
            print(op)

if __name__ == "__main__":
    find_ops()

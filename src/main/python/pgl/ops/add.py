import logging

import torch
import triton

def add(x, y):
    return torch.add(x, y)

logging.getLogger("triton").setLevel(logging.WARNING)


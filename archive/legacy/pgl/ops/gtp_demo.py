import torch
from pgl.ops.gtp import gtp_gpu


def main():
    # Example input: counts for three categories per row
    X = torch.tensor([[10., 5., 3.], [0., 0., 0.], [2., 8., 4.]], dtype=torch.float32)
    Y = gtp_gpu(X)
    print('Input shape:', tuple(X.shape))
    print('Output shape:', tuple(Y.shape))
    print('First row (6 models):', [round(v, 4) for v in Y[0].tolist()])


if __name__ == "__main__":
    main()

from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.numpy as np
from .img2col import *

def _pool_forward(X, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    max_idx = np.argmax(X_col, axis=0)
    out = np.array(X_col[max_idx, range(max_idx.size)])

    out = out.reshape(h_out, w_out, n, d)
    out = np.transpose(out, (2, 3, 0, 1))

    return out

class Max_pool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size):
        assert isinstance(input, Variable)

        def np_fn(input_np, kernel_size):

            return _pool_forward(input_np, kernel_size)

        np_args = (input.data, kernel_size)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Max_pool2d, Max_pool2d).backward(ctx, grad_output)


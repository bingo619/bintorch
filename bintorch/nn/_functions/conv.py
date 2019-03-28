from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.scipy.signal
import jax.numpy as np
from .img2col import *

conv = autograd.scipy.signal.convolve
class Conv2d(Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weights_np, bias=None, stride=1, padding=0):
            out = conv_forward(input_np, weights_np, bias, stride, padding)

            if bias is None:
                return out
            else:
                return out

        np_args = (input.data, weights.data, None if bias is None else bias.data)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Conv2d, Conv2d).backward(ctx, grad_output)


def conv_forward(X, W, b, stride=1, padding=0):
    # cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = np.matmul(W_col, X_col)
    if b is not None:
        out += b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = np.transpose(out, (3, 0, 1, 2))

    return out



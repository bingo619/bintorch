from bintorch.autograd import Function
from bintorch.autograd import Variable
import jax.numpy as np
from jax import jit


class Linear(Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weights_np, bias):
            out = np.matmul(input_np, weights_np.T)

            if bias is None:
                return out
            else:
                return out+bias

        np_args = (input.data, weights.data, None if bias is None else bias.data)
        return np_fn, np_args, jit(np_fn)(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Linear, Linear).backward(ctx, grad_output)

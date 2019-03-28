from bintorch.autograd import Function
from bintorch.autograd import Variable
import jax.numpy as np
from jax import jit


class ReLU(Function):

    @staticmethod
    def forward(ctx, input):
        assert isinstance(input, Variable)

        def np_fn(input_np):
            return input_np * (input_np > 0)

        np_args = (input.data, )
        return np_fn, np_args, jit(np_fn)(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(ReLU, ReLU).backward(ctx, grad_output)

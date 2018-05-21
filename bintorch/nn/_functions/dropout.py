from bintorch.autograd import Function
from bintorch.autograd import Variable
import autograd.numpy as np

class Dropout(Function):

    @staticmethod
    def forward(ctx, input, p=0.5, train=False):
        assert isinstance(input, Variable)

        def np_fn(input_np, noise):
            return input_np * noise

        noise = np.random.binomial(1, p, size=input.data.shape)
        if not train:
            noise.fill(1)
        if p == 1:
            noise.fill(0)
        np_args = (input.data, noise)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)
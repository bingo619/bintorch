from bintorch.autograd import Function
from bintorch.autograd import Variable
import jax.numpy as np
import numpy as onp
from jax import jit
from jax.scipy.special import logsumexp


class CrossEntropy(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input, targets, size_average=True):
            # probs = np.exp(input - np.max(input, axis=1, keepdims=True))
            # probs /= np.sum(probs, axis=1, keepdims=True)
            # N = input.shape[0]
            #
            # probs = [probs[i, targets[i]] for i in np.arange(N)]
            # ll = np.log(np.array(probs))
            # #
            # if size_average:
            #     return -np.sum(ll / N)
            # else:
            #     return -np.sum(ll)
            logits = input - logsumexp(input, 1, keepdims=True)
            return np.sum(logits * targets)

        np_args = (input.data, target.data, size_average)
        return np_fn, np_args, jit(np_fn)(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(CrossEntropy, CrossEntropy).backward(ctx, grad_output)


class MSELoss(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input_np, target_np, size_average=True):
            if size_average:
                return np.mean((input_np - target_np) ** 2)
            else:
                return np.sum((input_np - target_np) ** 2)

        np_args = (input.data, target.data, size_average)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(MSELoss, MSELoss).backward(ctx, grad_output)


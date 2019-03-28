
import jax.numpy as np

from ..function import Function


class Add(Function):

    @staticmethod
    def forward(ctx, a, b):
        def np_fn(a, b):
            return a+b

        np_args = (a.data, b.data)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Add, Add).backward(ctx, grad_output)

def sort_args(a, b):
    return (a, b, True) if isinstance(a, np.ndarray) else (b, a, False)

class View(Function):
    @staticmethod
    def forward(ctx, a, sizes):
        def np_fn(a, sizes):
            return np.reshape(a, sizes)

        np_args = (a.data, sizes)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(View, View).backward(ctx, grad_output)

# class AddConstant(Function):
#
#     @staticmethod
#     def forward(ctx, a, b):
#         tensor, constant, ctx.tensor_first = sort_args(a, b)
#
#         return tensor.add(constant)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         if ctx.tensor_first:
#             return grad_output, None
#         else:
#             return None, grad_output

import autograd.numpy as np
from autograd.core import make_vjp as _make_vjp, make_jvp as _make_jvp
from autograd.extend import primitive, defvjp_argnum, vspace
from autograd.wrap_util import unary_to_nary


@unary_to_nary
def elementwise_grad(fun, x, initial_grad=None):
    vjp, ans = _make_vjp(fun, x)
    if vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")
    return vjp(vspace(ans).ones() if initial_grad is None else initial_grad)


import jax.numpy as np
from jax import  jit, wraps, lu
from jax import vjp
import numpy as onp

# from autograd.core import make_vjp as _make_vjp, make_jvp as _make_jvp
# from autograd.extend import primitive, defvjp_argnum, vspace
# from autograd.wrap_util import unary_to_nary


# @unary_to_nary
# def elementwise_grad(fun, x, initial_grad=None):
#     vjp, ans = _make_vjp(fun, x)
#     if vspace(ans).iscomplex:
#         raise TypeError("Elementwise_grad only applies to real-output functions.")
#     return vjp(vspace(ans).ones() if initial_grad is None else initial_grad)
from jax.api import _argnums_partial, _check_scalar


def elementwise_grad(fun, x, initial_grad=None):
    # if not initial_grad:
    #     initial_grad = 1.0

    # ans, vjp_py = vjp(fun, x)
    # # _check_scalar(ans)
    # g = vjp_py(onp.ones((), onp.result_type(ans))  if initial_grad is None else initial_grad)

    grad_fun = grad(fun, initial_grad, x)
    return grad_fun


def grad(fun, initial_grad = None, argnums=0):
  """Creates a function which evaluates the gradient of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).

  Returns:
    A function with the same arguments as `fun`, that evaluates the gradient of
    `fun`. If `argnums` is an integer then the gradient has the same shape and
    type as the positional argument indicated by that integer. If argnums is a
    tuple of integers, the gradient is a tuple of values with the same shapes
    and types as the corresponding arguments.

  For example:

  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> grad_tanh(0.2)
  array(0.961043, dtype=float32)

  """
  value_and_grad_f = value_and_grad(fun, initial_grad, argnums)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def grad_f(*args, **kwargs):
    ans, g = value_and_grad_f(*args, **kwargs)
    return g

  return grad_f

def value_and_grad(fun, initial_grad = None, argnums=0):
  """Creates a function which evaluates both `fun` and the gradient of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).

  Returns:
    A function with the same arguments as `fun` that evaluates both `fun` and
    the gradient of `fun` and returns them as a pair (a two-element tuple). If
    `argnums` is an integer then the gradient has the same shape and type as the
    positional argument indicated by that integer. If argnums is a tuple of
    integers, the gradient is a tuple of values with the same shapes and types
    as the corresponding arguments.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def value_and_grad_f(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = _argnums_partial(f, argnums, args)
    ans, vjp_py = vjp(f_partial, *dyn_args)
    # _check_scalar(ans)


    g = vjp_py(onp.ones((), onp.result_type(ans)) if initial_grad is None else initial_grad)
    g = g[0] if isinstance(argnums, int) else g
    return (ans, g)

  return value_and_grad_f
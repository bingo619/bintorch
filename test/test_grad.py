import bintorch
from bintorch.autograd import Variable
import bintorch.nn.functional as F
import jax.numpy as np

target = Variable(np.array((1, 3, 4, 3, 3)), requires_grad=False)
y = Variable(np.zeros((5, 5)), requires_grad=False)

l = Variable(np.ones((5, 5)), requires_grad=True)
m = Variable(np.ones((5, 5)), requires_grad=True)

x = l + m + y

x = F.cross_entropy(x, target)

x.backward()

print(x.data)

print(l.grad)
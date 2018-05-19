from .module import Module
from .. import functional as F

class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return F.relu(input)
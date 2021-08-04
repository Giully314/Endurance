from __future__ import annotations

from variable import Variable
import random

def _nested_list_random_init(size):
    assert(isinstance(size, (tuple, list)))
    if len(size) == 1:
        return [Variable(random.uniform(-1, 1)) for _ in range(size[0])]

    return [_nested_list_random_init(size[1:]) for _ in range(size[0])]


def _list_random_init(size):
    assert(isinstance(size, (tuple, list)))
    
    if len(size) == 1:
        return [Variable(random.uniform(-1, 1)) for _ in range(size[0])]

    out = []
    for _ in range(size[0]):
        out += _list_random_init(size[1:])

    return out

class Tensor:
    """
    Class for semplify the operations inside linear layer, neural network, etc.
    This is just a basic "tensor", using python's list. 
    """

    #TODO add property decorator
    #TODO initialize a tensor from a list/numpy array
    def __init__(self, shape):
        self.shape = tuple(shape)
        #self.tensor = _nested_list_random_init(self.size)  i don't like this approach 
        self.tensor = _list_random_init(self.shape)


    


    def dot(self, x: Tensor) -> float:
        assert(self.shape == x.shape)

        return sum([z * v for z, v in zip(self.tensor, x.tensor)], 0)
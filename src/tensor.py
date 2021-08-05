from __future__ import annotations

from variable import Variable
import random


#NOTE: this file is not intended to be a GOOD implementation of a tensor data structure. The utility functions are written to work
#without efficiency in mind. The main purpose is to write a neural network from scrath using only python to see if i understand
#the foundations. I'm planning to write a good implementation in C++, but that is another project focused on another purpose.


#Not contigous list.
def _nested_list_random_init(shape):
    assert(isinstance(shape, (tuple, list)))
    if len(shape) == 1:
        return [Variable(random.uniform(-1, 1)) for _ in range(shape[0])]

    return [_nested_list_random_init(shape[1:]) for _ in range(shape[0])]


#Contigous list
def _cont_list_random_init(shape) -> list:
    assert(isinstance(shape, (tuple, list)))
    
    if len(shape) == 1:
        return [Variable(random.uniform(-1, 1)) for _ in range(shape[0])]

    out = []
    for _ in range(shape[0]):
        out += _cont_list_random_init(shape[1:])

    return out


#from nested list of float to contigous list of variable
def _init_from_nested_list(array: list) -> list:
    """
    [ [ [1, 2, 3], [3, 4, 5] ] , 
      [ [4, 5, 2], [9, 2, 1] ] ]
    """
    
    if not isinstance(array[0], (list, tuple)):
        return [Variable(x) for x in array]

    out = []
    for l in array:
        out += _init_from_nested_list(l)

    return out


def _shape_from_nested_list(array: list) -> list:
    if not isinstance(array[0], (list, tuple)):
        return [len(array)]

    return [len(array)] + _shape_from_nested_list(array[0])




def create_tensor_from_array(array) -> Tensor:
    t = Tensor()
    t.tensor = _init_from_nested_list(array)
    t.shape = tuple(_shape_from_nested_list(array))

    return t

def create_tensor(shape):
    t = Tensor()
    t.shape = tuple(shape)
    t.tensor = _cont_list_random_init(shape)
    
    return t


class Tensor:
    """
    Class for semplify the operations inside linear layer, neural network, etc.
    This is just a basic "tensor", using python's list. 
    To create a tensor, use the utility functions.
    """

    #TODO add property decorator
    #TODO add TensorView
    def __init__(self):
        self.shape = None
        self.tensor = None


    def dot(self, x: Tensor) -> float:
        assert(self.shape == x.shape)

        return sum([z * v for z, v in zip(self.tensor, x.tensor)], 0)


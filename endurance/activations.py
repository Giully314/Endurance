from variable import Variable, ReLUOperation
from typing import Union
import math

def _unary_relu(x):
    out = Variable(x.value if x.value > 0 else 0)
    op = ReLUOperation(x, None, out)
    out.operation = op
    return out

def relu(x: Union[Variable, list[Variable]]):
    if isinstance(x, list):
        out = [_unary_relu(v) for v in x]
    else:
        out = _unary_relu(x)   
    return out


#maybe optimize for backward pass with (1 - sigmoid) * sigmoid
def _unary_sigmoid(x):
    return 1.0 / (1.0 + math.e ** (-x))


def sigmoid(x: Union[Variable, list[Variable]]):
    if isinstance(x, list):
        out = [_unary_sigmoid(v) for v in x]
    else:
        out = _unary_sigmoid(x)
    return out

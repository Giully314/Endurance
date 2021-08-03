from __future__ import annotations

from typing import Type, Union
from numbers import Number 
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


#Note: The implementation is only for educational purposes. The code could be optimized and make it more flexible.
#I intentionally wrote the operations explicitely so i could think about what happens.
#I just wanted to try to implement backpropagation after it studied it from 1) https://cs231n.github.io/optimization-2/ 
#2) https://arxiv.org/abs/1502.05767 .

"""
Other resource that i looked up:    https://sidsite.com/posts/autodiff/   
                                    https://github.com/karpathy/micrograd
"""
@dataclass
class AtomicOperation(ABC):
    """
    An atomic operation cannot be further decomposed. This class serves to stores the type of the operation, the "children" and the 
    backward operation.
    """

    v1: Variable = None 
    v2: Variable = None
    out: Variable = None
    
    @abstractmethod
    def backward(self) -> None:
        """
        Backward the gradient based on the operation that generated this node.
        """
        ...

    @abstractmethod
    def __repr__(self):
        """
        Describe the type of the operation.
        """
        ...


@dataclass
class SumOperation(AtomicOperation):
    def backward(self) -> None:
        self.v1.gradient += self.out.gradient 
        self.v2.gradient += self.out.gradient

    def __repr__(self):
        return f"{self.v1.value} + {self.v2.value}"


#Leibniz's rule
@dataclass
class MultiplicationOperation(AtomicOperation):
    def backward(self) -> None:
        self.v1.gradient += (self.v2.value * self.out.gradient )
        self.v2.gradient += (self.v1.value * self.out.gradient ) 

    def __repr__(self):
        return f"{self.v1.value} * {self.v2.value}"


@dataclass
class PowerOperation(AtomicOperation):
    def backward(self) -> None:
        self.v1.gradient += ( self.v2.value * (self.v1.value ** (self.v2.value - 1))) * self.out.gradient 
        self.v2.gradient += ( math.log(self.v1.value) * self.v1.value ** self.v2.value) * self.out.gradient

    def __repr__(self):
        return f"{self.v1.value} ** {self.v2.value}"


@dataclass
class ReLUOperation(AtomicOperation):
    def backward(self) -> None:
        gradient = self.out.gradient if self.v1.value > 0 else 0
        self.v1.gradient += gradient 

    def __repr__(self):
        return f"ReLU({self.v1.value})"



class Variable:
    """
    This class represents a value (it could be compared to the node of a computational graph).
    Stores all the informations to calculate the backward pass.
    """
    
    #TODO Add support for +=, -=, ecc.

    def __init__(self, value: Number, dtype: Type = float):
        self.value = dtype(value)
        self.dtype = dtype  #for now i don't check if 2 variables have the same type for operations.
        self.gradient = 0.0
        self.operation = None

        if not isinstance(self.dtype(), (int, float)):
            raise TypeError("Type accepted: int and float")


    def zero_gradient(self):
        self.gradient = 0.0


    #sub and div are written in terms of add and mul.

    def __add__(self, other: Union[Number, Variable]) -> Variable:
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, self.dtype)       
        op = SumOperation(self, other, out)
        out.operation = op
        return out 


    #self - other written as self + (-other)
    def __sub__(self, other: Union[Number, Variable]) -> Variable:
        return self + (-other)

    def __mul__(self, other: Union[Number, Variable]) -> Variable:
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value, self.dtype)
        op = MultiplicationOperation(self, other, out)
        out.operation = op
        return out

    #self / other  written as self * (other ** -1) 
    def __truediv__(self, other: Union[Number, Variable]) -> Variable:
        return self * (other ** -1)

    def __pow__(self, other: Union[Number, Variable]) -> Variable:
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value ** other.value, self.dtype)
        op = PowerOperation(self, other, out)
        out.operation = op
        return out

    #-self
    def __neg__(self):
        return self * -1

    #other + self
    def __radd__(self, other: Number) -> Variable:
        return self + other

    #other - self
    def __rsub__(self, other: Number) -> Variable:
        return other + (-self)

    #other * self
    def __rmul__(self, other: Number) -> Variable:
        return self * other 

    #other / self
    def __rtruediv__(self, other: Union[Number, Variable]) -> Variable:
        return other * (self ** -1)

    #other ** self
    def __rpow__(self, other: Number) -> Variable:
        other = Variable(other)
        return other ** self


    def backward(self):
        #First order the nodes with topological sort (using a DFS approach).
        ordered_nodes = []
        visited = []    

        def topological_sort(node: Variable):
            if node.operation is None:
                ordered_nodes.append(node)
                visited.append(node)
                return 
            
            vertices = [v for v in node.operation.__dict__.values()]
            visited.append(node)

            for v in vertices:
                if v is not None:
                    if v not in visited:
                        topological_sort(v)
            
            ordered_nodes.append(node)

        topological_sort(self)
        
        self.gradient = 1.0
        for v in reversed(ordered_nodes):
            if v.operation is not None:
                v.operation.backward()


    def __repr__(self):
        return f"Value: {self.value} , Gradient: {self.gradient} , Operation: {self.operation}"
    

    def check_type(self, other: Variable) -> bool:
        if self.dtype != other.dtype:
            raise TypeError(f"Different types: {self.dtype} and {other.dtype}")
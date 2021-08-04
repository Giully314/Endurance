from variable import Variable
import random
from activations import relu
from abc import ABC, abstractmethod

class Module(ABC):
    
    @abstractmethod
    def parameters(self) -> list[Variable]:
        """
        Return all parameters.
        """
        ...

    @abstractmethod
    def zero_gradient(self) -> None:
        """
        Reset the gradient of all parameters.
        """
        ...


class ArtificialNeuron(Module):
    def __init__(self, in_features):
        
        #https://www.deeplearning.ai/ai-notes/initialization/
        self.weights = [Variable(random.uniform(-1.0, 1.0)) for _ in range(in_features)]  
        self.bias = Variable(0.0)

        #For now i treat the activation as something that is detached from the artificial neuron.
        # Maybe later on, i will change this behavior. 
        # self.activation = activation if activation is not None else relu


    def compute(self, x):
        assert(len(x) == len(self.weights))

        out = sum([x * v for x, v in zip(self.weights, x)], self.bias)
        return out

    def parameters(self) -> list[Variable]:
        return self.weights + [self.bias]

    def zero_gradient(self):
        for p in self.parameters():
            p.zero_gradient()

    def __call__(self, x):
        return self.compute(x)



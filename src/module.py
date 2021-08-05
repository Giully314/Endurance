from variable import Variable
import random
from abc import ABC, abstractmethod
from typing import Union

class Module(ABC):
    """
    Abstract class that provides basic operations on Variable (like zero_gradient) and defines an interface for neural network 
    operations.
    """
    
    @abstractmethod
    def parameters(self) -> list[Variable]:
        """
        Return all parameters.
        """
        ...

    @abstractmethod
    def compute(self, x: Union[Variable, list[Variable]]) -> Union[Variable, list[Variable]]:
        """
        Define the computation of the module.
        """
        ...
    
    def zero_gradient(self) -> None:
        """
        Reset the gradient of all parameters.
        """
        for p in self.parameters():
            p.zero_gradient()

    def __call__(self, x):
        return self.compute(x)



class ArtificialNeuron(Module):
    def __init__(self, in_features):
        
        #https://www.deeplearning.ai/ai-notes/initialization/
        self.weights = [Variable(random.uniform(-1.0, 1.0)) for _ in range(in_features)]  
        self.bias = Variable(0.0)

        #For now i treat the activation as something that is detached from the artificial neuron.
        # Maybe later on, i will change this behavior. 
        # self.activation = activation if activation is not None else relu


    def compute(self, x: Union[Variable, list[Variable]]) -> Union[Variable, list[Variable]]:
        assert(len(x) == len(self.weights))

        out = sum([z * v for z, v in zip(self.weights, x)], self.bias)
        return out

    def parameters(self) -> list[Variable]:
        return self.weights + [self.bias]


class LinearLayer(Module):
    def __init__(self, input_features, output_features):
        self.neurons = [ArtificialNeuron(input_features) for _ in output_features]


    def compute(self, x: Union[Variable, list[Variable]]) -> Union[Variable, list[Variable]]:
        out = [neuron(x) for neuron in self.neurons]
        
        return out[0] if len(out) == 1 else out


    def parameters(self) -> list[Variable]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class FeedForwardNetwork(Module):
    def __init__(self, neurons_per_layer: list[int]):
        self.layers = [LinearLayer(neurons_per_layer[i], neurons_per_layer[i+1]) for i in range(len(neurons_per_layer) - 1)]

    def compute(self, x: Union[Variable, list[Variable]]) -> Union[Variable, list[Variable]]:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Variable]:
        return [p for layer in self.layers for p in layer.parameters()]
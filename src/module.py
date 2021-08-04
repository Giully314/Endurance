from variable import Variable
import random
from activations import relu


class ArtificialNeuron:
    def __init__(self, in_features, activation=None):
        
        #https://www.deeplearning.ai/ai-notes/initialization/
        self.weights = [Variable(random.uniform(-1.0, 1.0)) for _ in range(in_features)]  
        self.bias = Variable(0.0)
        self.activation = activation if activation is not None else relu


    def compute(self, x):
        assert(len(x) == len(self.weights))

        out = sum([x * v for x, v in zip(self.weights, x)], self.bias)
        out = self.activation(out)
        return out
        
    def __call__(self, x):
        return self.compute(x)

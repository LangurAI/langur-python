import activation, initialization
from numpy import append

class Neuron:
    """
    This class is the parent of all of the neuron types.
    They all inherit from this class.
    """
    pass

class Perceptron(Neuron):
    """
    This class describes the Perceptron - the 
    basic neuron.
    """
    def __init__(self, input_size=5, activation=activation.step, initialization=initialization.He, bias_initialization=initialization.Zeros, bias=bias.One):
        self.bias = bias
        self.weights = append(bias_initialization(1), initialization(input_size-1))
        self.activation = activation
        self.input_size = input_size

    def __str__(self):
        return f"{self.weights}"

p = Perceptron()
print(p)
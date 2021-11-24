import activation, initialization
from numpy import append

class Neuron:
    """
    This class is the parent of all of the neuron types.
    They all inherit from this class.
    """
    def __init__(self, input_size):
        self.input_size = input_size

class Perceptron(Neuron):
    """
    This class describes the Perceptron - the 
    basic neuron.
    """
    def __init__(self, input_size=5, activation=activation.Step, initialization=initialization.He, bias_initialization=initialization.Zeros, bias=1):
        super().__init__(input_size)
        self.bias = bias
        self.weights = append(bias_initialization(1), initialization(input_size-1))
        self.activation = activation

    def __str__(self):
        return f"{self.weights}"

p = Perceptron()
print(p)
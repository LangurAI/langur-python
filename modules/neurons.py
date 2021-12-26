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
    def __init__(self, input_size=5, activation=activation.Step, initialization=initialization.He, bias_weight_initialization=initialization.Zeros, bias=1):
        super().__init__(input_size)
        self.bias = bias
        self.weights = append(bias_weight_initialization(1), initialization(input_size))
        self.activation = activation

    def calculate(self, input_values):
        """
        Calculates the activation function output on a given
        input.
        """
        return self.activation(sum(self.weights * (append(self.bias, input_values))))

    def __str__(self):
        return f"{self.weights}"

class Radial(Neuron):
    """
    Radial neuron model.
    """
    pass

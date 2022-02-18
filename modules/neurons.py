import activation, initialization
from numpy import array

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
    def __init__(self, input_size=5, activation=activation.Step, initialization=initialization.HeNormal, bias=initialization.Zeros, learning_rate=1):
        super().__init__(input_size)
        self.bias = bias(1)[0]
        self.weights = initialization(input_size)
        self.activation = activation
        self.learning_rate = learning_rate

    def calculate(self, input_values):
        """
        Calculates the activation function output on a given
        input.
        """
        return self.activation(self.bias + sum(array(self.weights) * array(input_values)))

    def __str__(self):
        return f"{self.weights}"

class Sigmoid(Perceptron):
    """
    This class describes a sigmoid neuron
    implementation with a continuous activation
    function (sigmoid or tanh).
    """

    def __init__(self, input_size=5, activation=activation.Sigmoid, initialization=initialization.HeUniform, bias=initialization.Zeros, learning_rate=1):
        Perceptron.__init__(self, input_size, activation, initialization, bias, learning_rate)


    def calculate(self, input_values):
        """
        Calculates the activation function output on a given
        input.
        """
        return self.activation(self.bias + sum(array(self.weights) * array(input_values)))

    def __str__(self):
        return f"{self.weights}"

class Radial(Neuron):
    """
    Radial neuron model.
    """
    pass

class Instar(Neuron):
    """
    Grossberg's instar neuron.
    """
    pass

class Outstar(Neuron):
    """
    Grossberg's outstar neuron.
    """
    pass

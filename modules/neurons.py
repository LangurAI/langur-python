import activation, initialization
import numpy as np

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
        return self.activation(self.bias + sum(np.array(self.weights) * np.array(input_values)))

    def __str__(self):
        return f"Perceptron with weights: {self.weights}"


class Instar(Neuron):
    """
    Class that represents Grossberg's
    instar neuron.
    """
    def __init__(self, input_size=5, activation=activation.Identity, initialization=initialization.HeNormal, learning_rate=1):
        super().__init__(input_size)
        self.weights = initialization(input_size)
        self.activation = activation
        self.learning_rate = learning_rate
        self.last = 1

    def __str__(self):
        return f"Instar with weights: {self.weights}"
    
    def normalizeVector(self, v):
        """
        Function that returns a normalized vector v.
        """
        return v/np.sqrt(np.inner(v,v))

    def normalChecker(self, v):
        """
        Checks whether a vector is normal with length 1.
        """
        return np.sqrt(sum(v ** 2))

    def vectorCosine(self, v):
        """
        Calculates the cosine between two vectors
        (because they are both normal and have length 1).
        """
        return np.dot(self.weights, v)
    
    def calculate(self, input_values):
        """
        Function that calculates the output of the instar.
        """
        self.last =  self.vectorCosine(self.normalizeVector(input_values))
        return self.activation(self.last)
    
    def train(self, input_values, y=None):
        """
        Function that trains the instar either
        unsupervised (default) or supervised
        using the Grossberg rule.
        """
        if y == None:
            y = self.last
        self.weights = self.normalizeVector(self.weights + self.learning_rate*abs(y)*(input_values-self.weights))
        return self.weights

# Review needed

class Outstar(Neuron): 
    """
    Class that represents Grossberg's
    outstar neuron.
    """
    def __init__(self, input_size=5, initialization=initialization.HeNormal, learning_rate=1):
        super().__init__(input_size)
        self.weights = initialization(input_size)
        self.learning_rate = learning_rate
        self.last = 1

    def __str__(self):
        return f"Outstar with weights: {self.weights}"
    
    def calculate(self, input_value):
        """
        Function that calculates the output vector
        of the outstar.
        """
        pass

    def train(self, input_value):
        """
        Function that trains the outstar 
        using the Grossberg rule.
        """
        pass

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
        return self.activation(self.bias + sum(np.array(self.weights) * np.array(input_values)))

    def __str__(self):
        return f"{self.weights}"

class Radial(Neuron):
    """
    Radial neuron model.
    """
    pass

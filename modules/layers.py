from neurons import *
import activation
import initialization

class Layer:
    pass

class InputLayer(Layer):
    def __init__(self, outputs=5):
        self.outputs = outputs

    def feed(self, next_layer):
        pass

class PerceptronLayer(Layer):
    """
    Perceptron (Dense) layer definition,
    with default 5 in's *dimensions[0]*
    and 5 neurons in layer *dimensions[1]*.
    """
    def __init__(self, dimensions=(5,5), activation=activation.Step, initialization = initialization.HeNormal, bias=initialization.Zeros, learning_rate=1):
        self.dimensions = dimensions
        self.neurons = [Perceptron(input_size=dimensions[0], activation=activation, initialization=initialization, bias=bias, learning_rate=learning_rate) for _ in range(dimensions[1])]

    def __str__(self):
        s = "Perceptron layer with neurons:\n" + "\n".join([str(i) + ": " + str(self.neurons[i]) for i in range(len(self.neurons))])
        return s

    def changeNeuron(self, index, activation=activation.Step, initialization=initialization.HeNormal, bias=initialization.Zeros, learning_rate=1):
        """
        Neurons are defined jointly, but are
        also hotswapable.
        """
        self.neurons[index] = Perceptron(input_size = self.dimensions[0], activation=activation, initialization=initialization, bias=bias, learning_rate=learning_rate)

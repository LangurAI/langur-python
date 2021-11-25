class Layer:
    pass

class InputLayer(Layer):
    def __init__(self, outputs=5, connections=[]):
        self.outputs = outputs

    def feed(self, next_layer):
        pass

class PerceptronLayer(Layer):
    def __init__(self, dimensions=[5,5]):
        self.dimensions = dimensions
from numpy import random, sqrt, zeros

def Zeros(input_size):
    return zeros(input_size)

def AlphaRandom(input_size, alpha=0.01):
    """
    Random weight initialization with multiplication by constant alpha.
    """
    return random.randn(input_size)*alpha

def Random(input_size):
    """
    Random weight initialization.
    """
    return random.randn(input_size)

def He(input_size):
    """
    He weight initialization.
    """
    return random.randn(input_size)*sqrt(2/(input_size))

def Xavier(input_size):
    """
    Xavier weight initialization.
    """
    return random.randn(input_size)*sqrt(1/(input_size))
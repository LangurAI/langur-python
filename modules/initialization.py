import numpy as np

def Zeros(inp):
    return np.zeros(inp)

def AlphaRandom(inp, alpha=0.01):
    """
    Random weight initialization with multiplication by constant alpha.
    """
    return np.random.randn(inp)*alpha

def RandomNormal(inp):
    """
    Random normally distributed weight initialization.
    """
    return np.random.randn(inp)

def RandomUniform(inp, low=-1.0, high=1.0):
    """
    Random uniformly distributed initialization.
    """
    return np.random.uniform(size=inp, low=low, high=high)

# Review needed

def HeNormal(inp):
    """
    Kaiming He's weight initialization implementation
    with normal distribution.
    """
    return np.random.randn(inp)*np.sqrt(2/(inp))

def HeUniform(inp):
    """
    Kaiming He's weight initialization implementation
    with uniform distribution.
    """
    return np.random.uniform(size=inp, low=-1*np.sqrt(6/inp), high=np.sqrt(6/inp))


def XavierNormal(inp, outp=1):
    """
    Xavier Glorot's weight initialization 
    with normal distribution.
    """
    return np.random.randn(inp) * np.sqrt(2/(inp+outp))

def XavierUniform(inp, outp=1):
    """
    Xavier Glorot's weight initialization 
    with uniform distribution.
    """
    return np.random.uniform(size=inp, low=-1*np.sqrt(6/(inp+outp)), high=np.sqrt(6/(inp+outp)))

### Aliases
Random = RandomUniform
KaimingNormal = HeNormal
KaimingUniform = HeUniform
GlorotNormal = XavierNormal
GlorotUniform = XavierUniform


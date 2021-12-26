from math import exp, log

def Step(v, threshold=0, act_value=1, inact_value=0):
    """
    Step function where the threshold,
    activated and inactivated value can be defined.
    """
    return act_value if v > threshold else inact_value

def Identity(v):
    """
    Identity activation function.
    """
    return v

def Sigmoid(v):
    """
    Sigmoid function.
    """
    return 1/(1+exp(-1*v))

def Tanh(v):
    """
    Hyperbolic tangent activation function.
    """
    return (exp(v)-exp(-1*v))/(exp(v)+exp(-1*v))

def ReLU(v):
    """
    ReLU activation function.
    """
    return 0 if v <= 0 else v

def PReLU(v, alfa=1):
    """
    Parametric ReLU activation function.
    """
    return alfa*v if v<0 else v

def ELU(v, alfa=1):
    """
    Exponential linear unit activation function.
    """
    return alfa*(exp(v)-1) if v<= 0 else v

def SELU(v, alfa=1.67326, beta=1.0507):
    """
    Scaled exponential linear unit activation function.
    """
    return beta*alfa*(exp(v)-1) if v < 0 else beta*v

def SiLU(v):
    """
    Sigmoid linear unit activation function.
    """
    return v/(1+exp(-1*v))

def LeakyReLU(v):
    """
    Leaky ReLU activation function.
    """
    return 0.01*v if v<0 else v

def Softplus(v):
    """
    Softplus activation function.
    """
    return log(1 + exp(v))

def Softsign(v):
    """
    Softsign activation function.
    """
    return v/(1+abs(v))

def Gaussian(v):
    """
    Gaussian activation function.
    """
    return exp(-1*v**2)

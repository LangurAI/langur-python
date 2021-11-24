from math import exp, log


def Step(v):
    """
    Default step function.
    """
    return 1 if sum(v) >= 0 else 0


def DefinedStep(v, threshold=0, act_value=1, inact_value=0):
    """
    Step function where the threshold,
    activated and inactivated value can be defined.
    """
    return act_value if sum(v) >= threshold else inact_value

def Identity(v):
    """
    Identity activation function.
    """
    return sum(v)

def Sigmoid(v):
    """
    Sigmoid function.
    """
    return 1/(1+exp(-1*sum(v)))

def Tanh(v):
    """
    Hyperbolic tangent activation function.
    """
    x = sum(v)
    return (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))

def ReLU(v):
    """
    ReLU activation function.
    """
    x = sum(v)
    return 0 if x <= 0 else x

def PReLU(v, alfa=1):
    """
    Parametric ReLU activation function.
    """
    x = sum(v)
    return alfa*x if x<0 else x

def ELU(v, alfa=1):
    """
    Exponential linear unit activation function.
    """
    x = sum(v)
    return alfa*(exp(x)-1) if x<= 0 else x

def SELU(v, alfa=1.67326, beta=1.0507):
    """
    Scaled exponential linear unit activation function.
    """
    x = sum(v)
    return beta*alfa*(exp(x)-1) if x < 0 else beta*x

def SiLU(v):
    """
    Sigmoid linear unit activation function.
    """
    x = sum(v)
    return x/(1+exp(-1*x))

def LeakyReLU(v):
    """
    Leaky ReLU activation function.
    """
    x = sum(v)
    return 0.01*x if x<0 else x

def Softplus(v):
    """
    Softplus activation function.
    """
    return log(1 + exp(sum(v)))

def Softsign(v):
    """
    Softsign activation function.
    """
    x = sum(v)
    return x/(1+abs(x))

def Gaussian(v):
    """
    Gaussian activation function.
    """
    return exp(-1*sum(v)**2)
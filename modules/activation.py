from math import exp, log

def Step(v, threshold=0, act_value=1, inact_value=0):
    """
    Step function where the threshold,
    activated and inactivated value can be defined.
    """
    return act_value if v > threshold else inact_value

def Signum(v):
    """
    Signum is a special case of the
    Step function, where the activated
    value (more than 0) is 1 and
    otherwise it returns -1.
    """
    return Step(v, 0, 1, -1)

def Identity(v):
    """
    Identity activation function.
    """
    return v

def Sigmoid(v, alfa=1):
    """
    Sigmoid function.
    """
    return 1/(1+exp(-1*alfa*v))

def Tanh(v, alfa=1):
    """
    Hyperbolic tangent activation function.
    """
    return (exp(alfa*v)-exp(-1*alfa*v))/(exp(alfa*v)+exp(-1*alfa*v))

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

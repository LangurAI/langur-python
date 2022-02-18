import numpy as np
from neurons import Neuron, Perceptron

# Testing numpy speed, quite fast
# Preparation for Grossberg's Instar and
# Outstar implementations.

def normalizeVector(v):
    """
    Function that returns a normalized vector v.
    """
    return v/np.sqrt(np.inner(v,v))

def normalChecker(v):
    """
    Checks whether a vector is normal with length 1.
    """
    return np.sqrt(sum(v ** 2))


def vectorCosine(v, u):
    """
    Calculates the cosine between two vectors
    (because they are both normal and have length 1).
    """
    cosine = np.dot(v, u)
    return cosine

class Instar(Neuron):
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

    def vectorCosine(self, v, u):
        """
        Calculates the cosine between two vectors
        (because they are both normal and have length 1).
        """
        return np.dot(v, u)

import numpy as np


def FullConnect(v1, v2, activation=1):
    return np.array(np.dot(v1, v2.T) / activation, dtype=int)

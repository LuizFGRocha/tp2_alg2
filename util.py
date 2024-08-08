import numpy as np

def minkowski_distance(x,y,p=2):
    return np.sum(np.abs(x - y) ** p) ** (1/p)
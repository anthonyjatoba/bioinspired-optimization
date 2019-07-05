import numpy as np

def sphere(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

def rastringin(x):
    y = 10 * len(x) + sum(map(lambda i: i ** 2 - 10 * np.cos(2 * np.pi * i), x))
    return y

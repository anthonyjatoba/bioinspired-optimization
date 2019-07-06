import numpy as np

def sphere(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

def rastringin(x):
    y = 10 * len(x) + sum(map(lambda i: i ** 2 - 10 * np.cos(2 * np.pi * i), x))
    return y

def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.array(x)
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term
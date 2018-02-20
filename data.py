import random
import matplotlib.pyplot as plt
import numpy as np

def STFunction(d=2):
    def f(x):
        val = 0
        for i in range(d):
            val += x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i]
        val *= 0.5
        return val
    return f

def STDeriv1(d=2):
    def f(x):
        val = 0
        for i in range(d):
            val += 4 * x[i] ** 3 - 32 * x[i] + 5
        val *= 0.5
        return val
    return f

def genTrainData(d=2, num_samples=1024):
    fn = STFunction(d=d)
    fn_der = STDeriv1(d=d)
    samples = []
    for n in range(num_samples):
        x = np.array([random.randint(-5, 5) for i in range(d)])
        y = fn(x)
        dy = fn_der(x)
        s = (x, y, dy)
        samples.append(s)
    return samples

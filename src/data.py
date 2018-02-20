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
        for j in range(d):
            if j == 0: 
                val += 4 * x[j] ** 3 - 32 * x[j] + 5
            else:
                val += x[j] ** 4 - 16 * x[j] ** 2 + 5 * x[j]
        val *= 0.5
        return val
    return f

def STDeriv2(d=2):
    def f(x):
        val = 0
        for j in range(d):
            if j == 1: 
                val += 4 * x[j] ** 3 - 32 * x[j] + 5
            else:
                val += x[j] ** 4 - 16 * x[j] ** 2 + 5 * x[j]
        val *= 0.5
        return val
    return f


def genTrainData(d=2, num_samples=1024):
    fn = STFunction(d=d)
    fnd1 = STDeriv1(d=d)
    fnd2 = STDeriv2(d=d)
    samples = []
    for n in range(num_samples):
        x = np.array([random.randint(-5, 5) for i in range(d)])
        y = fn(x)
        dy1 = fnd1(x)
        dy2 = fnd2(x)
        dy = np.array([dy1, dy2])
        s = (x, y, dy)
        samples.append(s)
    return samples

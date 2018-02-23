import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
        val = 0.5 * (4 * x[0] ** 3 - 32 * x[0] + 5)
        return val
    return f

def STDeriv2(d=2):
    def f(x):
        val = 0.5 * (4 * x[1] ** 3 - 32 * x[1] + 5)
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

def plotSTFunction():
    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])
    X = np.array(X)
    z = []
    fn = STFunction(d=2)
    for k in range(len(X)):
        z.append(fn(X[k]))
    z = np.array(z).reshape((40, 40))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title('Styblinski-Tang Function')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plotSTGrad():
    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])
    X = np.array(X)
    z = []
    grad1 = STDeriv1(d=2)
    grad2 = STDeriv2(d=2)
    for k in range(len(X)):
        z.append([grad1(X[k]), grad2(X[k])])
    z = np.array(z)
    z = np.array(z).reshape((40, 40, 2))

    fig = plt.figure()
    plt.title('Gradient Field of Styblinski-Tang Function')
    dz = plt.quiver(x, y, z[:, :, 0], z[:, :, 1])
    plt.show()

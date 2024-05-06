import matplotlib.pyplot as plt
import numpy as np


def plot_contour(f, x_min, x_max, y_min, y_max, title, paths=None, names=None):
    fig, ax = plt.subplots()
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Zs = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z, _, _ = f(np.array([X[i, j], Y[i, j]]), hessian_flag=False)
            Zs[i, j] = Z

    ax.contour(X, Y, Zs, levels=40)

    if paths is not None:
        for path, name in zip(paths, names):
            ax.plot(path[:, 0], path[:, 1], label=name, alpha=0.5)
        ax.legend()
    ax.set_title(title)
    plt.show()


def plot_iterations(f_values, names):
    fig, ax = plt.subplots()
    for f_value, name in zip(f_values, names):
        ax.plot(f_value, label=name)
    ax.legend()
    ax.set_title("Function values vs. iteration number")
    plt.show()

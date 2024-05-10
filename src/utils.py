import matplotlib.pyplot as plt
import numpy as np


def plot_contour(f, title, paths, names):
    fig, ax = plt.subplots()

    ls = ["-", "--"]
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for i, (path, name) in enumerate(zip(paths, names)):
        xs = path[:, 0]
        ys = path[:, 1]

        if i == 0:
            ax.plot(xs[0], ys[0], marker="o", color="k", alpha=0.5)
            ax.text(xs[0], ys[0], s=" Start", fontsize=12, ha="left", va="bottom")

        ax.plot(xs, ys, ls=ls[i], lw=2, label=name, alpha=0.9)
        x_min = min(x_min, np.min(xs))
        x_max = max(x_max, np.max(xs))
        y_min = min(y_min, np.min(ys))
        y_max = max(y_max, np.max(ys))
    ax.legend()

    x = np.linspace(x_min - 0.5, x_max + 0.5, 100)
    y = np.linspace(y_min - 0.5, y_max + 0.5, 100)
    X, Y = np.meshgrid(x, y)
    Zs = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z, _, _ = f(np.array([X[i, j], Y[i, j]]), hessian_flag=False)
            Zs[i, j] = Z

    CS = ax.contour(X, Y, Zs, levels=20, alpha=0.75, cmap="coolwarm")
    ax.clabel(CS, inline=True, fontsize=10)

    ax.set_title(title)
    plt.show()


def plot_iterations(f_values, names):
    fig, ax = plt.subplots()
    ls = ["-", "--"]
    for i, (f_value, name) in enumerate(zip(f_values, names)):
        ax.plot(f_value, ls=ls[i], alpha=0.9, label=name)
    ax.legend()
    ax.set_title("Function values vs. iteration number")
    plt.show()

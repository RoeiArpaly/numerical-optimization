import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon


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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


def plot_iterations(title, f_values, names):
    fig, ax = plt.subplots()
    ls = ["-", "--"]
    for i, (f_value, name) in enumerate(zip(f_values, names)):
        ax.plot(f_value, ls=ls[i], alpha=0.9, label=name)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective Value")
    plt.show()


def plot_feasible_region_3d(f, title, paths, names):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for i, (path, name) in enumerate(zip(paths, names)):
        xs = path[:, 0]
        ys = path[:, 1]
        zs = path[:, 2]
        ax.plot(
            xs,
            ys,
            zs,
            lw=2,
            label=name,
            alpha=0.9,
            marker="o",
            markersize=5,
            color="tab:blue",
            markeredgecolor="w",
        )
        x_min = min(x_min, np.min(xs))
        x_max = max(x_max, np.max(xs))
        y_min = min(y_min, np.min(ys))
        y_max = max(y_max, np.max(ys))
    ax.legend()

    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 1 - X - Y
    # Mask values where z would be negative
    Z = np.where(Z < 0, np.nan, Z)

    # Plot the feasible region (x + y + z = 1)
    ax.plot_surface(X, Y, Z, alpha=0.2, rstride=100, cstride=100, color="tab:blue")
    ax.view_init(elev=20, azim=330)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()


def plot_feasible_region_2d(f, title, paths, names):
    fig, ax = plt.subplots()

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

        ax.plot(xs, ys, lw=2, label=name, alpha=0.9)
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
            Zs[i, j] = abs(Z)  # correct the contours

    CS = ax.contour(X, Y, Zs, levels=20, alpha=0.75, cmap="coolwarm", zorder=-1)
    ax.clabel(CS, inline=True, fontsize=10)

    # Plot the feasible region
    vertices = np.array([[1, 0], [2, 0], [2, 1], [0, 1]])
    polygon = Polygon(vertices, closed=True, color="tab:blue", alpha=0.25)
    ax.add_patch(polygon)

    # Plot the lines defining the constraints
    y1 = -x + 1  # y >= -x + 1
    y2 = np.ones(x.size)  # y <= 1
    y3 = np.zeros(x.size)  # y >= 0
    x4 = np.ones(x.size) * 2  # x <= 2
    ax.plot(x, y1, color="k", linestyle="--", zorder=-1)
    ax.plot(x, y2, color="k", linestyle="--", zorder=-1)
    ax.plot(x, y3, color="k", linestyle="--", zorder=-1)
    ax.plot(x4, x, color="k", linestyle="--", zorder=-1)

    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

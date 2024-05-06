from numpy import (
    array,
    exp,
    sqrt,
)


def test_circles(x, hessian_flag):
    # f(x) = (x.T) * Q * x
    Q = array([[1, 0], [0, 1]])
    f = 1/2 * x.T @ Q @ x
    g = Q @ x
    h = Q if hessian_flag else None
    return f, g, h


def test_ellipses(x, hessian_flag):
    # f(x) = (x.T) * Q * x
    Q = array([[1, 0], [0, 100]])
    f = 1/2 * x.T @ Q @ x
    g = Q @ x
    h = Q if hessian_flag else None
    return f, g, h


def test_rotated_ellipses(x, hessian_flag):
    # f(x) = (x.T) * Q * x
    Q = array([[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]])
    Q = Q.T @ array([[100, 0], [0, 1]]) @ Q
    f = 1/2 * x.T @ Q @ x
    g = Q @ x
    h = Q if hessian_flag else None
    return f, g, h


def test_rosenbrock(x, hessian_flag):
    # f(x) = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = array([
        400 * x[0] ** 3 - 400 * x[0] * x[1] + 2 * x[0] - 2,
        200 * (x[1] - x[0] ** 2)
    ])
    h = array([
        [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ]) if hessian_flag else None
    return f, g, h


def test_linear(x, hessian_flag):
    # f(x) = a.T * x
    a = array([1, 2])
    f = a.T @ x
    g = a
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h


def test_smoothed_corner_triangles(x, hessian_flag):
    # f(x1, x2) = e^(x1+3x2-0.1) + e^(x1-3x2-0.1) + e^(-x1-0.1)
    first_power = exp(x[0] + 3 * x[1] - 0.1)
    second_power = exp(x[0] - 3 * x[1] - 0.1)
    third_power = exp(-x[0] - 0.1)
    f = first_power + second_power + third_power
    g = array([
        first_power + second_power - third_power,
        3 * second_power - 3 * first_power,
    ])
    h = array([
        [first_power + second_power + third_power, 3 * first_power - 3 * second_power],
        [3 * first_power - 3 * second_power, 9 * first_power + 9 * second_power]
    ]) if hessian_flag else None
    return f, g, h

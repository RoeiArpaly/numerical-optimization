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
        3 * (first_power - second_power),
    ])
    h = array([
        [first_power + second_power + third_power, 0],
        [0, 9 * (first_power + second_power)],
    ]) if hessian_flag else None
    return f, g, h


def test_qp(x, hessian_flag):
    """
    min x ** 2 + y ** 2 + (z + 1) ** 2
    Subject to: x + y + z = 1
    x ≥ 0
    y ≥ 0
    z ≥ 0
    The problem finds the closest probability vector to the point (0,0, −1) = 0.
    """
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    g = array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    h = array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]) if hessian_flag else None
    return f, g, h


def test_qp_ineq_constraint_1(x, hessian_flag):
    f = -x[0]
    g = array([-1, 0, 0])
    h = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if hessian_flag else None
    return f, g, h


def test_qp_ineq_constraint_2(x, hessian_flag):
    f = -x[1]
    g = array([0, -1, 0])
    h = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if hessian_flag else None
    return f, g, h


def test_qp_ineq_constraint_3(x, hessian_flag):
    f = -x[2]
    g = array([0, 0, -1])
    h = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if hessian_flag else None
    return f, g, h


def test_lp(x, hessian_flag):
    """
    max[x + y]
    Subject to: y ≥ -x + 1
    y ≤ 1
    x ≤ 2
    y ≥ 0
    """
    f = -x[0] - x[1]
    g = array([-1, -1])
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h


def test_lp_ineq_constraint_1(x, hessian_flag):
    f = -x[1] - x[0] + 1
    g = array([-1, -1])
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h


def test_lp_ineq_constraint_2(x, hessian_flag):
    f = x[1] - 1
    g = array([0, 1])
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h


def test_lp_ineq_constraint_3(x, hessian_flag):
    f = x[0] - 2
    g = array([1, 0])
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h


def test_lp_ineq_constraint_4(x, hessian_flag):
    f = -x[1]
    g = array([0, -1])
    h = array([[0, 0], [0, 0]]) if hessian_flag else None
    return f, g, h

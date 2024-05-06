"""
Requirements for implementing test_unconstrained_min.py:
a. See the very first, basic example in https://docs.python.org/3/library/unittest.html for test
module structure using Python’s unittest framework.
b. For each of the functions in your examples file, your testing module should trigger
minimization with all four methods, and with backtracking Wolfe conditions for step length.
c. The test run should create two plots for each example:
i. The contour lines of the objective with iteration paths of all methods
ii. The function values vs. the iteration number for all methods
"""

import unittest
import numpy as np

from examples import (
    test_circles,
    test_ellipses,
    test_rotated_ellipses,
    test_rosenbrock,
    test_linear,
    test_smoothed_corner_triangles,
)
from src.unconstrained_min import LineSearchMinimization
from src.utils import (
    plot_contour,
    plot_iterations,
)


class TestUnconstrainedMin(unittest.TestCase):
    x0 = np.array([1, 1]).T
    ROSENBROCK_X0 = np.array([-1, 2]).T
    TEST_FUNCTIONS = [
        test_circles,
        test_ellipses,
        test_rotated_ellipses,
        test_rosenbrock,
        test_linear,
        test_smoothed_corner_triangles,
    ]
    METHODS = ["gradient_descent", "newton"]
    OBJ_TOL = 10e-12
    PARAM_TOL = 10e-8
    MAX_ITER = 100
    MAX_ITER_ROSENBROCK = 10_000

    def test_unconstrained_minimization(self):

        for func in self.TEST_FUNCTIONS:
            print(f"Testing function: {func.__name__}")
            f_values = []
            paths = []
            for method in self.METHODS:
                x0 = self.ROSENBROCK_X0 if func == test_rosenbrock else self.x0
                max_iter = (
                    self.MAX_ITER_ROSENBROCK if
                    func == test_rosenbrock and method == "gradient_descent" else self.MAX_ITER
                )
                minimizer = LineSearchMinimization(method=method)
                minimizer.unconstrained_minimization(
                    f=func,
                    x0=x0,
                    obj_tol=self.OBJ_TOL,
                    param_tol=self.PARAM_TOL,
                    max_iter=max_iter,
                )
                paths.append(np.array(minimizer.x_path))
                f_values.append(minimizer.f_path)
                # self.assertTrue(success)

            plot_contour(
                f=func,
                x_min=-10,
                x_max=10,
                y_min=-10,
                y_max=10,
                title=f"Contour lines of {func.__name__}",
                paths=paths,
                names=self.METHODS,
            )
            plot_iterations(f_values, self.METHODS)


if __name__ == "__main__":
    unittest.main()

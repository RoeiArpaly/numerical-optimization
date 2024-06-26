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
    OBJ_TOL = 1e-12
    PARAM_TOL = 1e-8
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
                x, f_x, success = minimizer.unconstrained_minimization(
                    f=func,
                    x0=x0,
                    obj_tol=self.OBJ_TOL,
                    param_tol=self.PARAM_TOL,
                    max_iter=max_iter,
                )
                paths.append(np.array(minimizer.x_path))
                f_values.append(minimizer.f_path)
                # last iteration details:
                print(
                    f"method: {method} -"
                    f"(x, y): {x.round(3)}, f(x, y): {round(f_x, 3)}, success: {success}"
                )

            shape_name = func.__name__.removeprefix("test_").replace("_", " ").title()
            plot_contour(
                f=func,
                title=f"{shape_name}: Contour lines",
                paths=paths,
                names=self.METHODS,
            )
            plot_iterations(
                title=f"{shape_name}: Function values vs. iteration number",
                f_values=f_values,
                names=self.METHODS,
            )


if __name__ == "__main__":
    unittest.main()

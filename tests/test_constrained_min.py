import numpy as np
import unittest

from examples import (
    test_qp,
    test_qp_ineq_constraint_1,
    test_qp_ineq_constraint_2,
    test_qp_ineq_constraint_3,
    test_lp,
    test_lp_ineq_constraint_1,
    test_lp_ineq_constraint_2,
    test_lp_ineq_constraint_3,
    test_lp_ineq_constraint_4,
)
from src.constrained_min import InteriorPointMinimizer
from src.utils import (
    plot_feasible_region_2d,
    plot_feasible_region_3d,
    plot_iterations,
)


class TestUnconstrainedMin(unittest.TestCase):

    TEST_FUNCTIONS = [test_qp, test_lp]
    OBJ_TOL = 1e-5
    MAX_ITER = 100

    def test_constrained_minimization(self):

        for func in self.TEST_FUNCTIONS:
            print(f"Testing function: {func.__name__}")

            if func == test_qp:
                x0 = np.array([0.1, 0.2, 0.7])
                ineq_constraints = np.array([
                    test_qp_ineq_constraint_1,
                    test_qp_ineq_constraint_2,
                    test_qp_ineq_constraint_3,
                ])
                A = np.array([[1, 1, 1]])
                b = np.array([1])
                plot_func = plot_feasible_region_3d
            elif func == test_lp:
                x0 = np.array([0.5, 0.75])
                ineq_constraints = np.array([
                    test_lp_ineq_constraint_1,
                    test_lp_ineq_constraint_2,
                    test_lp_ineq_constraint_3,
                    test_lp_ineq_constraint_4,
                ])
                A = np.array([])
                b = np.array([])
                plot_func = plot_feasible_region_2d
            else:
                raise ValueError("Invalid function")

            # Solve the problem
            minimizer = InteriorPointMinimizer()
            x, f_x, success = minimizer.interior_pt(
                func=func,
                ineq_constraints=ineq_constraints,
                eq_constraints_mat=A,
                eq_constraints_rhs=b,
                x0=x0,
                tol=self.OBJ_TOL,
                max_iter=self.MAX_ITER,
            )
            # last iteration details:
            print(f"(x, y): {x.round(3)}, f(x, y): {round(f_x, 3)}, success: {success}")

            shape_name = func.__name__.removeprefix("test_").replace("_", " ").title()
            plot_func(
                f=func,
                title=f"{shape_name.upper()}: Algorithm Path",
                paths=np.array([minimizer.x_path_inner]),
                names=["interior_pt"],
            )
            plot_iterations(
                title=f"{shape_name.upper()}: Function values vs. iteration number",
                f_values=[minimizer.f_path_outer, minimizer.f_path_inner],
                names=["interior_pt_outer", "interior_pt_inner"],
            )


if __name__ == "__main__":
    unittest.main()

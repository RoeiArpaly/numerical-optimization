import numpy as np


class LineSearchMinimization:
    """
    Class for unconstrained minimization using line search methods

    Attributes:
    method (str): method to use for minimization
    """

    HESSIAN_METHODS = ["newton"]

    def __init__(self, method):
        self.method = method
        self.hessian_flag = method in self.HESSIAN_METHODS
        self.x_path = []
        self.f_path = []

    def unconstrained_minimization(self,
                                   f,
                                   x0,
                                   obj_tol,
                                   param_tol,
                                   max_iter,
                                   alpha=1.0,
                                   c1=0.01,
                                   c2=0.5):
        """
        This function implements the unconstrained minimization algorithm with Wolfe conditions.

        Parameters:
        -----------
        f: function
            The function to be minimized.
        x0: float
            The starting point.
        obj_tol: float
            The numeric tolerance for successful termination in terms of small enough change in
            objective function values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)),
            or in the Newton Decrement based approximation of the objective decrease.
        param_tol: float
            The numeric tolerance for successful termination in terms of small enough distance between
            two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
        max_iter: int
            The maximum allowed number of iterations.
        step_size: float, optional
            Initial step size for the line search.
        c1: float, optional
            Parameter for the sufficient decrease condition (Armijo condition) in Wolfe conditions.
        c2: float, optional
            Parameter for the curvature condition in Wolfe conditions.

        Returns:
        --------
        final_location: float
            The final location.
        final_objective_value: float
            The final objective value.
        success: bool
            A success/failure boolean flag.
        """
        x = x0
        f_x = None
        iter_count = 0
        success = False
        for _ in range(max_iter):

            f_x, grad, hess = f(x, hessian_flag=self.hessian_flag)

            self.x_path.append(x)
            self.f_path.append(f_x)

            # Find the descent direction
            if self.method == "newton":
                # # using pseudo-inverse to avoid singular matrix
                # hess_pinv = np.linalg.pinv(hess)
                # p = -hess_pinv @ grad
                p = -hess @ grad
            else:
                p = -grad #/ np.linalg.norm(grad)

            # find the alpha
            alpha = wolfe_conditions(f=f, x=x, p=p, alpha=alpha, c=c1, t=c2)

            # take the step
            x_next = x + alpha * p

            if (
                    np.linalg.norm(x_next - x) < param_tol or
                    np.abs(f(x_next, False)[0] - f_x) < obj_tol
            ):
                success = True
                break

            x = x_next
            iter_count += 1

        return x, f_x, success


def wolfe_conditions(f, x, p, alpha, c, t=0.5, tol=1e-8):
    """
    1. Armijo condition
        - f(x) - f(x - alpha * grad) > alpha * c * grad.T * p
    2. Curvature condition
        - grad^T * (f(x) - f(x - alpha * grad)) > k * grad.T * p

    c \in (0, 1)
    k \in (c, 1)

    """

    f_x, grad, h = f(x, False)
    while -f_x + f(x + alpha * p, False)[0] > alpha * c * grad.T @ p:
        # backtracking
        alpha *= t
        if alpha < tol:
            break
    return alpha

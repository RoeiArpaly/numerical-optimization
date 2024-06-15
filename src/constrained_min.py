import numpy as np

from src.unconstrained_min import wolfe_conditions


class InteriorPointMinimizer:
    T = 1
    MU = 10

    def __init__(self):
        self.success = False
        self.x_path_inner = []
        self.f_path_inner = []
        self.x_path_outer = []
        self.f_path_outer = []

    def interior_pt(
        self,
        func,
        ineq_constraints,
        eq_constraints_mat,
        eq_constraints_rhs,
        x0,
        tol=1e-8,
        max_iter=100,

        alpha=1.0,
        c1=0.01,
        c2=0.5,
        wolfe_tol=1e-6,
    ):
        _alpha = alpha
        eq_const_n = eq_constraints_mat.shape[0]
        x = x0
        t = self.T

        self.x_path_outer.append(x)
        self.f_path_outer.append(func(x, False)[0])
        f_x, g_x, h_x = self.update_step(func, x, ineq_constraints, t)
        for i in range(max_iter):
            if eq_const_n:
                block_matrix = np.concatenate([
                    np.concatenate([h_x, eq_constraints_mat.T], axis=1),
                    np.concatenate([eq_constraints_mat, np.zeros((eq_const_n, eq_const_n))], axis=1)
                ], axis=0)
            else:
                block_matrix = h_x

            eq_vec = np.concatenate([-g_x, np.zeros(block_matrix.shape[0] - g_x.shape[0])])

            x_prev = np.inf
            f_prev = np.inf
            for j in range(max_iter):
                p = np.linalg.solve(block_matrix, eq_vec)[: x.shape[0]]
                lambda_ = np.matmul(p.transpose(), np.matmul(h_x, p)) ** 0.5
                if 0.5 * (lambda_ ** 2) < tol or sum(abs(x_prev - x)) < tol or f_prev - f_x < tol:
                    break

                alpha = wolfe_conditions(f=func, x=x, p=p, alpha=_alpha, c=c1, t=c2, tol=wolfe_tol)

                x_prev = x
                f_prev = f_x
                x = x + alpha * p
                f_x, g_x, h_x = self.update_step(func, x, ineq_constraints, t)

            if ineq_constraints.shape[0] / t < tol:
                self.success = True
                break

            self.x_path_outer.append(x)
            self.f_path_outer.append(func(x, False)[0])

            t *= self.MU

        return x, func(x, True)[0], self.success

    def update_step(self, func, x, ineq_constraints, t):
        f_x, g_x, h_x = func(x, True)
        self.x_path_inner.append(x)
        self.f_path_inner.append(f_x)
        f_x, g_x, h_x = update_phi(ineq_constraints, x, f_x, g_x, h_x, t)
        return f_x, g_x, h_x


def phi(ineq_constraints, x):
    f_star = 0
    g_star = 0
    h_star = 0
    for func in ineq_constraints:
        f_x, g_x, h_x = func(x, True)
        f_star += np.log(-f_x)
        g = g_x / f_x
        g_star += g
        g_mesh = np.tile(
            g.reshape(g.shape[0], -1), (1, g.shape[0])
        ) * np.tile(g.reshape(g.shape[0], -1).T, (g.shape[0], 1))
        h_star += (h_x * f_x - g_mesh) / f_x**2
    return -f_star, -g_star, -h_star


def update_phi(ineq_constraints, x, f_x, g_x, h_x, t):
    f_x_phi, g_x_phi, h_x_phi = phi(ineq_constraints, x)
    f_x = t * f_x + f_x_phi
    g_x = t * g_x + g_x_phi
    h_x = t * h_x + h_x_phi
    return f_x, g_x, h_x

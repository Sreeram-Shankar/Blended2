import numpy as np

#defines the finite difference Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    n = len(x)
    f0 = fun(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    return J

#solves the nonlinear system of equations
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = y0.copy()
    for _ in range(max_iter):
        r = residual(y)
        if np.linalg.norm(r) < tol:
            return y
        J = jac(y) if jac else finite_diff_jac(residual, y)
        dy = np.linalg.solve(J, -r)
        y += dy
        if np.linalg.norm(dy) < tol:
            break
    return y

#defines the gl1 Jacobian and residual
def gl1_residual_and_jac(f, t, y, h):
    t_mid = t + 0.5 * h

    def R_gl1(y_next):
        y_mid = 0.5 * (y + y_next)
        return y_next - y - h * f(t_mid, y_mid)

    def J_gl1(y_next):
        y_mid = 0.5 * (y + y_next)
        Jf = finite_diff_jac(lambda z: f(t_mid, z), y_mid)
        return np.eye(len(y)) - 0.5 * h * Jf

    return R_gl1, J_gl1

#defines the bdf2 Jacobian and residuel
def bdf2_residual_and_jac(f, t, y, y_prev, h):
    t_next = t + h

    def R_bdf2(y_next):
        return (3 * y_next - 4 * y + y_prev) / (2 * h) - f(t_next, y_next)

    def J_bdf2(y_next):
        Jf = finite_diff_jac(lambda z: f(t_next, z), y_next)
        return (3 / (2 * h)) * np.eye(len(y)) - Jf

    return R_bdf2, J_bdf2

#defines the stiffness proxy
def stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14):
    num = np.linalg.norm(f_n - f_prev)
    den = max(np.linalg.norm(y_n - y_prev), eps)
    return h * num / den

#defines the adaptive weight parameter a
def adapt_a(sigma, p=1.5, a_min=0, a_max=1):
    a = (sigma**p) / (1 + sigma**p)
    return float(np.clip(a, a_min, a_max))

#the main solver for the blended method
def solve_blended_fixed(f, t_span, y0, h, p=1.5, a_min=0, a_max=1):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0

    #the first step is pure implicit midpoint
    f_prev = f(t0, y0)
    R_gl1, J_gl1 = gl1_residual_and_jac(f, t0, y0, h)
    y1 = newton_solve(R_gl1, y0.copy(), J_gl1)
    Y[1] = y1
    a_hist = [0.0, 0.0]

    for n in range(1, N):
        t = t_grid[n]
        y = Y[n]
        y_prev = Y[n - 1]

        f_n = f(t, y)
        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        #builds the individual residuals and Jacobians
        R_gl1, J_gl1 = gl1_residual_and_jac(f, t, y, h)
        R_bdf2, J_bdf2 = bdf2_residual_and_jac(f, t, y, y_prev, h)

        #blends the residual
        def R_blend(y_next): return a * R_bdf2(y_next) + (1 - a) * R_gl1(y_next)

        #blends the Jacobian
        def J_blend(y_next): return a * J_bdf2(y_next) + (1 - a) * J_gl1(y_next)

        #solve the nonlinear system for blended residual
        y_guess = Y[n]
        y_next = newton_solve(R_blend, y_guess, J_blend)

        Y[n + 1] = y_next
        a_hist.append(a)
        f_prev = f_n
    return t_grid, Y, np.array(a_hist)
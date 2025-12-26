import numpy as np

#computes the finite difference Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    n = len(x)
    f0 = np.asarray(fun(x), dtype=float)
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        dx = np.zeros(n, dtype=float)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = np.asarray(fun(x + dx), dtype=float)
        J[:, j] = (f1 - f0) / step
    return J

#solves the nonlinear system of equations
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = np.asarray(y0, dtype=float).copy()
    for it in range(1, max_iter + 1):
        #performs LU decomposition to solve the linear system
        r = np.asarray(residual(y), dtype=float)
        nr = np.linalg.norm(r)
        if nr < tol: return y, True, it

        #computes the Jacobian
        J = jac(y) if jac is not None else finite_diff_jac(residual, y)
        try: dy = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError: return y, False, it

        #updates the solution
        y = y + dy
        if np.linalg.norm(dy) < tol: return y, True, it
    return y, False, max_iter

#defines the gl1 Jacobian and residual
def gl1_residual_and_jac(f, t, y, h, jac_eps=1e-8):
    t_mid = t + 0.5 * h

    def R_gl1(y_next):
        y_mid = 0.5 * (y + y_next)
        return y_next - y - h * f(t_mid, y_mid)

    def J_gl1(y_next):
        y_mid = 0.5 * (y + y_next)
        Jf = finite_diff_jac(lambda z: f(t_mid, z), y_mid, eps=jac_eps)
        return np.eye(len(y), dtype=float) - 0.5 * h * Jf

    return R_gl1, J_gl1

#defines the bdf2 Jacobian and residuel
def bdf2_residual_and_jac(f, t, y, y_prev, h, jac_eps=1e-8):
    t_next = t + h

    def R_bdf2(y_next):
        return (3.0 * y_next - 4.0 * y + y_prev) / (2.0 * h) - f(t_next, y_next)

    def J_bdf2(y_next):
        Jf = finite_diff_jac(lambda z: f(t_next, z), y_next, eps=jac_eps)
        return (3.0 / (2.0 * h)) * np.eye(len(y), dtype=float) - Jf

    return R_bdf2, J_bdf2

#defines the stiffness proxy
def stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14):
    num = np.linalg.norm(f_n - f_prev)
    den = max(np.linalg.norm(y_n - y_prev), eps)
    return h * num / den

#defines the adaptive weight parameter a
def adapt_a(sigma, p=1.5, a_min=0, a_max=1):
    s = float(max(sigma, 0.0))
    a = (s**p) / (1.0 + s**p)
    return float(np.clip(a, a_min, a_max))

#computes the second order derivative estimate using the directional derivative of the Jacobian
def direction_second_derivative(f, t, y, eps=1e-6):
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(f(t, y), dtype=float)
    nf = np.linalg.norm(f0)
    if nf < 1e-14: return np.zeros_like(y), f0

    #computes the direction vector
    v = f0 / nf

    #computes the step size
    delta = eps * nf
    f1 = np.asarray(f(t, y + delta * v), dtype=float)

    #computes the second derivative estimate
    y_ddot = (f1 - f0) / max(delta, 1e-30)
    return y_ddot, f0

#computes the weighted RMS norm of the error vector
def wrms_norm(err, y, y_new, atol, rtol):
    y = np.asarray(y, dtype=float)
    y_new = np.asarray(y_new, dtype=float)
    err = np.asarray(err, dtype=float)
    d = len(y)

    #converts the absolute tolerance to an array
    atol = np.asarray(atol, dtype=float)
    if atol.size == 1: atol = np.full(d, float(atol), dtype=float)

    #computes the scale factor for the error vector
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    scale = np.maximum(scale, 1e-30)
    return float(np.sqrt(np.mean((err / scale) ** 2)))


#proposes the next step size from the curvature defect
def h_proposal(h, E, order=2, safety=0.9, growth=2.0, shrink=0.2,h_min=1e-12, h_max=1e2):
    if E <= 0.0: h_new = growth * h
    else: h_new = h * safety * (1.0 / E) ** (1.0 / order)
    h_new = min(max(h_new, shrink * h), growth * h)
    h_new = min(max(h_new, h_min), h_max)
    return h_new


#the main solver for the blended method
def solve_blended2_adaptive(f, t_span, y0, h0, atol=1e-6, rtol=1e-3, p=1.5, a_min=0.05, a_max=0.98, newton_tol=1e-10, newton_max_iter=12, jac_eps=1e-8, curv_eps=1e-6, safety=0.9, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1, max_reject=20, max_steps=None):
    #initializes the solver
    t0, tf = float(t_span[0]), float(t_span[1])
    y = np.asarray(y0, dtype=float).copy()
    t = t0
    h = float(h0)
    T = [t]
    Y = [y.copy()]
    a_hist = [0.0]
    sigma_hist = [0.0]
    wrms_hist = [0.0]
    h_hist = [h]
    newton_iter_hist = [0]
    reject_hist = [0]
    reject_newton = 0
    reject_curvature = 0
    accept_count = 0
    step_count = 0

    #helper function to compute the curvature defect
    def curvature_defect_wrms(t_here, y_here, y_new_guess, h_here):
        #computes the curvature defect using the second order derivative estimate
        y_ddot, _ = direction_second_derivative(f, t_here, y_here, eps=curv_eps)

        #defines the error vector as the curvature defect
        err_vec = 0.5 * (h_here ** 2) * y_ddot
        return wrms_norm(err_vec, y_here, y_new_guess, atol=atol, rtol=rtol)

    #bootstraps the solver with the first step using the GL1 method
    f_prev = np.asarray(f(t, y), dtype=float)

    #clips the initial step size
    h = min(h, tf - t) if tf > t else h
    if h <= 0:
        info = {"a": np.array(a_hist), "sigma": np.array(sigma_hist), "E_wrms": np.array(wrms_hist), "h": np.array(h_hist), "newton_iters": np.array(newton_iter_hist), "rejects": np.array(reject_hist)}
        return np.array(T), np.array(Y), info

    #solves the initial step using the GL1 method
    R0, J0 = gl1_residual_and_jac(f, t, y, h, jac_eps=jac_eps)
    y1, ok, iters = newton_solve(R0, y.copy(), J0, tol=newton_tol, max_iter=newton_max_iter)
    if not ok: raise RuntimeError("Newton failed on the initial GL1 startup step.")

    #accepts the initial step
    y_prev = y.copy()
    y = y1.copy()
    t = t + h

    #computes the curvature defect at the accepted state to set the next step size
    E = curvature_defect_wrms(t - h, y_prev, y, h)
    h_next = h_proposal(h, E, order=2, safety=safety, growth=growth, shrink=shrink, h_min=h_min, h_max=h_max)

    #records the initial step
    T.append(t)
    Y.append(y.copy())
    a_hist.append(0.0)
    sigma_hist.append(0.0)
    wrms_hist.append(E)
    h_hist.append(h)
    newton_iter_hist.append(iters)
    reject_hist.append(0)
    h = h_next

    #the loop that runs the solver till tim reaches the final time
    while t < tf:
        if max_steps is not None and len(T) >= max_steps: break
        h = min(h, tf - t)
        if h < h_min: break
        rejects = 0
        accepted = False

        while not accepted:
            f_n = np.asarray(f(t, y), dtype=float)

            #computes the stiffness blending weight
            sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
            a = adapt_a(sigma, p=p, a_min=a_min, a_max=a_max)

            #builds the individual method Jacobian and residuals
            R_gl1, J_gl1 = gl1_residual_and_jac(f, t, y, h, jac_eps=jac_eps)
            R_bdf2, J_bdf2 = bdf2_residual_and_jac(f, t, y, y_prev, h, jac_eps=jac_eps)

            #builds the blended residual and Jacobian
            def R_blend(y_next): return a * R_bdf2(y_next) + (1.0 - a) * R_gl1(y_next)
            def J_blend(y_next): return a * J_bdf2(y_next) + (1.0 - a) * J_gl1(y_next)

            #solves the nonlinear system for the blended residual
            y_next, ok, iters = newton_solve(R_blend, y.copy(), J_blend, tol=newton_tol, max_iter=newton_max_iter)

            if ok:
                #computes the curvature defect
                E = curvature_defect_wrms(t, y, y_next, h)

                #accepts the step if the curvature defect is small
                if E <= 1.0:
                    #accepts the step
                    t_next = t + h
                    accept_count += 1

                    #chooses the next step size from the curvature defect
                    h_new = h_proposal(h, E, order=2, safety=safety, growth=growth, shrink=shrink, h_min=h_min, h_max=h_max)

                    #updates the state, history, and step size
                    y_prev = y.copy()
                    y = y_next.copy()
                    t = t_next
                    f_prev = f_n
                    T.append(t)
                    Y.append(y.copy())
                    a_hist.append(a)
                    sigma_hist.append(sigma)
                    wrms_hist.append(E)
                    h_hist.append(h)
                    newton_iter_hist.append(iters)
                    h = h_new
                    accepted = True
                #rejects the step if the curvature defect is large
                else:
                    reject_curvature += 1
                    h = max(shrink * h, h_min)
                    reject_hist.append(rejects)
                    accepted = False
                    continue
            else:
                #rejects the step if Newton fails
                reject_newton += 1
                rejects += 1
                if rejects > max_reject: raise RuntimeError(f"Newton repeatedly failed at t={t:.6e}. Last h={h:.3e}, rejects={rejects}.")
                h = max(shrink * h, h_min)
    info = {"a": np.array(a_hist), "sigma": np.array(sigma_hist), "E_wrms": np.array(wrms_hist), "h": np.array(h_hist), "newton_iters": np.array(newton_iter_hist), "rejects": np.array(reject_hist)}
    return np.array(T), np.array(Y), info
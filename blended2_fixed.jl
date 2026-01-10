using LinearAlgebra

#defines the finite difference Jacobian
function finite_diff_jac(fun, x, eps=1e-8)
    n = length(x)
    f0 = fun(x)
    J = zeros(n, n)
    for j in 1:n
        dx = zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    end
    return J
end

#solves the nonlinear system of equations
function newton_solve(residual, y0, jac=nothing, tol=1e-10, max_iter=12)
    y = copy(y0)
    for _ in 1:max_iter
        r = residual(y)
        if norm(r) < tol
            return y
        end
        J = jac !== nothing ? jac(y) : finite_diff_jac(residual, y)
        dy = J \ (-r)
        y += dy
        if norm(dy) < tol
            break
        end
    end
    return y
end

#defines the gl1 Jacobian and residual
function gl1_residual_and_jac(f, t, y, h)
    t_mid = t + 0.5 * h

    function R_gl1(y_next)
        y_mid = 0.5 * (y + y_next)
        return y_next - y - h * f(t_mid, y_mid)
    end

    function J_gl1(y_next)
        y_mid = 0.5 * (y + y_next)
        Jf = finite_diff_jac(z -> f(t_mid, z), y_mid)
        return Matrix{Float64}(I, length(y), length(y)) - 0.5 * h * Jf
    end

    return R_gl1, J_gl1
end

#defines the bdf2 Jacobian and residuel
function bdf2_residual_and_jac(f, t, y, y_prev, h)
    t_next = t + h

    function R_bdf2(y_next)
        return (3 * y_next - 4 * y + y_prev) / (2 * h) - f(t_next, y_next)
    end

    function J_bdf2(y_next)
        Jf = finite_diff_jac(z -> f(t_next, z), y_next)
        return (3 / (2 * h)) * Matrix{Float64}(I, length(y), length(y)) - Jf
    end

    return R_bdf2, J_bdf2
end

#defines the stiffness proxy
function stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14)
    num = norm(f_n - f_prev)
    den = max(norm(y_n - y_prev), eps)
    return h * num / den
end

#defines the adaptive weight parameter a
function adapt_a(sigma, p=1.5, a_min=0, a_max=1)
    a = (sigma^p) / (1 + sigma^p)
    return Float64(clamp(a, a_min, a_max))
end

#the main solver for the blended method
function solve_blended_fixed(f, t_span, y0, h; p=1.5, a_min=0, a_max=1)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = LinRange(t0, tf, N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0

    #the first step is pure implicit midpoint
    f_prev = f(t0, y0)
    R_gl1, J_gl1 = gl1_residual_and_jac(f, t0, y0, h)
    y1 = newton_solve(R_gl1, copy(y0), J_gl1)
    Y[2, :] = y1
    a_hist = [0.0, 0.0]

    for n in 1:(N-1)
        t = t_grid[n + 1]
        y = Y[n + 1, :]
        y_prev = Y[n, :]

        f_n = f(t, y)
        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        #builds the individual residuals and Jacobians
        R_gl1, J_gl1 = gl1_residual_and_jac(f, t, y, h)
        R_bdf2, J_bdf2 = bdf2_residual_and_jac(f, t, y, y_prev, h)

        #blends the residual
        function R_blend(y_next)
            return a * R_bdf2(y_next) + (1 - a) * R_gl1(y_next)
        end

        #blends the Jacobian
        function J_blend(y_next)
            return a * J_bdf2(y_next) + (1 - a) * J_gl1(y_next)
        end

        #solve the nonlinear system for blended residual
        y_guess = Y[n + 1, :]
        y_next = newton_solve(R_blend, y_guess, J_blend)

        Y[n + 2, :] = y_next
        push!(a_hist, a)
        f_prev = f_n
    end
    return collect(t_grid), Y, a_hist
end
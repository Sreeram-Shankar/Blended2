using LinearAlgebra
using Statistics

#computes the finite difference Jacobian
function finite_diff_jac(fun, x, eps=1e-8)
    x = Float64.(x)
    n = length(x)
    f0 = Float64.(fun(x))
    J = zeros(n, n)
    for j in 1:n
        dx = zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = Float64.(fun(x + dx))
        J[:, j] = (f1 - f0) / step
    end
    return J
end

#solves the nonlinear system of equations
function newton_solve(residual, y0, jac=nothing, tol=1e-10, max_iter=12)
    y = Float64.(copy(y0))
    for it in 1:max_iter
        #performs LU decomposition to solve the linear system
        r = Float64.(residual(y))
        nr = norm(r)
        if nr < tol
            return y, true, it
        end

        #computes the Jacobian
        J = jac !== nothing ? jac(y) : finite_diff_jac(residual, y)
        dy = nothing
        try
            dy = J \ (-r)
        catch
            return y, false, it
        end

        #updates the solution
        y = y + dy
        if norm(dy) < tol
            return y, true, it
        end
    end
    return y, false, max_iter
end

#defines the gl1 Jacobian and residual
function gl1_residual_and_jac(f, t, y, h, jac_eps=1e-8)
    t_mid = t + 0.5 * h

    function R_gl1(y_next)
        y_mid = 0.5 * (y + y_next)
        return y_next - y - h * f(t_mid, y_mid)
    end

    function J_gl1(y_next)
        y_mid = 0.5 * (y + y_next)
        Jf = finite_diff_jac(z -> f(t_mid, z), y_mid, jac_eps)
        return Matrix{Float64}(I, length(y), length(y)) - 0.5 * h * Jf
    end

    return R_gl1, J_gl1
end

#defines the bdf2 Jacobian and residuel
function bdf2_residual_and_jac(f, t, y, y_prev, h, jac_eps=1e-8)
    t_next = t + h

    function R_bdf2(y_next)
        return (3.0 * y_next - 4.0 * y + y_prev) / (2.0 * h) - f(t_next, y_next)
    end

    function J_bdf2(y_next)
        Jf = finite_diff_jac(z -> f(t_next, z), y_next, jac_eps)
        return (3.0 / (2.0 * h)) * Matrix{Float64}(I, length(y), length(y)) - Jf
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
    s = Float64(max(sigma, 0.0))
    a = (s^p) / (1.0 + s^p)
    return Float64(clamp(a, a_min, a_max))
end

#computes the second order derivative estimate using the directional derivative of the Jacobian
function direction_second_derivative(f, t, y, eps=1e-6)
    y = Float64.(y)
    f0 = Float64.(f(t, y))
    nf = norm(f0)
    if nf < 1e-14
        return zeros(length(y)), f0
    end

    #computes the direction vector
    v = f0 / nf

    #computes the step size
    delta = eps * nf
    f1 = Float64.(f(t, y + delta * v))

    #computes the second derivative estimate
    y_ddot = (f1 - f0) / max(delta, 1e-30)
    return y_ddot, f0
end

#computes the weighted RMS norm of the error vector
function wrms_norm(err, y, y_new, atol, rtol)
    y = Float64.(y)
    y_new = Float64.(y_new)
    err = Float64.(err)
    d = length(y)

    #converts the absolute tolerance to an array
    if atol isa Number
        atol = fill(Float64(atol), d)
    else
        atol = Float64.(atol)
    end

    #computes the scale factor for the error vector
    scale = atol + rtol * max.(abs.(y), abs.(y_new))
    scale = max.(scale, 1e-30)
    return Float64(sqrt(mean((err ./ scale) .^ 2)))
end


#proposes the next step size from the curvature defect
function h_proposal(h, E, order=2, safety=0.9, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e2)
    if E <= 0.0
        h_new = growth * h
    else
        h_new = h * safety * (1.0 / E) ^ (1.0 / order)
    end
    h_new = min(max(h_new, shrink * h), growth * h)
    h_new = min(max(h_new, h_min), h_max)
    return h_new
end


#the main solver for the blended method
function solve_blended2_adaptive(f, t_span, y0, h0; atol=1e-6, rtol=1e-3, p=1.5, a_min=0.05, a_max=1.0, newton_tol=1e-10, newton_max_iter=12, jac_eps=1e-8, curv_eps=1e-6, safety=0.9, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1, max_reject=20, max_steps=nothing)
    #initializes the solver
    t0, tf = Float64(t_span[1]), Float64(t_span[2])
    y = Float64.(copy(y0))
    t = t0
    h = Float64(h0)
    T = [t]
    Y = [copy(y)]
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
    function curvature_defect_wrms(t_here, y_here, y_new_guess, h_here)
        #computes the curvature defect using the second order derivative estimate
        y_ddot, _ = direction_second_derivative(f, t_here, y_here, curv_eps)

        #defines the error vector as the curvature defect
        err_vec = 0.5 * (h_here ^ 2) * y_ddot
        return wrms_norm(err_vec, y_here, y_new_guess, atol, rtol)
    end

    #bootstraps the solver with the first step using the GL1 method
    f_prev = Float64.(f(t, y))

    #clips the initial step size
    h = tf > t ? min(h, tf - t) : h
    if h <= 0
        info = Dict("a" => a_hist, "sigma" => sigma_hist, "E_wrms" => wrms_hist, "h" => h_hist, "newton_iters" => newton_iter_hist, "rejects" => reject_hist)
        Y_matrix = length(Y) > 0 ? reduce(hcat, Y)' : zeros(0, length(y0))
        return T, Y_matrix, info
    end

    #solves the initial step using the GL1 method
    R0, J0 = gl1_residual_and_jac(f, t, y, h, jac_eps)
    y1, ok, iters = newton_solve(R0, copy(y), J0, newton_tol, newton_max_iter)
    if !ok
        throw(ErrorException("Newton failed on the initial GL1 startup step."))
    end

    #accepts the initial step
    y_prev = copy(y)
    y = copy(y1)
    t = t + h

    #computes the curvature defect at the accepted state to set the next step size
    E = curvature_defect_wrms(t - h, y_prev, y, h)
    h_next = h_proposal(h, E, 2, safety, growth, shrink, h_min, h_max)

    #records the initial step
    push!(T, t)
    push!(Y, copy(y))
    push!(a_hist, 0.0)
    push!(sigma_hist, 0.0)
    push!(wrms_hist, E)
    push!(h_hist, h)
    push!(newton_iter_hist, iters)
    push!(reject_hist, 0)
    h = h_next

    #the loop that runs the solver till tim reaches the final time
    while t < tf
        if max_steps !== nothing && length(T) >= max_steps
            break
        end
        h = min(h, tf - t)
        if h < h_min
            break
        end
        rejects = 0
        accepted = false

        while !accepted
            f_n = Float64.(f(t, y))

            #computes the stiffness blending weight
            sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
            a = adapt_a(sigma, p, a_min, a_max)

            #builds the individual method Jacobian and residuals
            R_gl1, J_gl1 = gl1_residual_and_jac(f, t, y, h, jac_eps)
            R_bdf2, J_bdf2 = bdf2_residual_and_jac(f, t, y, y_prev, h, jac_eps)

            #builds the blended residual and Jacobian
            function R_blend(y_next)
                return a * R_bdf2(y_next) + (1.0 - a) * R_gl1(y_next)
            end
            function J_blend(y_next)
                return a * J_bdf2(y_next) + (1.0 - a) * J_gl1(y_next)
            end

            #solves the nonlinear system for the blended residual
            y_next, ok, iters = newton_solve(R_blend, copy(y), J_blend, newton_tol, newton_max_iter)

            if ok
                #computes the curvature defect
                E = curvature_defect_wrms(t, y, y_next, h)

                #accepts the step if the curvature defect is small
                if E <= 1.0
                    #accepts the step
                    t_next = t + h
                    accept_count += 1

                    #chooses the next step size from the curvature defect
                    h_new = h_proposal(h, E, 2, safety, growth, shrink, h_min, h_max)

                    #updates the state, history, and step size
                    y_prev = copy(y)
                    y = copy(y_next)
                    t = t_next
                    f_prev = f_n
                    push!(T, t)
                    push!(Y, copy(y))
                    push!(a_hist, a)
                    push!(sigma_hist, sigma)
                    push!(wrms_hist, E)
                    push!(h_hist, h)
                    push!(newton_iter_hist, iters)
                    h = h_new
                    accepted = true
                #rejects the step if the curvature defect is large
                else
                    reject_curvature += 1
                    h = max(shrink * h, h_min)
                    push!(reject_hist, rejects)
                    accepted = false
                    continue
                end
            else
                #rejects the step if Newton fails
                reject_newton += 1
                rejects += 1
                if rejects > max_reject
                    throw(ErrorException("Newton repeatedly failed at t=$(t). Last h=$(h), rejects=$(rejects)."))
                end
                h = max(shrink * h, h_min)
            end
        end
    end
    info = Dict("a" => a_hist, "sigma" => sigma_hist, "E_wrms" => wrms_hist, "h" => h_hist, "newton_iters" => newton_iter_hist, "rejects" => reject_hist)
    Y_matrix = reduce(hcat, Y)'
    return T, Y_matrix, info
end
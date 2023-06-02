function _brusselator2d(x::T, y::T, t) where {T}
    return ifelse((((x - T(0.3))^2 + (y - T(0.6))^2) <= T(0.1)^2) && (t >= T(1.1)), T(5),
                  T(0))
end

_brusselator2d_limit(a, N) = ifelse(a == N + 1, 1, ifelse(a == 0, N, a))

function _init_brusselator2d(xyd)
    T = eltype(xyd)
    N = length(xyd)
    u = zeros(T, N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(T(3) / T(2))
        u[I, 2] = 27 * (x * (1 - x))^(T(3) / T(2))
    end
    return u
end

function brusselator_2d_loop!(du, u, p, t; N, xyd_brusselator)
    A, B, alpha, dx = p
    alpha /= dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = (_brusselator2d_limit(i + 1, N),
                              _brusselator2d_limit(i - 1, N),
                              _brusselator2d_limit(j + 1, N),
                              _brusselator2d_limit(j - 1, N))
        du[i, j, 1] = alpha *
                      (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                       4u[i, j, 1]) +
                      B +
                      u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
                      _brusselator2d(x, y, t)
        du[i, j, 2] = alpha *
                      (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                       4u[i, j, 2]) +
                      A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
    return du
end

function brusselator(::Val{2}; xyd_start::T = 0.0f0, xyd_stop::T = 1.0f0, xyd_length = 32,
                     t_start::T = 0.0f0, t_stop::T = 11.5f0, ps = nothing) where {T <: Real}
    xyd_brusselator = range(xyd_start; stop = xyd_stop, length = xyd_length)
    if ps === nothing
        ps = T.((3.4, 1.0, 10.0, step(xyd_brusselator)))
    else
        @assert length(ps) == 4
        @assert step(xyd_brusselator) == ps[4]
        @assert eltype(ps) == T
    end

    function brusselator_func(du, u, p, t)
        brusselator_2d_loop!(du, u, p, t; N = xyd_length, xyd_brusselator)
    end

    u0 = _init_brusselator2d(xyd_brusselator)
    return ODEProblem(brusselator_func, u0, (t_start, t_stop), ps)
end

function brusselator_sparse(::Val{2}; xyd_start::T = 0.0f0, xyd_stop::T = 1.0f0,
                            xyd_length = 32,
                            t_start::T = 0.0f0, t_stop::T = 11.5f0,
                            ps = nothing) where {T <: Real}
    xyd_brusselator = range(xyd_start; stop = xyd_stop, length = xyd_length)
    if ps === nothing
        ps = T.((3.4, 1.0, 10.0, step(xyd_brusselator)))
    else
        @assert length(ps) == 4
        @assert step(xyd_brusselator) == ps[4]
        @assert eltype(ps) == T
    end

    function brusselator_func(du, u, p, t)
        brusselator_2d_loop!(du, u, p, t; N = xyd_length, xyd_brusselator)
    end

    u0 = _init_brusselator2d(xyd_brusselator)

    du0 = copy(u0)
    jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> brusselator_2d_loop!(du, u, ps,
                                                                               0.0;
                                                                               N = xyd_length,
                                                                               xyd_brusselator),
                                               du0, u0)

    f = ODEFunction(brusselator_func; jac_prototype = Float32.(jac_sparsity))

    return ODEProblem(f, u0, (t_start, t_stop), ps)
end

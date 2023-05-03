struct MixedPrecisionLinsolve{T <: Real} <: LinearSolve.AbstractFactorization
    linalg::LinearSolve.AbstractFactorization
    precision::Type{T}
end

MixedPrecisionLinsolve() = MixedPrecisionLinsolve(RFLUFactorization(), Float32)

function SciMLBase.init(prob::LinearProblem, alg::MixedPrecisionLinsolve,
                        args...;
                        kwargs...)
    if prob.u0 !== nothing
        prob = remake(prob; A = alg.precision.(prob.A), b = alg.precision.(prob.b))
    else
        u0 = similar(prob.b, size(prob.A, 2))
        fill!(u0, false)
        prob = remake(prob; A = alg.precision.(prob.A), b = alg.precision.(prob.b), u0 = u0)
    end
    cache = SciMLBase.init(prob, alg.linalg, args...; kwargs...)
    return LinearSolve.@set! cache.alg = alg
end

function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::MixedPrecisionLinsolve; kwargs...)
    SciMLBase.solve(cache, alg.linalg; kwargs...)
end

function LinearSolve.set_A(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                                          Ttol, issq, condition},
                           A) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
                                     Ttol, issq, condition}
    LinearSolve.@set! cache.A = cache.alg.precision.(A)
    LinearSolve.@set! cache.isfresh = true
    return cache
end

function LinearSolve.set_b(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                                          Ttol, issq, condition},
                           b) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
                                     Ttol, issq, condition}
    LinearSolve.@set! cache.b = cache.alg.precision.(b)
    return cache
end

# struct MixedPrecisionLinsolve{T <: Real} <: LinearSolve.AbstractFactorization
#     linalg::LinearSolve.AbstractFactorization
#     precision::Type{T}
# end

# MixedPrecisionLinsolve() = MixedPrecisionLinsolve(RFLUFactorization(), Float32)

# function LinearSolve.init_cacheval(alg::MixedPrecisionLinsolve, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
#                             verbose::Bool, assumptions::LinearSolve.OperatorAssumptions)
#     LinearSolve.init_cacheval(alg.linalg, alg.precision.(A), alg.precision.(b), alg.precision.(u), Pl, Pr, maxiters::Int, abstol, reltol,
#     verbose::Bool, assumptions::LinearSolve.OperatorAssumptions)
# end

# function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::MixedPrecisionLinsolve; kwargs...)
#     A = cache.A
#     A = convert(AbstractMatrix, A)
#     fact, ipiv = cache.cacheval
#     if cache.isfresh
#         fact = LinearSolve.RecursiveFactorization.lu!(alg.precision.(A), ipiv, Val(true), Val(true))
#         cache = LinearSolve.set_cacheval(cache, (fact, ipiv))
#     end
#     y = ldiv!(cache.u, cache.cacheval[1], alg.precision.(cache.b))
#     SciMLBase.build_linear_solution(alg, y, nothing, cache)
# end

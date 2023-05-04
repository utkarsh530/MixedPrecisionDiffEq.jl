struct MixedPrecisionLinsolve{T <: Real} <: LinearSolve.AbstractFactorization
    linalg::LinearSolve.AbstractFactorization
    precision::Type{T}
end

MixedPrecisionLinsolve() = MixedPrecisionLinsolve(RFLUFactorization(), Float32)

# function SciMLBase.init(prob::LinearProblem, alg::MixedPrecisionLinsolve, args...;
#                         alias_A = LinearSolve.default_alias_A(alg, prob.A, prob.b),
#                         alias_b = LinearSolve.default_alias_b(alg, prob.A, prob.b),
#                         abstol = LinearSolve.default_tol(eltype(prob.A)),
#                         reltol = LinearSolve.default_tol(eltype(prob.A)),
#                         maxiters::Int = length(prob.b),
#                         verbose::Bool = false,
#                         Pl = LinearSolve.IdentityOperator(size(prob.A, 1)),
#                         Pr = LinearSolve.IdentityOperator(size(prob.A, 2)),
#                         assumptions = LinearSolve.OperatorAssumptions(Val(issquare(prob.A))),
#                         kwargs...)

#     # if prob.u0 !== nothing
#     #     prob = remake(prob; A = alg.precision.(prob.A), b = alg.precision.(prob.b))
#     # else
#     #     u0 = similar(prob.b, size(prob.A, 2))
#     #     fill!(u0, false)
#     #     prob = remake(prob; A = alg.precision.(prob.A), b = alg.precision.(prob.b), u0 = u0)
#     # end
#     LinearSolve.@unpack A, b, u0, p = prob
#     A = alias_A ? A : deepcopy(A)
#     b = if b isa LinearSolve.SparseArrays.AbstractSparseArray && !(A isa Diagonal)
#         Array(b) # the solution to a linear solve will always be dense!
#     elseif alias_b
#         b
#     else
#         deepcopy(b)
#     end

#     u0 = if u0 !== nothing
#         u0
#     else
#         u0 = similar(b, size(A, 2))
#         fill!(u0, false)
#     end

#     cacheval = LinearSolve.init_cacheval(alg.linalg, A, b, u0, Pl, Pr, maxiters, abstol, reltol, verbose,
#                                 assumptions)
#     isfresh = true
#     Tc = typeof(cacheval)

#     cache = LinearSolve.LinearCache{
#                         typeof(A),
#                         typeof(b),
#                         typeof(u0),
#                         typeof(p),
#                         typeof(alg),
#                         Tc,
#                         typeof(Pl),
#                         typeof(Pr),
#                         typeof(reltol),
#                         LinearSolve.__issquare(assumptions),
#                         LinearSolve.__conditioning(assumptions)
#                         }(A,
#                             b,
#                             u0,
#                             p,
#                             alg,
#                             cacheval,
#                             isfresh,
#                             Pl,
#                             Pr,
#                             abstol,
#                             reltol,
#                             maxiters,
#                             verbose,
#                             assumptions)
#     return cache
# end

# function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::MixedPrecisionLinsolve; kwargs...)
#     SciMLBase.solve(cache, alg.linalg; kwargs...)
# end

# function LinearSolve.set_A(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
#                                                           Ttol, issq, condition},
#                            A) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
#                                      Ttol, issq, condition}
#     cache.A .= (A)
#     LinearSolve.@set! cache.isfresh = true
#     return cache
# end

# function LinearSolve.set_b(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
#                                                           Ttol, issq, condition},
#                            b) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
#                                      Ttol, issq, condition}
#     cache.b .= (b)
#     return cache
# end

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

struct MixedPrecisionLinearCache{Tlc, TA, Tb}
    linearcache::Tlc
    Arp::TA
    brp::Tb
end

function SciMLBase.init(prob::LinearSolve.LinearProblem, alg::MixedPrecisionLinsolve,
                        args...;
                        alias_A = LinearSolve.default_alias_A(alg, prob.A, prob.b),
                        alias_b = LinearSolve.default_alias_b(alg, prob.A, prob.b),
                        abstol = LinearSolve.default_tol(eltype(prob.A)),
                        reltol = LinearSolve.default_tol(eltype(prob.A)),
                        maxiters::Int = length(prob.b),
                        verbose::Bool = false,
                        Pl = LinearSolve.IdentityOperator(size(prob.A, 1)),
                        Pr = LinearSolve.IdentityOperator(size(prob.A, 2)),
                        assumptions = LinearSolve.OperatorAssumptions(Val(issquare(prob.A))),
                        kwargs...)
    LinearSolve.@unpack A, b, u0, p = prob

    A = alias_A ? A : deepcopy(A)
    b = if b isa LinearSolve.SparseArrays.AbstractSparseArray && !(A isa Diagonal)
        Array(b) # the solution to a linear solve will always be dense!
    elseif alias_b
        b
    else
        deepcopy(b)
    end

    u0 = if u0 !== nothing
        u0
    else
        u0 = similar(b, size(A, 2))
        fill!(u0, false)
    end

    # A = alg.precision.(A)
    # b = alg.precision.(b)

    cacheval = LinearSolve.init_cacheval(alg.linalg, A, b, u0, Pl, Pr, maxiters, abstol,
                                         reltol, verbose,
                                         assumptions)
    isfresh = true
    Tc = typeof(cacheval)

    cache = LinearSolve.LinearCache{
                                    typeof(A),
                                    typeof(b),
                                    typeof(u0),
                                    typeof(p),
                                    typeof(alg.linalg),
                                    Tc,
                                    typeof(Pl),
                                    typeof(Pr),
                                    typeof(reltol),
                                    LinearSolve.__issquare(assumptions),
                                    LinearSolve.__conditioning(assumptions)
                                    }(A,
                                      b,
                                      u0,
                                      p,
                                      alg.linalg,
                                      cacheval,
                                      isfresh,
                                      Pl,
                                      Pr,
                                      abstol,
                                      reltol,
                                      maxiters,
                                      verbose,
                                      assumptions)
    return cache
end

function set_A(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                              Ttol, issq, condition},
               A) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
                         Ttol, issq, condition}
    cache.A .= A
    @set! cache.isfresh = true
    return cache
end

function set_b(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                              Ttol, issq, condition},
               b) where {TA, Tb, Tu, Tp, Talg <: MixedPrecisionLinsolve, Tc, Tl, Tr,
                         Ttol, issq, condition}
    cache.b .= b
    return cache
end

function SciMLBase.solve(cache::MixedPrecisionLinearCache, alg::MixedPrecisionLinsolve;
                         kwargs...)
    SciMLBase.solve(cache, alg.linalg; kwargs...)
end

# function SciMLBase.solve(cache::MixedPrecisionLinearCache, args...; kwargs...)
#     solve(cache, cache.alg, args...; kwargs...)
# end

# function SciMLBase.solve(cache::MixedPrecisionLinearCache, alg::MixedPrecisionLinsolve; kwargs...)
#     A = cache.linearcache.A
#     b = cache.linearcache.b
#     A = convert(AbstractMatrix, A)
#     fact, ipiv = cache.linearcache.cacheval
#     ## copy
#     cache.Arp .= A
#     cache.brp .= b
#     if cache.linearcache.isfresh
#         fact = LinearSolve.RecursiveFactorization.lu!(cache.Arp, ipiv, Val(true), Val(true))
#         LinearSolve.@set! cache.linearcache = LinearSolve.set_cacheval(cache.linearcache, (fact, ipiv))
#     end
#     y = ldiv!(cache.linearcache.u, cache.linearcache.cacheval[1], cache.brp)
#     SciMLBase.build_linear_solution(alg, y, nothing, cache)
# end

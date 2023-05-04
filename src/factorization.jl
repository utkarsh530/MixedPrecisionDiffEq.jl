struct MixedRFLUFactorization{P, T} <: LinearSolve.AbstractFactorization
    MixedRFLUFactorization(::Val{P}, ::Val{T}) where {P, T} = new{P, T}()
end

function MixedRFLUFactorization(; pivot = Val(true), thread = Val(true))
    MixedRFLUFactorization(pivot, thread)
end

function SciMLBase.init(prob::LinearSolve.LinearProblem, alg::MixedRFLUFactorization,
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

    A = Float32.(A)
    b = Float32.(b)

    cacheval = LinearSolve.init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol, reltol,
                                         verbose,
                                         assumptions)
    isfresh = true
    Tc = typeof(cacheval)

    cache = LinearSolve.LinearCache{
                                    typeof(A),
                                    typeof(b),
                                    typeof(u0),
                                    typeof(p),
                                    typeof(alg),
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
                                      alg,
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

function LinearSolve.init_cacheval(alg::MixedRFLUFactorization, A, b, u, Pl, Pr,
                                   maxiters::Int,
                                   abstol, reltol, verbose::Bool,
                                   assumptions::OperatorAssumptions)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    LinearSolve.ArrayInterface.lu_instance(convert(AbstractMatrix, A)), ipiv
end

function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::MixedRFLUFactorization{P, T};
                         kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = cache.cacheval
    if cache.isfresh
        fact = LinearSolve.RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T))
        cache = LinearSolve.set_cacheval(cache, (fact, ipiv))
    end
    y = ldiv!(cache.u, cache.cacheval[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function set_A(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                              Ttol, issq, condition},
               A) where {TA, Tb, Tu, Tp, Talg <: MixedRFLUFactorization, Tc, Tl, Tr,
                         Ttol, issq, condition}
    cache.A .= A
    @set! cache.isfresh = true
    return cache
end

function set_b(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                              Ttol, issq, condition},
               b) where {TA, Tb, Tu, Tp, Talg <: MixedRFLUFactorization, Tc, Tl, Tr,
                         Ttol, issq, condition}
    cache.b .= b
    return cache
end

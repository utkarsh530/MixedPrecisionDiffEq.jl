struct MixedPrecisionCudaOffloadFactorization{T <: Real} <:
       LinearSolve.AbstractFactorization
    precision::Type{T}
end

MixedPrecisionCudaOffloadFactorization() = MixedPrecisionCudaOffloadFactorization(Float32)

function SciMLBase.init(prob::LinearSolve.LinearProblem,
                        alg::MixedPrecisionCudaOffloadFactorization,
                        args...;
                        alias_A = LinearSolve.default_alias_A(alg, prob.A, prob.b),
                        alias_b = LinearSolve.default_alias_b(alg, prob.A, prob.b),
                        abstol = Float64(LinearSolve.default_tol(eltype(prob.A))),
                        reltol = Float64(LinearSolve.default_tol(eltype(prob.A))),
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
    A = CUDA.CuArray{alg.precision}(A)
    b = CUDA.CuArray{alg.precision}(b)

    cacheval = LinearSolve.init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol,
                                         reltol, verbose,
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

function LinearSolve.set_A(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                                          Ttol, issq, condition},
                           A) where {TA, Tb, Tu, Tp,
                                     Talg <: MixedPrecisionCudaOffloadFactorization, Tc, Tl,
                                     Tr,
                                     Ttol, issq, condition}
    #cache.A .= A
    copyto!(cache.A, A)
    LinearSolve.@set! cache.isfresh = true
    return cache
end

function LinearSolve.set_b(cache::LinearSolve.LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr,
                                                          Ttol, issq, condition},
                           b) where {TA, Tb, Tu, Tp,
                                     Talg <: MixedPrecisionCudaOffloadFactorization, Tc, Tl,
                                     Tr,
                                     Ttol, issq, condition}
    # cache.b .= b
    copyto!(cache.b, b)
    return cache
end

function SciMLBase.solve(cache::LinearSolve.LinearCache,
                         alg::MixedPrecisionCudaOffloadFactorization;
                         kwargs...)
    if cache.isfresh
        fact = LinearSolve.do_factorization(alg, cache.A, cache.b, cache.u)
        cache = LinearSolve.set_cacheval(cache, fact)
    end

    # copyto!(cache.u, cache.b)
    cache.u .= Array(ldiv!(cache.cacheval, cache.b))
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

function LinearSolve.do_factorization(alg::MixedPrecisionCudaOffloadFactorization, A, b, u)
    A isa Union{AbstractMatrix, SciMLBase.AbstractSciMLOperator} ||
        error("LU is not defined for $(typeof(A))")

    if A isa Union{MatrixOperator, DiffEqArrayOperator}
        A = A.A
    end

    fact = qr(A)
    return fact
end

# function LinearSolve.init_cacheval(alg::MixedPrecisionCudaOffloadFactorization, A, b, u, Pl, Pr, maxiters::Int,
#                        abstol, reltol, verbose::Bool, assumptions::LinearSolve.OperatorAssumptions)
#     ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
#     lu_instance(convert(AbstractMatrix, A))
# end

# function lu_instance(A::CUDA.CuMatrix{T}) where {T}
#     noUnitT = typeof(zero(T))
#     luT = LinearAlgebra.lutype(noUnitT)
#     ipiv = CUDA.CuArray{Int32}(undef, 0)
#     info = zero(LinearAlgebra.BlasInt)
#     return LU{luT}(similar(A,1,1), ipiv, info)
# end

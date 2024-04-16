using MixedPrecisionDiffEq, OrdinaryDiffEq

include("./bin/problems/ode/brusselator.jl")
prob_float64 = brusselator(Val(2); xyd_start = 0.0, xyd_stop = 1.0, xyd_length = 32,
                           t_start = 0.0,
                           t_stop = 11.0)

prob_float32 = brusselator(Val(2); xyd_start = 0.0f0, xyd_stop = 1.0f0, xyd_length = 2,
                           t_start = 0.0f0,
                           t_stop = 11.0f0)

using LinearSolve, MixedPrecisionDiffEq, Random

Random.seed!(1234)

A = rand(100, 100)
b = rand(100)

AFloat32 = Float32.(A)
bFloat32 = Float32.(b)

prob = LinearProblem(A, b)

prob_f32 = LinearProblem(AFloat32, bFloat32)

linsolve = init(prob, MixedPrecisionLinsolve())
sol = solve(prob, MixedPrecisionLinsolve())

sol = solve(prob_float64, TRBDF2(; linsolve = MixedPrecisionCudaOffloadFactorization()))

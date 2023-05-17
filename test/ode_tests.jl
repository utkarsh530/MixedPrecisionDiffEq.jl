using MixedPrecisionDiffEq, OrdinaryDiffEq, LinearSolve

include(joinpath(dirname(@__DIR__), "bin", "problems", "ode", "brusselator.jl"))

prob_brusselator = brusselator(Val(2); xyd_start = 0.0, xyd_stop = 1.0, xyd_length = 16,
                               t_start = 0.0,
                               t_stop = 11.0)

linalgs = [RFLUFactorization(), FastLUFactorization(), LUFactorization()]

@info "Float32 test with Mixed Precision Linear Solvers"

osol = solve(prob_brusselator, TRBDF2(; linsolve = RFLUFactorization()))

osol = solve(prob_brusselator, TRBDF2(; linsolve = MixedRFLUFactorization()))


@show osol.stats.nnonliniter
for linalg in linalgs
    @info linalg
    linsolve = MixedPrecisionLinsolve(linalg)
    sol = solve(prob_brusselator, TRBDF2(; linsolve = linsolve))
    @show sol.stats.nnonliniter
    @test sol.retcode == SciMLBase.ReturnCode.Success
end

@info "Float16 test with Mixed Precision Linear Solvers"
for linalg in [RFLUFactorization(), LUFactorization()]
    @info linalg
    linsolve = MixedPrecisionLinsolve(linalg, Float16)
    sol = solve(prob_brusselator, TRBDF2(; linsolve = linsolve))
    @show sol.stats.nnonliniter
    @test sol.retcode == SciMLBase.ReturnCode.Success
end




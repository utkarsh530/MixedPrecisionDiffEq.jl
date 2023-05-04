using BenchmarkTools
using MixedPrecisionDiffEq, LinearSolve, OrdinaryDiffEq

include(joinpath(dirname(@__DIR__), "bin", "problems", "brusselator.jl"))

@show Threads.nthreads()

Ns = 22:4:40

for N in Ns
    prob = brusselator(Val(2); xyd_start = 0.0, xyd_stop = 1.0, xyd_length = N,
                       t_start = 0.0,
                       t_stop = 11.0)
    sim_time_mp = @belapsed solve($prob,
                                  TRBDF2(;
                                         linsolve = MixedPrecisionLinsolve(RFLUFactorization())),
                                  save_everystep = false, dense = false)
    sim_time_orig = @belapsed solve($prob, TRBDF2(; linsolve = RFLUFactorization()),
                                    save_everystep = false, dense = false)
    @show sim_time_mp
    @show sim_time_orig
    println("Speed-up: ", sim_time_orig / sim_time_mp)
end

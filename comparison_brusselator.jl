using NonlinearSolve, MixedPrecisionDiffEq, LinearSolve

using SciMLNLSolve

using MixedPrecisionDiffEq, OrdinaryDiffEq

include("./bin/problems/ode/brusselator.jl")

# prob_float64 = brusselator(Val(2); xyd_start = 0.0, xyd_stop = 1.0, xyd_length = 32,
#                            t_start = 0.0,
#                            t_stop = 11.0)

# Ns = [i for i in 2:4:50]

# times_f32_gpu = Float64[]
# times_f64 = Float64[]

# for N in Ns
#     @show N
#     prob_float64 = brusselator(Val(2); xyd_start = 0.0, xyd_stop = 1.0, xyd_length = N,
#                            t_start = 0.0,
#                            t_stop = 11.0)

#     solve(prob_float64, TRBDF2(;linsolve = MixedPrecisionCudaOffloadFactorization()), save_everystep = false, dense = false)
#     benchmark_timef32_gpu = @elapsed solve(prob_float64, TRBDF2(;linsolve = MixedPrecisionCudaOffloadFactorization()), save_everystep = false, dense = false)

#     solve(prob_float64, TRBDF2(;linsolve = MixedPrecisionLinsolve()), save_everystep = false, dense = false)
#     benchmark_timef32_gpu = @elapsed solve(prob_float64, TRBDF2(;linsolve = MixedPrecisionLinsolve()), save_everystep = false, dense = false)

#     solve(prob_float64, TRBDF2(;linsolve = RFLUFactorization()), save_everystep = false, dense = false)
#     benchmark_timef64 = @elapsed solve(prob_float64, TRBDF2(;linsolve = RFLUFactorization()), save_everystep = false, dense = false)

#     push!(times_f64, benchmark_timef64)
#     push!(times_f32, benchmark_timef32)
#     push!(times_f32_gpu, benchmark_timef32_gpu)
# end

using Plots

using JLD2

data = load("times_bruss.jld2")

times_f32 = data["times_f32"]
times_f64 = data["times_f64"]
times_f32_gpu = data["times_f32_gpu"]

ytick = 10 .^ round.(range(2, -5, length = 15), digits = 2)

xtick = 0:500:5000

plot(2Ns .^ 2, times_f64, linewidth = 3, marker = :circle, yaxis = :log, xticks = xtick,
     yticks = ytick,
     label = "Double Precision", xlabel = "N (Size of the problem)",
     ylabel = "Time (s)", legend = :topleft, dpi = 600)

plot!(2Ns .^ 2, times_f32, linewidth = 3, marker = :circle, yaxis = :log,
      label = "Mixed Precision")

savefig("times_vs_N_bruss.png")

plot!(2Ns .^ 2, times_f32_gpu, linewidth = 3, marker = :circle, yaxis = :log,
      label = "GPU-Mixed Precision")

savefig("times_vs_N_bruss_gpu.png")

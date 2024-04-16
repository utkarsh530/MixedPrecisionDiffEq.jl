using NonlinearSolve, MixedPrecisionDiffEq, LinearSolve

using SciMLNLSolve

include(joinpath(@__DIR__, "bin", "problems", "non_linear", "chandrashekhar.jl"))

function custom_linsolve(x, A, b)
    copyto!(x, solve(LinearProblem(A, b), MixedPrecisionLinsolve(RFLUFactorization())).u)
end

Ns = [2^i for i in 1:10]

iters_f32 = Int[]
iters_f64 = Int[]

res_f32 = Float64[]
res_f64 = Float64[]

for N in Ns
    @show N
    x_start = ones(N)
    prob_chandrashekhar = NonlinearProblem(p23_f!, x_start)
    sol = solve(prob_chandrashekhar, NLSolveJL(; method = :newton), abstol = 1e-15)
    solnl = solve(prob_chandrashekhar,
                  NLSolveJL(; method = :newton, linsolve = custom_linsolve), abstol = 1e-15)
    push!(iters_f64, sol.original.iterations)
    push!(iters_f32, solnl.original.iterations)
    push!(res_f64, sol.original.residual_norm)
    push!(res_f32, solnl.original.residual_norm)
end

using Plots

plot(Ns, iters_f64, linewidth = 4, marker = :circle, xaxis = :log2, xticks = Ns,
     yticks = 4:10,
     label = "Double Precision", xlabel = "N (Size of the problem)",
     ylabel = "Iteration count")
plot!(Ns, iters_f32, linewidth = 3, ls = :dash, marker = :circle, xaxis = :log2,
      xticks = Ns, label = "Mixed Precision")

savefig("iteration_vs_N.png")

plot(Ns, res_f64, linewidth = 4, marker = :circle, xaxis = :log2, xticks = Ns,
     label = "Double Precision", xlabel = "N (Size of the problem)",
     ylabel = "Residue")
plot!(Ns, res_f32, linewidth = 3, ls = :dash, marker = :circle, xaxis = :log2, xticks = Ns,
      label = "Mixed Precision")

savefig("residue_vs_N.png")

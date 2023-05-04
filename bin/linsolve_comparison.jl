using Random, BenchmarkTools

# Random.seed!(1234)

# function offloadf32_solve(A,b)
#     Float64.(Float32.(A)\Float32.(b))
# end

# ns = 2:10:200

# times_f32 = Float64[]

# times_f64 = Float64[]
# for n in ns
#     @info n
#     A = rand(Float64, n, n)
#     b = rand(Float64, n)

#     t1 = @belapsed $A\$b
#     t2 = @belapsed offloadf32_solve($A, $b)
#     push!(times_f64, t1)
#     push!(times_f32, t2)
# end

using Plots, JLD2

data = load("times_linsolve.jld2")

times_f32 = data["times_f32"]
times_f64 = data["times_f64"]

plot(ns, times_f64, label = "Linear Solve: Float64", linewidth = 2, marker = :circle)
plot!(ns, times_f32, label = "Linear Solve: Offload Float32", linewidth = 2,
      marker = :circle)

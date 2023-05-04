using SafeTestsets, Test

# @time @testset "Regression" begin include("ode_tests.jl") end
@time @testset "Benchmarking" begin include("benchmark.jl") end

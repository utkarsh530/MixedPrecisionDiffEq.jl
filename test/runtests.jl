using SafeTestsets, Test

@time @testset "Regression" begin include("ode_tests.jl") end
@time @testset "ODE Tests" begin include("ode_tests.jl") end

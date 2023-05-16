[![CI](https://github.com/utkarsh530/MixedPrecisionDiffEq.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/utkarsh530/MixedPrecisionDiffEq.jl/actions/workflows/CI.yml)

# MixedPrecisionDiffEq.jl
Mixed Precision ODE solvers in Julia compatible with SciML ecosystem support!

# Setup

For GPU usage, ensure that you have a NVIDIA GPU working. While in the directory, run:

```
$ julia --project=. --threads=auto
julia> using Pkg
julia> Pkg.instantiate()
julia> Pkg.precompile()
```

# Usage

## Mixed precision methods in linear systems

```julia

using LinearSolve, MixedPrecisionDiffEq

prob = LinearProblem(rand(100,100), rand(100))

sol = solve(prob, MixedPrecisionLinsolve())
```

The default matrix factorization choice is `RFLUFactorization`, which is a recursive LU factorization.

However, you can simply use any factorization used by `LinearSolve.jl` by simply changing the argument as:


```julia
sol = solve(prob, MixedPrecisionLinsolve(FastLUFactorization()))
```

## Using mixed precision methods in ODEs


```julia

using OrdinaryDiffEq, MixedPrecisionDiffEq

function f!(du, u, p, t)
  du[1] = -p[1]*u[1] + p[2]*u[2]*u[3]
  du[2] = p[1]*u[1] - p[2]*u[2]*u[3] - p[3]*u[2]*u[2]
  du[3] = p[3]*u[2]*u[2]
end


u0 = [1.0,0.0,0.0]
tspan = (0.0, 1e5)
p = [0.04, 1e4, 3e7]

prob = ODEProblem(f!,u0, tspan, p)

sol = solve(prob, TRBDF2(;linsolve = MixedPrecisionLinsolve()))
```

## GPU acceleration of linear solve with mixed precision

The use GPU acceleration, run:

```julia

sol = solve(prob, TRBDF2(;linsolve = MixedPrecisionCudaOffloadFactorization()))
```

The GPU method only supports LU factorization as of now.

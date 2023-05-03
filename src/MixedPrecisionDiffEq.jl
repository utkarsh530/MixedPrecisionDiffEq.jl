module MixedPrecisionDiffEq

using OrdinaryDiffEq, DiffEqBase, MacroTools, SciMLBase
using LinearAlgebra, LinearSolve, Setfield

include("nlsolve.jl")
include("linsolve.jl")
export MixedPrecisionNLSolverAlgorithm, MixedPrecisionLinsolve
end # module MixedPrecisionDiffEq

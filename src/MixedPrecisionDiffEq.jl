module MixedPrecisionDiffEq

using OrdinaryDiffEq, DiffEqBase, MacroTools

include("nlsolve.jl")
export MixedPrecisionNLSolverAlgorithm
end # module MixedPrecisionDiffEq

module MixedPrecisionDiffEq

using SciMLBase, LinearAlgebra, LinearSolve

include("linsolve.jl")
include("factorization.jl")
export MixedPrecisionLinsolve, MixedRFLUFactorization
end # module MixedPrecisionDiffEq

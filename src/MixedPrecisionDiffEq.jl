module MixedPrecisionDiffEq

using SciMLBase, LinearAlgebra, LinearSolve
import CUDA

include("linsolve.jl")
include("factorization.jl")
include("gpu_linsolve.jl")
export MixedPrecisionLinsolve, MixedRFLUFactorization,
       MixedPrecisionCudaOffloadFactorization
end # module MixedPrecisionDiffEq

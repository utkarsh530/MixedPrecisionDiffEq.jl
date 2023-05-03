# Utility to get the number of non linear solve failures
struct NLSolverStats{active}
    nlsolvefails::Vector{Vector{Int64}}
end

NLSolverStats(active::Bool = false) = NLSolverStats{active}([Int64[]])

@inline OrdinaryDiffEq.initialize!(::NLSolverStats{false}) = nothing

@inline function OrdinaryDiffEq.initialize!(stats::NLSolverStats{true})
    length(stats.nlsolvefails[end]) == 0 && return nothing
    return push!(stats.nlsolvefails, zeros(Int64, length(stats.nlsolvefails[end])))
end

@inline nlsolvefail!(::NLSolverStats{false}, args...) = nothing

@inline @inbounds function nlsolvefail!(stats::NLSolverStats{true}, didfail::Bool,
                                        stage::Int)
    nf = stats.nlsolvefails[end]
    lnf = length(nf)
    stage > lnf && append!(nf, zeros(Int64, stage - lnf))
    return nf[stage] += didfail
end

# MixedPrecisionNLSolverAlgorithm: Is used to construct the MixedPrecisionNLSolver
struct MixedPrecisionNLSolverAlgorithm{A <: OrdinaryDiffEq.AbstractNLSolverAlgorithm,
                                       N <: NLSolverStats} <:
       OrdinaryDiffEq.AbstractNLSolverAlgorithm
    alg::A
    nlsolverstats::N
end

function MixedPrecisionNLSolverAlgorithm(; alg = NLNewton(),
                                         collect_statistics::Bool = false)
    return MixedPrecisionNLSolverAlgorithm(alg, NLSolverStats(collect_statistics))
end

# Actual MixedPrecisionNLSolver
mutable struct MixedPrecisionNLSolver{A, iip, NL <: OrdinaryDiffEq.AbstractNLSolver{A, iip},
                                      N <: NLSolverStats} <:
               OrdinaryDiffEq.AbstractNLSolver{A, iip}
    current_stage::Int
    nlsolver::NL
    nlsolverstats::N
end

function OrdinaryDiffEq.build_nlsolver(alg, nlalg::MixedPrecisionNLSolverAlgorithm, u,
                                       uprev, p,
                                       t, dt,
                                       f, rate_prototype, ::Type{uEltypeNoUnits},
                                       ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                       γ, c, α,
                                       iip) where {uEltypeNoUnits, uBottomEltypeNoUnits,
                                                   tTypeNoUnits}
    uBottomEltypeNoUnits_c = Float32
    uEltypeNoUnits_c = Float32
    tTypeNoUnits_c = Float32
    # @show Float32.(f)

    # @show uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, typeof(u), typeof(p), typeof(t), typeof(dt), typeof(c), typeof(α)
    nlsolver = OrdinaryDiffEq.build_nlsolver(alg, nlalg.alg, Float32.(u), Float32.(uprev),
                                             Float32.(p), Float32.(t), Float32.(dt), f,
                                             Float32.(rate_prototype),
                                             uBottomEltypeNoUnits_c,
                                             uEltypeNoUnits_c, tTypeNoUnits_c, Float32.(γ),
                                             Float32.(c), Int32.(α),
                                             iip)

    # nlsolver = OrdinaryDiffEq.build_nlsolver(alg, nlalg.alg, u, uprev,p, t, dt, f,
    # rate_prototype, uBottomEltypeNoUnits,
    # uEltypeNoUnits, tTypeNoUnits, γ, c, α,
    # iip)
    OrdinaryDiffEq.initialize!(nlalg.nlsolverstats)
    return MixedPrecisionNLSolver(1, nlsolver, nlalg.nlsolverstats)
end

function Base.getproperty(nlsolver::MixedPrecisionNLSolver, name::Symbol)
    if name ∈
       (:current_stage, :nlsolver, :nlsolverstats, :to)
        return getfield(nlsolver, name)
    else
        return getproperty(nlsolver.nlsolver, name)
    end
end

function Base.setproperty!(nlsolver::MixedPrecisionNLSolver, name::Symbol, val)
    if name ∈
       (:current_stage, :nlsolver, :nlsolverstats)
        return setfield!(nlsolver, name, val)
    else
        return setproperty!(nlsolver.nlsolver, name, val)
    end
end

MacroTools.@forward MixedPrecisionNLSolver.nlsolver OrdinaryDiffEq.get_status,
                                                    OrdinaryDiffEq.get_new_W_γdt_cutoff,
                                                    OrdinaryDiffEq.initial_η,
                                                    OrdinaryDiffEq.isnewton,
                                                    OrdinaryDiffEq.isJcurrent,
                                                    OrdinaryDiffEq.isfirstcall,
                                                    OrdinaryDiffEq.isfirststage,
                                                    OrdinaryDiffEq.getnfails,
                                                    OrdinaryDiffEq.get_new_W!,
                                                    OrdinaryDiffEq.get_W,
                                                    OrdinaryDiffEq.set_W_γdt!,
                                                    OrdinaryDiffEq.du_cache,
                                                    OrdinaryDiffEq.du_alias_or_new,
                                                    OrdinaryDiffEq.compute_step!,
                                                    OrdinaryDiffEq.apply_step!,
                                                    OrdinaryDiffEq.postamble!

# Leads to method ambiguity without this
function OrdinaryDiffEq.set_new_W!(nlsolver::MixedPrecisionNLSolver, b::Bool)
    return OrdinaryDiffEq.set_new_W!(nlsolver.nlsolver, b)
end

function OrdinaryDiffEq.nlsolvefail(nlsolver::MixedPrecisionNLSolver)
    didfail = OrdinaryDiffEq.nlsolvefail(nlsolver.nlsolver)
    nlsolvefail!(nlsolver.nlsolverstats, didfail, nlsolver.current_stage - 1)
    return didfail
end

function OrdinaryDiffEq.setfirststage!(nlsolver::MixedPrecisionNLSolver, val::Bool)
    OrdinaryDiffEq.setfirststage!(nlsolver.nlsolver, val)
    return val && (nlsolver.current_stage = 1)
end

function OrdinaryDiffEq.markfirststage!(nlsolver::MixedPrecisionNLSolver)
    return OrdinaryDiffEq.setfirststage!(nlsolver, true)
end

function Base.resize!(nlsolver::MixedPrecisionNLSolver, args...; kwargs...)
    return resize!(nlsolver.nlsolver, args...; kwargs...)
end

function OrdinaryDiffEq.nlsolve!(nlsolver::MixedPrecisionNLSolver,
                                 integrator::SciMLBase.DEIntegrator, args...; kwargs...)
    z = OrdinaryDiffEq.nlsolve!(nlsolver.nlsolver, integrator, args...; kwargs...)
    nlsolver.current_stage += 1
    return z
end

module ACP

include("ACP_MC.jl")
include("BackwardKernel.jl")


using .ACP_MC, .BackwardKernel

#export generate, log_marginal, alloc_dist
export Particle, smc_weight
export EventQueue, sum, length, eltype, iterate

end

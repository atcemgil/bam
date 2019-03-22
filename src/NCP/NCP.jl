module NCP

include("NCP_Exact.jl")
include("NCP_EM.jl")
include("NCP_VB.jl")
include("NCP_MC.jl")
include("BackwardKernel.jl")


using .NCP_Exact, .NCP_EM, .NCP_VB, .NCP_MC, .BackwardKernel

export allocations, full, generate
export log_marginal, alloc_dist, EP_dist
export log_posterior_T, log_PT
export standard_EM
export standard_VB
export Particle, smc_weight, filtering_proposal, resample
export EventQueue, sum, length, eltype, iterate

end
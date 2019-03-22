module sNMF

include("sNMF_Exact.jl")
include("sNMF_VB.jl")
include("sNMF_MC.jl")
include("BackwardKernel.jl")


using .sNMF_Exact, .sNMF_VB, .sNMF_MC, .BackwardKernel

export generate, log_marginal
export standard_VB, online_VB
export sParticle, smc_weight, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end
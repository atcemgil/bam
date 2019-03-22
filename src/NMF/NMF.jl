module NMF

include("NMF_Exact.jl")
include("NMF_EM.jl")
include("NMF_VB.jl")
include("NMF_MC.jl")
include("BackwardKernel.jl")


using .NMF_Exact, .NMF_EM, .NMF_VB, .NMF_MC, .BackwardKernel

export generate, log_marginal
export standard_EM, dual_EM 
export standard_VB, online_VB
export Particle, smc_weight, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end
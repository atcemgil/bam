module SCP

include("SCP_Exact.jl")
include("SCP_EM.jl")
include("SCP_VB.jl")
include("SCP_MC.jl")
include("BackwardKernel.jl")


using .SCP_Exact, .SCP_EM, .SCP_VB, .SCP_MC, .BackwardKernel

export generate, log_marginal
export standard_EM, dual_EM, standard_VB
export Particle, smc_weight, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end
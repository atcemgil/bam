module CP

include("CP_Exact.jl")
include("CP_EM.jl")
include("CP_VB.jl")
include("CP_MC.jl")
include("BackwardKernel.jl")


using .CP_Exact, .CP_EM, .CP_VB, .CP_MC, .BackwardKernel

export generate, log_marginal
export standard_EM, dual_EM 
export standard_VB, online_VB
export Particle, smc_weight, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end
module sNMF_VB

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions

export standard_VB, online_VB

function standard_VB(X::Array{ℜ,2}, R::Ƶ ; μ::ℜ=1.0, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ = size(X,1)

    a::ℜ, b::ℜ = I*I*γ, γ/μ
    α_R, α_IR = fill(a/R,R), fill(2.0*a/(I*R),I,R)
    S₊ = sum(X)
    
    log_λ = digamma(S₊ + a) - log(b + 1)
    log_θ_R = log.(rand(Dirichlet(α_R.+0.1)))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r].+0.1))],I,R)
    
    S_R, S_IR, S⁺::ℜ = zeros(ℜ,R), zeros(ℜ,I,R), 0.0  
    s, p, log_ρ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), Array{ℜ}(undef,R)
    
    ELBO::Array{ℜ} = zeros(ℜ,EPOCHS)
                         
    X_full = similar(X)
    for eph=1:EPOCHS
        S_R .= 0.0
        S_IR .= 0.0
        S⁺ = 0.0
                                        
        for i₂=1:I, i₁=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i₁,:] .+ log_θ_IR[i₂,:]
            log_ρ₊ = logsumexp(log_ρ)
            p .= exp.(log_ρ .- log_ρ₊)
            
            X_full[i₁,i₂] = isnan(X[i₁,i₂]) ? exp(log_ρ₊) : X[i₁,i₂]
            s .= X_full[i₁,i₂] .* p
            
            S_IR[i₁,:] .+= s
            S_IR[i₂,:] .+= s
            S_R .+= s
            S⁺ += X_full[i₁,i₂]

            ELBO[eph] += X[i₁,i₂]*log_ρ₊ - lgamma(X[i₁,i₂] + 1.0)
        end

        ELBO[eph] += a*log(b) - (a + S⁺)*log(b+1.0) - sum(lgamma.(α_R)) + sum(lgamma.(α_R .+ S_R))
        ELBO[eph] += sum(lgamma.(2.0 .* α_R)) - sum(lgamma.(α_IR)) 
        ELBO[eph] += sum(lgamma.(α_IR .+ S_IR)) - sum(lgamma.(2.0 .* (α_R .+ S_R)))
        ELBO[eph] -= S⁺ * log_λ + sum(S_R .* log_θ_R) + sum(S_IR .* log_θ_IR)
                                        
        log_λ = digamma(S⁺+a) - log(b + 1.0)
        log_θ_R .= digamma.(S_R .+ α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR .+ α_IR) .- digamma.(S_R.+α_R)'
        
    end
    return ELBO, X_full, log_λ, (log_θ_R, log_θ_IR)
end

end
module SCP_VB

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions

export standard_VB

function standard_VB(X::Array{ℜ,3}, X⁺::Array{ℜ,2}, R::Ƶ ; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_KR, α_IKR, α_JKR = fill(a/R,R), fill(a/(K*R),K,R), fill(a/(I*K*R),I,K,R), fill(a/(J*K*R),J,K,R)

    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = log.(rand(Dirichlet(α_R)))
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K ))],K,R)
    log_θ_IKR = reshape([log(θ_ikr) for k=1:K, r=1:R for θ_ikr=rand(Dirichlet(α_IKR[:,k,r] .+ 1.0/I))],I,K,R)
    log_θ_JKR = reshape([log(θ_jkr) for k=1:K, r=1:R for θ_jkr=rand(Dirichlet(α_JKR[:,k,r] .+ 1.0/J))],J,K,R)
    
    s, S_R, S_KR, S_IKR, S_JKR, S⁺ = zeros(ℜ,K,R), zeros(ℜ,R), zeros(ℜ,K,R), zeros(ℜ,I,K,R), zeros(ℜ,J,K,R), sum(X⁺)
    π, log_π, log_ρ = Array{ℜ}(undef,K,R), Array{ℜ}(undef,K,R), Array{ℜ}(undef,K,R)
    log_ρ_ij::ℜ, log_ρ_ijk::Array{ℜ} = 0.0, Array{ℜ}(undef,K)
    
    ELBO::Array{ℜ} = zeros(ℜ,EPOCHS)                     
    X_full = copy(X)

    for eph=1:EPOCHS
        S_R .= 0.0
        S_KR .= 0.0
        S_IKR .= 0.0
        S_JKR .= 0.0
                                        
        for j=1:J, i=1:I #order of traversal is important
            
            log_ρ .= log_λ .+ log_θ_R' .+ log_θ_KR .+ log_θ_IKR[i,:,:] .+ log_θ_JKR[j,:,:]
    
            if isnan(X[i,j,1])
                log_ρ_ij = logsumexp(log_ρ)
                log_π .= log_ρ .- log_ρ_ij
                π .= exp.(log_π)
                s = X⁺[i,j] .* π
                X_full[i,j,:] .= sum(s,dims=2)[:,1]
                ELBO[eph] += X⁺[i,j] * log_ρ_ij - lgamma(X⁺[i,j] + 1.0) 
            else
                log_ρ_ijk = logsumexp(log_ρ;dims=2)                                    
                log_π .= log_ρ .- log_ρ_ijk 
                π .= exp.(log_π)
                s = X[i,j,:] .* π
                ELBO[eph] += sum(X[i,j,:] .* log_ρ_ijk .- lgamma.(X[i,j,:] .+ 1.0)) 
            end

            S_R .+= sum(s,dims=1)[1,:]
            S_KR .+= s
            S_JKR[j,:,:] .+= s
            S_IKR[i,:,:] .+= s
                                                            
        end

        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) + sum(lgamma.(α_KR)) - sum(lgamma.(α_KR .+ S_KR))
        ELBO[eph] += sum(lgamma.(α_IKR .+ S_IKR)) - sum(lgamma.(α_IKR))
        ELBO[eph] += sum(lgamma.(α_JKR .+ S_JKR)) - sum(lgamma.(α_JKR))
        ELBO[eph] -= S⁺ * log_λ + sum(S_R .* log_θ_R) + sum(S_KR .* log_θ_KR) 
        ELBO[eph] -= sum(S_JKR .* log_θ_JKR) + sum(S_IKR .* log_θ_IKR)
                                        
        log_λ = digamma(S⁺+a) - log(b+1.0)
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R.+α_R)'
        log_θ_IKR .= digamma.(S_IKR.+α_IKR) .- reshape(digamma.(S_KR.+α_KR),1,K,R)
        log_θ_JKR .= digamma.(S_JKR.+α_JKR) .- reshape(digamma.(S_KR.+α_KR),1,K,R)

    end
    return ELBO, X_full, log_λ, (log_θ_R, log_θ_KR, log_θ_IKR, log_θ_JKR)
end

end
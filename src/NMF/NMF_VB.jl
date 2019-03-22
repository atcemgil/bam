module NMF_VB

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions

export standard_VB, online_VB

function standard_VB(X::Array{ℜ,2}, K::Ƶ ; μ::ℜ=nanmean(X), γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    S₊ = sum(X)
    
    #log_λ = log(rand(Gamma(a,1.0/b)))
    log_λ = digamma(S₊ + a) - log(b + 1)
    log_θ_K = log.(rand(Dirichlet(α_K.+0.1)))
    log_θ_JK = reshape([log(θ_jk) for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k].+0.1))],J,K)
    log_θ_IK = reshape([log(θ_ik) for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k].+0.1))],I,K)
    
    S_K, S_JK, S_IK, S⁺::ℜ = zeros(ℜ,K), zeros(ℜ,J,K), zeros(ℜ,I,K), 0.0  
    s, p, log_ρ, log_ρ₊ = Array{ℜ}(undef,K), Array{ℜ}(undef,K), Array{ℜ}(undef,K), 0.0
    
    ELBO::Array{ℜ} = zeros(ℜ,EPOCHS)          
    X_full = similar(X)

    for eph=1:EPOCHS
        S_K .= 0.0
        S_JK .= 0.0
        S_IK .= 0.0
        S⁺ = 0.0
                                        
        for j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_K .+ log_θ_JK[j,:] .+ log_θ_IK[i,:]
            log_ρ₊ = logsumexp(log_ρ)
            p .= exp.(log_ρ .- log_ρ₊)

            X_full[i,j] = isnan(X[i,j]) ? exp(log_ρ₊) : X[i,j]
            s .= X_full[i,j] .* p
  
            S_JK[j,:] .+= s
            S_IK[i,:] .+= s
            S_K[:] .+= s
            S⁺ += X_full[i,j]

            ELBO[eph] += isnan(X[i,j]) ? X_full[i,j] : X[i,j]*log_ρ₊ - lgamma(X[i,j] + 1.0)
        end
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) - sum(lgamma.(α_JK)) + sum(lgamma.(α_JK .+ S_JK))
        ELBO[eph] += sum(lgamma.(α_K)) - sum(lgamma.(α_IK)) - sum(lgamma.(α_K .+ S_K)) + sum(lgamma.(α_IK .+ S_IK))
        ELBO[eph] -= S⁺ * log_λ + sum(S_K .* log_θ_K) + sum(S_JK .* log_θ_JK) + sum(S_IK .* log_θ_IK)
                                        
        log_λ = digamma(S⁺+a) - log(b+1.0)
        log_θ_K .= digamma.(S_K.+α_K) .- digamma(S⁺+a)
        log_θ_JK .= digamma.(S_JK.+α_JK) .- digamma.(S_K.+α_K)'
        log_θ_IK .= digamma.(S_IK.+α_IK) .- digamma.(S_K.+α_K)'

    end

    return ELBO, X_full, log_λ, (log_θ_K, log_θ_IK, log_θ_JK)
end

function online_VB(X::Array{ℜ,2}, K::Ƶ, Ω::AbstractArray{ℜ,1}; μ::ℜ=nanmean(X), γ::ℜ=0.1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_K = log.(rand(Dirichlet(α_K)))
    log_θ_JK = reshape([log(θ_jk) for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k]))],J,K)
    log_θ_IK = reshape([log(θ_ik) for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k]))],I,K)
    
    S_K, S_JK, S_IK, S⁺::ℜ = zeros(ℜ,K), zeros(ℜ,J,K), zeros(ℜ,I,K), 0.0  
    s, p, log_ρ, log_ρ₊ = Array{ℜ}(undef,K), Array{ℜ}(undef,K), Array{ℜ}(undef,K), 0.0
    
    ELBO::Array{ℜ} = zeros(ℜ,length(Ω))          
    X_full = similar(X)

    for (eph,ω) in enumerate(Ω)
        S_K .= 0.0
        S_JK .= 0.0
        S_IK .= 0.0
        S⁺ = 0.0
                                        
        for j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_K .+ log_θ_JK[j,:] .+ log_θ_IK[i,:]
            log_ρ₊ = logsumexp(log_ρ)
            p .= exp.(log_ρ .- log_ρ₊)
                                            
            X_full[i,j] = isnan(X[i,j]) ? ω*exp(log_ρ₊) : rand(Binomial(Ƶ(X[i,j]),ω)) 
            s .= X_full[i,j] .* p

            S_JK[j,:] .+= s
            S_IK[i,:] .+= s
            S_K[:] .+= s
            S⁺ += X_full[i,j]

            ELBO[eph] += isnan(X[i,j]) ? X_full[i,j] : X[i,j]*log_ρ₊ - lgamma(X[i,j] + 1.0)
        end

        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) - sum(lgamma.(α_JK)) + sum(lgamma.(α_JK .+ S_JK))
        ELBO[eph] += sum(lgamma.(α_K)) - sum(lgamma.(α_IK)) - sum(lgamma.(α_K .+ S_K)) + sum(lgamma.(α_IK .+ S_IK))
        ELBO[eph] -= S⁺ * log_λ + sum(S_K .* log_θ_K) + sum(S_JK .* log_θ_JK) + sum(S_IK .* log_θ_IK)
                                        
        log_λ = digamma(S⁺ + a) - log(b + ω) ## should be checked
        log_θ_K .= digamma.(S_K.+α_K) .- digamma(S⁺+a)
        log_θ_JK .= digamma.(S_JK.+α_JK) .- digamma.(S_K.+α_K)'
        log_θ_IK .= digamma.(S_IK.+α_IK) .- digamma.(S_K.+α_K)'

    end
    return ELBO, X_full, log_λ, (log_θ_K, log_θ_IK, log_θ_JK)
end

end
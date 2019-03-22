module NMF_EM

include("../Misc.jl")

using .Misc
using Distributions, SpecialFunctions

export standard_EM, dual_EM

function standard_EM(X::Array{ℜ,2}, K::Ƶ; μ::ℜ=nanmean(X), γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_K = log.(rand(Dirichlet(α_K)))
    log_θ_JK = reshape([log(θ_jk) for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k]))],J,K)
    log_θ_IK = reshape([log(θ_ik) for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k]))],I,K)
                            
    S_K, S_JK, S_IK, S⁺::ℜ = zeros(ℜ,K), zeros(ℜ,J,K), zeros(ℜ,I,K), 0.0  
    s, p, log_ρ, log_ρ₊ = Array{ℜ}(undef,K), Array{ℜ}(undef,K), Array{ℜ}(undef,K), 0.0
                         
    X_full = similar(X)
    for eph=1:EPOCHS
        S_K .= α_K
        S_JK .= α_JK
        S_IK .= α_IK
        S⁺ = a
                                        
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
        end
                                        
        log_λ = log(max(S⁺ - 1.0, ϵ)) - log(b+1.0)
        log_θ_K .= log.(max.(S_K .- 1.0, ϵ))
        log_θ_JK .= log.(max.(S_JK .- 1.0, ϵ))
        log_θ_IK .= log.(max.(S_IK .- 1.0, ϵ))

        log_θ_K .-= logsumexp(log_θ_K;dims=1)
        log_θ_JK .-= logsumexp(log_θ_JK;dims=1)
        log_θ_IK .-= logsumexp(log_θ_IK;dims=1)
    end
    return X_full, log_λ, (log_θ_K, log_θ_IK, log_θ_JK)
end

function dual_EM(X::Array{ℜ,2}, K::Ƶ; μ::ℜ=nanmean(X), γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_K = log.(rand(Dirichlet(α_K)))
    log_θ_JK = reshape([log(θ_jk) for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k]))],J,K)
    log_θ_IK = reshape([log(θ_ik) for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k]))],I,K)
                            
    S_K, S_JK, S_IK, S⁺::ℜ = zeros(ℜ,K), zeros(ℜ,J,K), zeros(ℜ,I,K), 0.0  
    s, p, log_ρ, log_ρ₊ = Array{ℜ}(undef,K), Array{ℜ}(undef,K), Array{ℜ}(undef,K), 0.0
                         
    X_full = similar(X)
    for eph=1:EPOCHS
        S_K .= α_K
        S_JK .= α_JK
        S_IK .= α_IK
        S⁺ = a
                                        
        for j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_K .+ log_θ_JK[j,:] .+ log_θ_IK[i,:]
            log_ρ₊ = logsumexp(log_ρ)
            p .= exp.(log_ρ .- log_ρ₊)
                                            
            X_full[i,j] = isnan(X[i,j]) ? floor(exp(log_ρ₊)) : X[i,j] 
            s .= finucan(X_full[i,j],p)

            S_JK[j,:] .+= s
            S_IK[i,:] .+= s
            S_K[:] .+= s
            S⁺ += X_full[i,j]
        end
                                        
        log_λ = digamma(S⁺+a) - log(b+1.0)
        log_θ_K .= digamma.(S_K.+α_K) .- digamma(S⁺+a)
        log_θ_JK .= digamma.(S_JK.+α_JK) .- digamma.(S_K.+α_K)'
        log_θ_IK .= digamma.(S_IK.+α_IK) .- digamma.(S_K.+α_K)'
    end
    return X_full, S⁺, (S_K, S_IK, S_JK)
end

end
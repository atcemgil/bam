module NMF_Exact

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export generate, log_marginal

function generate(I::Ƶ, J::Ƶ, K::Ƶ; μ::ℜ=1.0, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    λ = rand(Gamma(a,1.0/b))
    θ_K = rand(Dirichlet(α_K))
    θ_JK = reshape([θ_jk for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k]))],J,K)
    θ_IK = reshape([θ_ik for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k]))],I,K)
                            
    θ_IJ = Array{ℜ}(undef,I,J)
    @einsum θ_IJ[i,j] = θ_K[k]*θ_JK[j,k]*θ_IK[i,k]
    Λ_IJ = λ.*θ_IJ
    
    X::Array{ℜ} = map.(λ_ij -> rand(Poisson(λ_ij)), Λ_IJ)
    return X, λ, (θ_K, θ_IK, θ_JK)
end

function generate(I::Ƶ, J::Ƶ, K::Ƶ, T::Ƶ; γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ = I*J*γ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    θ_K = rand(Dirichlet(α_K))
    θ_JK = reshape([θ_jk for k=1:K for θ_jk=rand(Dirichlet(α_JK[:,k]))],J,K)
    θ_IK = reshape([θ_ik for k=1:K for θ_ik=rand(Dirichlet(α_IK[:,k]))],I,K)
                            
    θ_IJ = Array{ℜ}(undef,I,J)
    @einsum θ_IJ[i,j] = θ_K[k]*θ_JK[j,k]*θ_IK[i,k]
    θ_flat = reshape(θ_IJ, I*J)

    X::Array{ℜ} = reshape(rand(Multinomial(T,θ_flat)),I,J)
    return X, (θ_K, θ_IK, θ_JK)
end

function log_marginal(X::Array{ℜ,2}, K::Ƶ; μ::ℜ=nanmean(X), γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)
    X_flat = reshape(Int.(X),I*J)
    
    a::ℜ, b::ℜ = I*J*γ, γ/μ
    α_K, α_JK, α_IK = fill(a/K,K), fill(a/(K*J),J,K), fill(a/(K*I),I,K)
    
    s, S_K, S_JK, S_IK, S⁺ = zeros(ℜ,K), zeros(ℜ,K), zeros(ℜ,J,K), zeros(ℜ,I,K), sum(X)
    log_PX, log_PS = -Inf, -Inf
    i::Ƶ, j::Ƶ = 0, 0
    
    log_C = a*log(b) - (a + S⁺)*log(b + 1.0) #+ lgamma(a + S⁺) - lgamma(a)  
    log_C += sum(lgamma.(α_K)) - sum(lgamma.(α_IK)) - sum(lgamma.(α_JK))
    
    if K == 1 
        S_K .= S⁺
        S_JK[:,1] = sum(X,dims=1)
        S_IK[:,1] = sum(X,dims=2)
        log_PS = log_C
        log_PS += sum(lgamma.(α_IK .+ S_IK)) + sum(lgamma.(α_JK .+ S_JK)) - sum(lgamma.(α_K .+ S_K))
        log_PS -= sum(lgamma.(X .+ 1.0))
        return log_PS
    end
    
    for S_div = product(map(X_ij -> combinations(1:(X_ij+K-1),K-1), X_flat)...)
        S_K .= α_K
        S_JK .= α_JK 
        S_IK .= α_IK
        log_PS = 0.0
        for (ij, part) = enumerate(S_div)
            i, j = mod(ij-1,I)+1, div(ij-1,I)+1
            s[1] = part[1]-1
            for k=2:K-1
                s[k] = part[k] - part[k-1] - 1
            end
            s[K] = X[i,j] + K - 1 - part[K-1]
            
            S_K .+= s
            S_JK[j,:] .+= s
            S_IK[i,:] .+= s
            log_PS -= sum(lgamma.(s .+ 1.0))
        end
        log_PS += sum(lgamma.(S_IK)) + sum(lgamma.(S_JK)) - sum(lgamma.(S_K)) + log_C
        log_PX = logsumexp([log_PX,log_PS])
    end
    
    return log_PX
end

end

module SCP_Exact

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export generate, log_marginal

function generate(I::Ƶ, J::Ƶ, K::Ƶ, R::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_KR, α_IKR, α_JKR = fill(a/R,R), fill(a/(K*R),K,R), fill(a/(I*K*R),I,K,R), fill(a/(J*K*R),J,K,R)
    
    λ = rand(Gamma(a,1.0/b))
    θ_R = rand(Dirichlet(α_R))
    θ_KR = reshape([θ_kr for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r]))],K,R)
    θ_IKR = reshape([θ_ikr for k=1:K, r=1:R for θ_ikr=rand(Dirichlet(α_IKR[:,k,r]))],I,K,R)
    θ_JKR = reshape([θ_jkr for k=1:K, r=1:R for θ_jkr=rand(Dirichlet(α_JKR[:,k,r]))],J,K,R)
                            
    Λ_IJK::Array{ℜ} = [λ * sum(θ_R .* θ_KR[k,:] .* θ_IKR[i,k,:] .* θ_JKR[j,k,:]) for i=1:I, j=1:J, k=1:K]

    X::Array{ℜ} = map.(λ_ijk -> rand(Poisson(λ_ijk)), Λ_IJK)
    X⁺::Array{ℜ} = sum(X,dims=3)[:,:]
    return X, X⁺, λ, (θ_R, θ_KR, θ_IKR, θ_JKR)
end

function generate(I::Ƶ, J::Ƶ, K::Ƶ, R::Ƶ, T::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_KR, α_IKR, α_JKR = fill(a/R,R), fill(a/(K*R),K,R), fill(a/(I*K*R),I,K,R), fill(a/(J*K*R),J,K,R)
    
    θ_R = rand(Dirichlet(α_R))
    θ_KR = reshape([θ_kr for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r]))],K,R)
    θ_IKR = reshape([θ_ikr for k=1:K, r=1:R for θ_ikr=rand(Dirichlet(α_IKR[:,k,r]))],I,K,R)
    θ_JKR = reshape([θ_jkr for k=1:K, r=1:R for θ_jkr=rand(Dirichlet(α_JKR[:,k,r]))],J,K,R)
                            
    θ_IJK::Array{ℜ} = [sum(θ_R .* θ_KR[k,:] .* θ_IKR[i,k,:] .* θ_JKR[j,k,:]) for i=1:I, j=1:J, k=1:K]
    θ_flat = reshape(θ_IJK, I*J*K)

    X::Array{ℜ} = reshape(rand(Multinomial(T,θ_flat)),I,J,K)
    X⁺::Array{ℜ} = sum(X,dims=3)[:,:,1]
    return X, X⁺, (θ_R, θ_KR, θ_IKR, θ_JKR)
end

function comb(X_ij::Array{ℜ,1}, X⁺_ij::Ƶ) where {ℜ <: Real, Ƶ <: Int}
    K::Ƶ = length(X_ij)
    if isnan(X_ij[1])
        return combinations(1:(X⁺_ij+K-1), K-1)
    else
        return combinations(cumsum(Int.(X_ij) .+ 1)[1:K-1], K-1)
    end
end

function log_marginal(X::Array{ℜ,3}, X⁺::Array{ℜ,2}, R::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_KR, α_IKR, α_JKR = fill(a/(K*R),K,R), fill(a/(I*K*R),I,K,R), fill(a/(J*K*R),J,K,R)
    
    s, S_KR, S_IKR, S_JKR, S⁺ = zeros(ℜ,R), zeros(ℜ,K,R), zeros(ℜ,I,K,R), zeros(ℜ,J,K,R), sum(X⁺)
    log_PX, log_PS = -Inf, -Inf
    i::Ƶ, j::Ƶ , k::Ƶ = 0, 0, 0
    
    log_C::ℜ = a*log(b) - (a + S⁺)*log(b + 1.0)
    log_C += sum(lgamma.(α_KR)) - sum(lgamma.(α_IKR)) - sum(lgamma.(α_JKR))

    X_full = zeros(ℜ,I,J,K)

    for X_div = product([comb(X[i,j,:], Int(X⁺[i,j])) for i=1:I,j=1:J]...)
        if K==1
            X_full[:,:,1] .= X⁺
        else
            X_full .= 0.0
            for (ij, part) = enumerate(X_div) ## fix X
                i, j = mod(ij-1,I)+1, div(ij-1,I)+1
                X_full[i,j,1] = part[1]-1
                for k=2:K-1
                    X_full[i,j,k] = part[k] - part[k-1] - 1
                end
                X_full[i,j,K] = X⁺[i,j] + K-1 - part[K-1]
            end
        end
        
        if R==1
            S_KR[:,1] .= sum(X_full,dims=(1,2))[1,1,:]
            S_IKR[:,:,1] .= sum(X_full,dims=2)[:,1,:]
            S_JKR[:,:,1] .= sum(X_full,dims=1)[1,:,:]

            log_PS = log_C
            log_PS += sum(lgamma.(S_IKR .+ α_IKR)) + sum(lgamma.(S_JKR .+ α_JKR)) - sum(lgamma.(S_KR .+ α_KR))
            log_PS -= sum(lgamma.(X_full .+ 1.0))
            log_PX = logsumexp([log_PX,log_PS])
        else
            for S_div = product(map(X_ijk -> combinations(1:(X_ijk+R-1),R-1), X_full)...)
                S_KR .= α_KR
                S_IKR .= α_IKR
                S_JKR .= α_JKR

                log_PS = 0.0
                for (ijk, part) = enumerate(S_div)
                    i, j, k = mod(ijk-1,I)+1, mod(div(ijk-1,I),J)+1, div(ijk-1,I*J)+1
                    s[1] = part[1]-1
                    for r=2:R-1
                        s[r] = part[r] - part[r-1] - 1
                    end
                    s[R] = X_full[i,j,k] + R-1 - part[R-1]
            
                    S_KR[k,:] .+= s
                    S_IKR[i,k,:] .+= s
                    S_JKR[j,k,:] .+= s
                    log_PS -= sum(lgamma.(s .+ 1.0))
                end
                log_PS += sum(lgamma.(S_IKR)) + sum(lgamma.(S_JKR)) - sum(lgamma.(S_KR)) + log_C
                log_PX = logsumexp([log_PX,log_PS])
            end
        end
    end
    return log_PX
end

end
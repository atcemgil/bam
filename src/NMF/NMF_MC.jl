module NMF_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions

export Particle, smc_weight, particle_mcmc

mutable struct Particle{ℜ <: Real}
    S_K::Array{ℜ,1}
    S_IK::Array{ℜ,2}
    S_JK::Array{ℜ,2}
    function Particle(I::Int,J::Int,K::Int; γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*J*γ
        return new{ℜ}(fill(a/K,K), fill(a/(K*I),I,K), fill(a/(K*J),J,K))
    end
    function Particle(S_K::Array{ℜ,1}, S_IK::Array{ℜ,2}, S_JK::Array{ℜ,2}) where {ℜ<:Real}
        return new{ℜ}(S_K, S_IK, S_JK)
    end
end

function smc_weight(X::Array{ℜ,2}, K::Ƶ, N::Ƶ=1; μ::ℜ=nanmean(X), γ::ℜ=0.1, adaptive::Bool=true) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)
    T::Ƶ = Ƶ(sum(X))

    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)

    a::ℜ, b::ℜ = I*J*γ, γ/μ
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(lgamma.(X .+ 1.0))

    P = [Particle(I,J,K; γ=γ) for n=1:N]
    P_temp = [Particle(I,J,K; γ=γ) for n=1:N]

    log_wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(-log(N),N)
    Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(1.0/N,N), zeros(ℜ,N)

    log_νᵗ::ℜ, log_q_k::Array{ℜ}, q_k::Array{ℜ}= 0.0, zeros(ℜ,K), zeros(ℜ,K)
    iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ  = 0, 0, 0, 0.0

    for (t,(iᵗ, jᵗ)) in enumerate(EventQueue(X))
        for (n,p) in enumerate(P)
            log_q_k .= log.(p.S_JK[jᵗ,:]) .+ log.(p.S_IK[iᵗ,:]) .- log.(p.S_K)
            log_νᵗ = logsumexp(log_q_k)
            log_q_k .-= log_νᵗ
            q_k .= exp.(log_q_k)
            kᵗ = rand(Categorical(q_k))

            p.S_K[kᵗ] += 1.0
            p.S_JK[jᵗ,kᵗ] += 1.0 
            p.S_IK[iᵗ,kᵗ] += 1.0

            log_wᵗ[n] += log_νᵗ
        end
        
        log_Zᵗ = logsumexp(log_wᵗ)
        log_Wᵗ .= log_wᵗ .- log_Zᵗ
        log_Zᵗ -= log(N)
        Wᵗ .= exp.(log_Wᵗ)

        ESS[t] = 1.0/sum(Wᵗ .* Wᵗ)
        
        if ESS[t] < ESS_min[t]
            for n=1:N
                P_temp[n].S_K .= P[n].S_K
                P_temp[n].S_JK .= P[n].S_JK
                P_temp[n].S_IK .= P[n].S_IK
            end
            
            cum_Wᵗ .= cumsum(Wᵗ)
            uᵗ = rand()/N
            p_id = 1
            
            for n=1:N # systematic resampling
                while cum_Wᵗ[p_id] < uᵗ
                    p_id += 1
                end
                P[n].S_K .= P_temp[p_id].S_K
                P[n].S_JK .= P_temp[p_id].S_JK
                P[n].S_IK .= P_temp[p_id].S_IK
                uᵗ += 1.0/N
            end
            log_wᵗ .= log_Zᵗ
        end
    end
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_K .-= a/K
    P[p_id].S_JK .-= a/(K*J)
    P[p_id].S_IK .-= a/(I*K)
    return log_Zᵗ, ESS, P[p_id]
end

function particle_mcmc(X::Array{ℜ,2}, K::Ƶ, N::Ƶ=1; μ::ℜ=nanmean(X), γ::ℜ=0.1, EPOCHS::Ƶ=1, adaptive::Bool=false) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ = size(X)
    a::ℜ, b::ℜ = I*J*γ, γ/μ
    
    log_PX::Array{ℜ}, stays::Ƶ = zeros(ℜ, EPOCHS), 0
    log_PX[1], ESS::Array{ℜ}, part::Particle{ℜ} = smc_weight(X, K, N; μ=μ, γ=γ, adaptive=adaptive)
    log_PX_prop::ℜ, ESS_prop::Array{ℜ}, part_prop::Particle{ℜ} = smc_weight(X, K, N; μ=μ, γ=γ, adaptive=adaptive)
    
    for eph=2:EPOCHS
        log_PX_prop, ESS_prop, part_prop = smc_weight(X, K, N; μ=μ, γ=γ, adaptive=adaptive)
        
        if log(rand()) < log_PX_prop - log_PX[eph-1]
            log_PX[eph] = log_PX_prop
            ESS .= ESS_prop
            part.S_K .= part_prop.S_K
            part.S_IK .= part_prop.S_IK
            part.S_JK .= part_prop.S_JK
        else
            log_PX[eph] = log_PX[eph-1]
            stays += 1
        end
    end
    log_Z_τ::ℜ = logsumexp(log_PX) - log(EPOCHS)
    return log_Z_τ, log_PX, 1.0 - stays./EPOCHS, part
end

end
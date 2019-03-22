module SCP_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions

export Particle, smc_weight, particle_mcmc

mutable struct Particle{ℜ <: Real}
    S_KR::Array{ℜ,2}
    S_IKR::Array{ℜ,3}
    S_JKR::Array{ℜ,3}
    function Particle(I::Int,J::Int, K::Int, R::Int; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*J*K*γ
        return new{ℜ}(fill(a/(R*K),K,R), fill(a/(I*K*R),I,K,R), fill(a/(J*K*R),J,K,R))
    end
    function Particle(S_KR::Array{ℜ,2}, S_IKR::Array{ℜ,3}, S_JKR::Array{ℜ,3}) where {ℜ<:Real}
        return new{ℜ}(S_KR, S_IKR, S_JKR)
    end
end

function smc_weight(X::Array{ℜ,3}, X⁺::Array{ℜ,2}, R::Ƶ, N::Ƶ=1; μ::ℜ=3.5, γ::ℜ=0.1, adaptive::Bool=true) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    T::Ƶ = Ƶ(sum(X⁺))
    
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)

    P = [Particle(I,J,K,R; μ=μ, γ=γ) for n=1:N]
    P_temp = [Particle(I,J,K,R; μ=μ, γ=γ) for n=1:N]
    
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0)
    
    for j=1:J, i=1:I
        log_Zᵗ -= isnan(X[i,j,1]) ? lgamma(X⁺[i,j] + 1.0) : sum(lgamma.(X[i,j,:] .+ 1.0)) 
    end
    
    log_wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(-log(N),N)
    Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(1.0/N,N), zeros(ℜ,N)
    
    log_νᵗ::ℜ, log_q_r::Array{ℜ}, q_r::Array{ℜ} = 0.0, zeros(ℜ,R), zeros(ℜ,R)
    log_q_k::Array{ℜ}, q_k::Array{ℜ} = zeros(ℜ,K), zeros(ℜ,K)
    rᵗ::Ƶ, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ = 0, 0, 0, 0, 0.0

    for (t,(iᵗ, jᵗ, kᵗ)) in enumerate(EventQueue(X, X⁺))
        for (n,p) in enumerate(P)

            if kᵗ == 0
                log_q_k .= logsumexp(log.(p.S_IKR[iᵗ,:,:]) .+ log.(p.S_JKR[jᵗ,:,:]) .- log.(p.S_KR),2)[:,1]
                log_νᵗ = logsumexp(log_q_k)
                log_q_k .-= log_νᵗ
                q_k .= exp.(log_q_k)
                kᵗ = rand(Categorical(q_k))

                log_q_r .= log.(p.S_IKR[iᵗ,kᵗ,:]) .+ log.(p.S_JKR[jᵗ,kᵗ,:]) .- log.(p.S_KR[kᵗ,:])
                log_q_r .-= logsumexp(log_q_r)
            else
                log_q_r .= log.(p.S_IKR[iᵗ,kᵗ,:]) .+ log.(p.S_JKR[jᵗ,kᵗ,:]) .- log.(p.S_KR[kᵗ,:])
                log_νᵗ = logsumexp(log_q_r)
                log_q_r .-= log_νᵗ
            end

            q_r .= exp.(log_q_r)
            rᵗ = rand(Categorical(q_r))

            p.S_KR[kᵗ,rᵗ] += 1.0 
            p.S_IKR[iᵗ,kᵗ,rᵗ] += 1.0 
            p.S_JKR[jᵗ,kᵗ,rᵗ] += 1.0 

            log_wᵗ[n] += log_νᵗ
        end
        
        log_Zᵗ = logsumexp(log_wᵗ)
        log_Wᵗ .= log_wᵗ .- log_Zᵗ
        log_Zᵗ -= log(N)
        Wᵗ .= exp.(log_Wᵗ)

        ESS[t] = 1.0/sum(Wᵗ .* Wᵗ)
        
        if ESS[t] < ESS_min[t]
            for n=1:N
                P_temp[n].S_KR .= P[n].S_KR
                P_temp[n].S_IKR .= P[n].S_IKR
                P_temp[n].S_JKR .= P[n].S_JKR
            end
            
            cum_Wᵗ .= cumsum(Wᵗ)
            uᵗ = rand()/N
            p_id = 1
            
            for n=1:N # systematic resampling
                while cum_Wᵗ[p_id] < uᵗ
                    p_id += 1
                end
                P[n].S_KR .= P_temp[p_id].S_KR
                P[n].S_IKR .= P_temp[p_id].S_IKR
                P[n].S_JKR .= P_temp[p_id].S_JKR
                uᵗ += 1.0/N
            end
            log_wᵗ .= log_Zᵗ
        end
    end
    p_id = rand(Categorical(Wᵗ))
    #p_w, p_id = findmax(Wᵗ)
    P[p_id].S_KR .-= a/(K*R)
    P[p_id].S_IKR .-= a/(I*K*R)
    P[p_id].S_JKR .-= a/(J*K*R)
    return log_Zᵗ, ESS, P[p_id]
end
    
function particle_mcmc(X::Array{ℜ,3}, X⁺::Array{ℜ,2}, R::Ƶ, N::Ƶ=1; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, adaptive=false) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    
    log_PX::Array{ℜ}, stays::Ƶ = zeros(ℜ, EPOCHS), 0
    log_PX[1], ESS::Array{ℜ}, part::Particle{ℜ} = smc_weight(X, X⁺, R, N; μ=μ, γ=γ, adaptive=adaptive)
    log_PX_prop::ℜ, ESS_prop::Array{ℜ}, part_prop::Particle{ℜ} = smc_weight(X, X⁺, R, N; μ=μ, γ=γ, adaptive=adaptive)
    
    for eph=2:EPOCHS
        log_PX_prop, ESS_prop, part_prop = smc_weight(X, X⁺, R, N; μ=μ, γ=γ, adaptive=adaptive)
        
        if log(rand()) < log_PX_prop - log_PX[eph-1]
            log_PX[eph] = log_PX_prop
            ESS .= ESS_prop
            part.S_KR .= part_prop.S_KR 
            part.S_IKR .= part_prop.S_IKR
            part.S_JKR .= part_prop.S_JKR
        else
            log_PX[eph] = log_PX[eph-1]
            stays += 1
        end
    end

    log_Z_τ::ℜ = logsumexp(log_PX) - log(EPOCHS)
    return log_Z_τ, log_PX, 1.0 - stays./EPOCHS, part
end

end
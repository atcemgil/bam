module CP_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions

export Particle, smc_weight, particle_mcmc

mutable struct Particle{ℜ <: Real}
    S_R::Array{ℜ,1}
    S_IR::Array{ℜ,2}
    S_JR::Array{ℜ,2}
    S_KR::Array{ℜ,2}
    function Particle(I::Int,J::Int, K::Int, R::Int; γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*J*K*γ
        return new{ℜ}(fill(a/R,R), fill(a/(R*I),I,R), fill(a/(R*J),J,R), fill(a/(R*K),K,R))
    end
    function Particle(S_R::Array{ℜ,1}, S_IR::Array{ℜ,2}, S_JR::Array{ℜ,2}, S_KR::Array{ℜ,2}) where {ℜ<:Real}
        return new{ℜ}(S_R, S_IR, S_JR, S_KR)
    end
end

function smc_weight(X::Array{ℜ,3}, R::Ƶ, N::Ƶ=1; μ::ℜ=3.5, γ::ℜ=0.1, adaptive::Bool=true) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    T::Ƶ = Ƶ(sum(X))
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(lgamma.(X .+ 1.0)) 
    P = [Particle(I,J,K,R; γ=γ) for n=1:N]
    P_temp = [Particle(I,J,K,R; γ=γ) for n=1:N]
    log_wᵗ::Array{ℜ}, Wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(1.0/N,N), fill(-log(N),N), zeros(ℜ,N)

    log_νᵗ::ℜ, log_q_r::Array{ℜ}, q_r::Array{ℜ}= 0.0, zeros(ℜ,R), zeros(ℜ,R)
    rᵗ::Ƶ, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ  = 0, 0, 0, 0, 0.0

    for (t,(iᵗ, jᵗ, kᵗ)) in enumerate(EventQueue(X))
        for (n,p) in enumerate(P)
                   
            log_q_r .= log.(p.S_IR[iᵗ,:]) .+ log.(p.S_JR[jᵗ,:]) .+ log.(p.S_KR[kᵗ,:]) .- 2.0.*log.(p.S_R)
            log_νᵗ = logsumexp(log_q_r)
            log_q_r .-= log_νᵗ
            q_r .= exp.(log_q_r)

            rᵗ = rand(Categorical(q_r))
            p.S_R[rᵗ] += 1.0 
            p.S_IR[iᵗ,rᵗ] += 1.0 
            p.S_JR[jᵗ,rᵗ] += 1.0 
            p.S_KR[kᵗ,rᵗ] += 1.0 

            log_wᵗ[n] += log_νᵗ
        end
        
        log_Zᵗ = logsumexp(log_wᵗ)
        log_Wᵗ .= log_wᵗ .- log_Zᵗ
        log_Zᵗ -= log(N)
        Wᵗ .= exp.(log_Wᵗ)

        ESS[t] = 1.0/sum(Wᵗ .* Wᵗ)
        
        if ESS[t] < ESS_min[t]
            for n=1:N
                P_temp[n].S_R .= P[n].S_R
                P_temp[n].S_IR .= P[n].S_IR
                P_temp[n].S_JR .= P[n].S_JR
                P_temp[n].S_KR .= P[n].S_KR
            end
            
            cum_Wᵗ .= cumsum(Wᵗ)
            uᵗ = rand()/N
            p_id = 1
            
            for n=1:N # systematic resampling
                while cum_Wᵗ[p_id] < uᵗ
                    p_id += 1
                end
                P[n].S_R .= P_temp[p_id].S_R
                P[n].S_IR .= P_temp[p_id].S_IR
                P[n].S_JR .= P_temp[p_id].S_JR
                P[n].S_KR .= P_temp[p_id].S_KR
                uᵗ += 1.0/N
            end
            log_wᵗ .= log_Zᵗ
        end
    end
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_R .-= a/R
    P[p_id].S_IR .-= a/(R*I)
    P[p_id].S_JR .-= a/(R*J)
    P[p_id].S_KR .-= a/(R*K)
    return log_Zᵗ, ESS, P[p_id]
end

function particle_mcmc(X::Array{ℜ,3}, R::Ƶ, N::Ƶ=1; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, adaptive=false) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    
    log_PX::Array{ℜ}, stays::Ƶ = zeros(ℜ, EPOCHS), 0
    log_PX[1], ESS::Array{ℜ}, part::Particle{ℜ} = smc_weight(X, R, N; μ=μ, γ=γ, adaptive=adaptive)
    log_PX_prop::ℜ, ESS_prop::Array{ℜ}, part_prop::Particle{ℜ} = smc_weight(X, R, N; μ=μ, γ=γ, adaptive=adaptive)
    
    for eph=2:EPOCHS
        log_PX_prop, ESS_prop, part_prop = smc_weight(X, R, N; μ=μ, γ=γ, adaptive=adaptive)
        
        if log(rand()) < log_PX_prop - log_PX[eph-1]
            log_PX[eph] = log_PX_prop
            ESS .= ESS_prop
            part.S_R .= part_prop.S_R 
            part.S_IR .= part_prop.S_IR 
            part.S_JR .= part_prop.S_JR
            part.S_KR .= part_prop.S_KR
        else
            log_PX[eph] = log_PX[eph-1]
            stays += 1
        end
    end

    log_Z_τ::ℜ = logsumexp(log_PX) - log(EPOCHS)
    return log_Z_τ, log_PX, 1.0 - stays./EPOCHS, part
end

end
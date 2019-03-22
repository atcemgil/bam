module sNMF_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions

export Particle, smc_weight, particle_mcmc

mutable struct Particle{ℜ <: Real}
    S_R::Array{ℜ,1}
    S_IR::Array{ℜ,2}
    function Particle(I::Int, R::Int; γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*I*γ
        return new{ℜ}(fill(a/R,R), fill(2.0 * a/(I*R),I,R))
    end
    function Particle(S_R::Array{ℜ,1}, S_IR::Array{ℜ,2}) where {ℜ<:Real}
        return new{ℜ}(S_R, S_IR)
    end
end

function smc_weight(X::Array{ℜ,2}, R::Ƶ, N::Ƶ=1; μ::ℜ=1.0, γ::ℜ=0.1, adaptive::Bool=true) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ = size(X,1)
    T::Ƶ = Ƶ(sum(X))
    
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)

    P = [Particle(I,R; γ=γ) for n=1:N]
    P_temp = [Particle(I,R; γ=γ) for n=1:N]
    
    a::ℜ, b::ℜ = I*I*γ, γ/μ
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(lgamma.(X .+ 1.0))
    
    log_wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(-log(N),N)
    Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(1.0/N,N), zeros(ℜ,N)
    
    log_νᵗ::ℜ, log_q_r::Array{ℜ}, q_r::Array{ℜ} = 0.0, zeros(ℜ,R), zeros(ℜ,R)
    rᵗ::Ƶ, i₁ᵗ::Ƶ, i₂ᵗ::Ƶ, uᵗ::ℜ = 0, 0, 0, 0.0

    for (t,(i₁ᵗ, i₂ᵗ)) in enumerate(EventQueue(X))
        for (n,p) in enumerate(P)

            log_q_r .= log.(p.S_IR[i₁ᵗ,:]) .+ log.(p.S_IR[i₂ᵗ,:] .+ (i₁ᵗ == i₂ᵗ)) .- log.(4.0 .* p.S_R .+ 2)            
            log_νᵗ = logsumexp(log_q_r)
            log_q_r .-= log_νᵗ

            q_r .= exp.(log_q_r)
            rᵗ = rand(Categorical(q_r))

            p.S_R[rᵗ] += 1.0 
            p.S_IR[i₁ᵗ,rᵗ] += 1.0 
            p.S_IR[i₂ᵗ,rᵗ] += 1.0 

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
                
                uᵗ += 1.0/N
            end
            log_wᵗ .= log_Zᵗ
        end
    end
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_R .-= a/R
    P[p_id].S_IR .-= 2.0 .* a/I*R
    return log_Zᵗ, ESS, P[p_id]
end

end
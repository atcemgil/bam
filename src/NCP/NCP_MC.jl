module NCP_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions
import Base.Iterators: product

export Particle, smc_weight, filtering_proposal, resample

mutable struct Particle{ℜ <: Real}
    S_R::Array{ℜ}
    S_I::Array{Array{ℜ}}
    function Particle(X::Array, R::Int; a::ℜ=0.0) where {ℜ <: Real}
        return new{ℜ}(fill(a/R,R), [fill(a/(R*Iₙ),R,Iₙ) for Iₙ ∈ size(X)])
    end
    function Particle(S_R::Array{ℜ}, S_I::Array{Array{ℜ}}) where {ℜ <: Real}
        return new{ℜ}(S_R, S_I)
    end
end

function resample(W::Array,Π::Array{Particle{ℜ}},u=rand()) where {ℜ <: Real} # systematic resampling
    P, N = length(W), length(Π[1].S_I)
    j = 0
    cum_Wⱼ = cum_Wᵢ = -u
    for i ∈ 1:P
        rᵢ = ceil(cum_Wᵢ + P*W[i]) - ceil(cum_Wᵢ) # number of replicas for ith particle
        for _ ∈ 2.0:rᵢ
            j+=1
            while ceil(cum_Wⱼ+ P*W[j]) - ceil(cum_Wⱼ) > 0 # find next j to be replaced
                cum_Wⱼ += P*W[j]
                j+=1
            end
            cum_Wⱼ += P*W[j]
            
            # replace j by i
            Π[j].S_R .= Π[i].S_R
            for n ∈ 1:N
                Π[j].S_I[n] .= Π[i].S_I[n]
            end
        end
        cum_Wᵢ += P*W[i]
    end
end

function smc_weight(X::Array{ℜ,N}, R::Ƶ, P::Ƶ=1; a=1.0, b=a/nansum(X), resampling=true) where {ℜ<:Real, Ƶ<:Integer,N}
    T = Ƶ(round(sum(X)))
    ESS = P
    
    Π = [Particle(X, R; a=a) for p=1:P]
    log_Z = a*log(b) - (a+T)*log(b + 1) - sum(lgamma, X .+ 1)

    log_w, W = fill(log_Z,P), fill(1.0/P,P)
    log_q, q = zeros(R), zeros(R)
    
    for i ∈ EventQueue(X)
        for p ∈ 1:P
            log_q .= (1-N) .* log.(Π[p].S_R)
            for  n ∈ 1:N
                log_q .+= log.(Π[p].S_I[n][:,i[n]])
            end
            
            log_ν = logsumexp(log_q)
            log_q .-= log_ν
            q .= exp.(log_q)
            r = rand(Categorical(q))
                
            Π[p].S_R[r] += 1.0
            for n ∈ 1:N
                Π[p].S_I[n][r,i[n]] += 1.0
            end
            
            log_w[p] += log_ν
        end
        
        log_Z = logmeanexp(log_w)
        W .= exp.(log_w .- logsumexp(log_w))
        ESS = 1.0/sum(W .* W)
        
        if resampling && ESS < P/2
            resample(W,Π)
            log_w .= log_Z
        end
    end
    p = rand(Categorical(W))
    return log_Z, ESS, Π[p]
end

function filtering_proposal(X::Array{ℜ,N}, R::Ƶ, P::Ƶ=1; a=1.0, b=a/nansum(X)) where {ℜ<:Real, Ƶ<:Integer,N}
    I = size(X)    
    α_I = (a/R) ./ I
    α_R = a/R

    S, S_R, S_I = zeros(ℜ,R,I...), zeros(ℜ,R), [zeros(ℜ,R,Iₙ) for Iₙ ∈ size(X)]

    S₊ = sum(X)  
    log_Z = a*log(b) - (a+S₊)*log(b + 1) + lgamma(a + S₊) - lgamma(a)
    log_w, log_q, q = fill(log_Z,P), zeros(R), zeros(R)
    
    for p ∈ 1:P
        S .= 0
        S_R .= 0 
        for n ∈ 1:N
            S_I[n] .= 0
        end

        for (t,i) in enumerate(EventQueue(X))
            log_q .=  sum(n -> log.(S_I[n][:,i[n]] .+ α_I[n]),1:N) - (N-1)*log.(S_R .+ α_R) .- log(t+a-1)
            q .= exp.(log_q .- logsumexp(log_q))
            r = rand(Categorical(q))

            S_R[r] += 1
            for n ∈ 1:N
                S_I[n][r,i[n]] += 1
            end
            S[r,i...] += 1
            
            log_w[p] += log_q[r]
        end
        log_w[p] -= sum(lgamma,S .+ 1)
    end
    return log_w
end

end
module ACP_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions
import Base.Iterators: product

export Particle, smc_weight

mutable struct Particle{ℜ <: Real}
    S_R::Array{ℜ,1}
    S_I::Array{Array{ℜ,2}}
    allocations
    function Particle(X::Array{ℜ}, R::Ƶ) where {ℜ<:Real, Ƶ<:Int}
        return new{ℜ}(zeros(ℜ,R), [zeros(ℜ,R,Iₙ) for Iₙ ∈ size(X)], [Tuple(zeros(length(size(X))+2)) for i in 1:sum(X)])
    end
    function Particle(S_R::Array{ℜ,1}, S_I::Array{Array{ℜ,2}}) where {ℜ<:Real}
        return new{ℜ}(S_R, S_I, [Tuple(zeros(length(S_I)+2)) for i in 1:sum(S_R)])
    end
end

function smc_weight(X::Array{ℜ,N}, R::Ƶ, P::Ƶ=1; a=1.0, b=a/nansum(X), debug=true) where {ℜ<:Real, Ƶ<:Integer,N}
    X_full = deepcopy(X)
    X = sum(X_full, N)
    I = size(X)

    α_I = (a/R) ./ I
    α_R = a/R
    
    T = Ƶ(round(sum(X)))
    ESS = zeros(T)
    
    Π::Array{Particle} = [Particle(X, R) for p=1:P]
    Π₂::Array{Particle} = [Particle(X, R) for p=1:P]
    
    log_Z = a*log(b) - (a+T)*log(b + 1) + lgamma(a + T) - lgamma(a) - sum(lgamma, X .+ 1)

    log_w, W, cum_W = fill(log_Z,P), fill(1.0/P,P), zeros(P)
    log_q, q = zeros(R), zeros(R)
    r = 1
    for (t,i) in enumerate(EventQueue(X_full))
        z = i[end]
        i = i[1:end-1]
        for (p,π) ∈ enumerate(Π)

            log_q .=  sum(n -> log.(π.S_I[n][:,i[n]] .+ α_I[n]),1:(N-1)) - (N-1-1)*log.(π.S_R .+ α_R) .- log(t+a-1)
            log_ν = logsumexp(log_q)
            log_q .-= log_ν 
            
            q .= exp.(log_q)
            r = rand(Categorical(q))
            
            π.S_R[r] += 1
            for n ∈ 1:(N-1)
                π.S_I[n][r,i[n]] += 1
            end
            
            log_w[p] += log_ν
            π.allocations[t] = Tuple([i...,r,z])
        end

        log_Z = logmeanexp(log_w)
        W .= exp.(log_w .- logsumexp(log_w))

        ESS[t] = 1.0/sum(W .* W)

        if ESS[t] < P/2
            cum_W .= cumsum(W)
            u = rand()/P
            
            for p ∈ 1:P
                Π₂[p].S_R .= Π[p].S_R
                for n ∈ 1:(N-1)
                    Π₂[p].S_I[n] .= Π[p].S_I[n]
                end
            end
            
            p₂ = 1
            for p ∈ 1:P # systematic resampling
                while cum_W[p₂] < u
                    p₂ += 1
                end
                Π[p].S_R .= Π₂[p₂].S_R
                for n ∈ 1:(N-1)
                    Π[p].S_I[n] .= Π₂[p₂].S_I[n]
                end
                u += 1.0/P
            end
            log_w .= log_Z
        end
    end
    p = rand(Categorical(W))
    return log_Z, ESS, Π[p]
end
end

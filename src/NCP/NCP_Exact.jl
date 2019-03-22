module NCP_Exact

include("../Misc.jl")
include("NCP_MC.jl")
include("NCP_VB.jl")

using .Misc
import .NCP_MC, .NCP_VB

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export allocations, full, generate
export log_marginal, alloc_dist, EP_dist
export log_posterior_T, log_PT

allocations(X::Array{ℜ,N}, H::Vararg{Ƶ}) where {ℜ<:Real, Ƶ<:Integer, N} = Channel(ctype=Array{ℜ,N+length(H)}) do c
    I, R = size(X), prod(H)
    for partition = product(map(Xᵢ -> combinations(1:(Xᵢ+R-1),R-1), X)...)
        push!(c,reshape([sᵢ for (i,pᵢ) ∈ enumerate(partition) 
                        for sᵢ = diff([0, pᵢ..., X[i]+R]) .- 1],H...,I...))
    end
end

full(X::Array{ℜ,N}, T::ℜ) where {ℜ<:Real, N} = Channel(ctype=Array{ℜ,N}) do c
    miss = findall(isnan,X)
    M = length(miss)
    Tₘ = Int.(T - nansum(X))
    for partition ∈ combinations(1:(Tₘ+M-1),M-1)
        X_full = copy(X)
        X_full[miss] .= diff([0, partition..., Tₘ+M]) .- 1
        push!(c,X_full)
    end
end

function generate(T::Ƶ, R::Ƶ, I::Vararg{Ƶ}; a::ℜ=1.0) where {ℜ<:Real, Ƶ<:Integer}
    α_I = (a/R) ./ I
    α_R = a/R
    
    θ_R = rand(Dirichlet(R,α_R))
    θ_I = [rand(Dirichlet(Iₙ,α_Iₙ),R) for (Iₙ,α_Iₙ) in zip(I,α_I)]
                                    
    D_R = rand(Categorical(θ_R),T)
    D_I = [rand(Categorical(θ_Iₙ[:,r])) for r ∈ D_R, θ_Iₙ ∈ θ_I]
    
    return cooccurrences(D_I,I), (θ_R, θ_I...)
end

function log_marginal(X::Array{ℜ,N}, R::Ƶ; a=1.0, b=a/nansum(X)) where {ℜ<:Real, Ƶ<:Integer, N}
    I = size(X)
    α_I = (a/R) ./ I
    α_R = a/R
    
    S_R, S₊ = Array{ℜ}(undef,R), sum(X)
    S_I = [Array{ℜ}(undef,R,Iₙ) for Iₙ in I]
    
    log_PX = -Inf
    log_C = a*log(b) - (a+S₊)*log(b + 1) + lgamma(a + S₊) - lgamma(a) 
    
    for S ∈ allocations(X,R)
        log_PS = log_C - sum(lgamma, S .+ 1)
        for (n, Iₙ) ∈ enumerate(I)
            S_I[n] .= sum(S,(2:n...,n+2:N+1...))
            log_PS += sum(lbeta(α_I[n] .+ S_I[n];dims=2)) - R*lbeta(α_I[n],Iₙ)
        end
        S_R .= sum(S,(2:N+1...,))
        log_PS += lbeta(α_R .+ S_R) - lbeta(α_R,R)
        log_PX = logsumexp([log_PX,log_PS]) 
    end
    return log_PX
end

function alloc_dist(X::Array{ℜ,N}, R::Ƶ; a=1.0, b=a/nansum(X)) where {ℜ<:Real, Ƶ<:Integer, N}
    I = size(X)
    α_I = (a/R) ./ I
    α_R = a/R
    
    S_R, S₊ = Array{ℜ}(undef,R), sum(X)
    S_I = [Array{ℜ}(undef,R,Iₙ) for Iₙ in I]
    
    nS = prod(x -> binomial(Ƶ(x+R-1),R-1), X)
    log_P = zeros(nS)
    
    log_C = a*log(b) - (a+S₊)*log(b + 1) + lgamma(a + S₊) - lgamma(a) 
    
    for (s, S) ∈ enumerate(allocations(X,R))
        log_P[s] = log_C - sum(lgamma, S .+ 1)
        for (n, Iₙ) ∈ enumerate(I)
            S_I[n] .= sum(S,(2:n...,n+2:N+1...))
            log_P[s] += sum(lbeta(α_I[n] .+ S_I[n];dims=2)) - R*lbeta(α_I[n],Iₙ)
        end
        S_R .= sum(S,(2:N+1...,))
        log_P[s] += lbeta(α_R .+ S_R) - lbeta(α_R,R)
    end
    return log_P
end

function EP_dist(X::Array{ℜ,N}, R::Ƶ) where {ℜ<:Real, Ƶ<:Integer, N}
    I = size(X)

    S_R, S₊ = Array{ℜ}(undef,R), sum(X)
    S_I = [Array{ℜ}(undef,R,Iₙ) for Iₙ in I]
    
    nS = prod(x -> binomial(Ƶ(x+R-1),R-1), X)
    dₑₚ = zeros(nS)
    
    for (s, S) ∈ enumerate(allocations(X,R))
        for (n, Iₙ) ∈ enumerate(I)
            S_I[n] .= sum(S,(2:n...,n+2:N+1...))
            dₑₚ[s] += sum(!iszero, S_I[n]) 
        end
        S_R .= sum(S,(2:N+1...,))
        dₑₚ[s] += (N-1) * sum(!iszero, S_R) + !iszero(S₊)
    end
    return dₑₚ
end

function log_posterior_T(X::Array{ℜ,N}, T::ℜ , R::Ƶ; a=1.0, b=a/nansum(X), 
    smc::Bool=false, elbo::Bool=false, particles::Int=1000, EPOCHS::Int=100, M::Int=100) where {ℜ<:Real, Ƶ<:Integer, N}
    if smc
        log_PX = [NCP_MC.smc_weight(X_full,R,particles; a=a, b=b)[1] for X_full ∈ full(X,T)]
        return logsumexp(log_PX)
    elseif elbo
        log_PX = nanmax([NCP_VB.standard_VB(X_full,R; a=a, b=b, EPOCHS=EPOCHS)[1][end] for m ∈ 1:M, X_full ∈ full(X,T)],1)
        return logsumexp(log_PX)
    else
        log_PX = [log_marginal(X_full,R; a=a, b=b) for X_full ∈ full(X,T)]
        return logsumexp(log_PX)
    end
end

end
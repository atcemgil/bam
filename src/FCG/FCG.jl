module FCG

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export generate, log_marginal

function generate(T::Ƶ, dims::Vararg{Ƶ}; γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    len = prod(dims)
    θ_flat = rand(Dirichlet(γ,len))

    X::Array{ℜ} = reshape(rand(Multinomial(T,θ_flat)),dims)

    return X, reshape(θ_flat,dims)
end

function log_marginal(X::Array{ℜ,N}; μ::ℜ=nanmean(X), γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int, N}
    dims, len = size(X), length(X)
    X₊ = sum(X)
    
    a::ℜ, b::ℜ = len*γ, γ/μ
    
    log_PX = a*log(b) - (a + X₊)*log(b + 1.0) + lgamma(a + X₊) - lgamma(a) - sum(lgamma.(X .+ 1.0))
    log_PX += sum(lgamma.(γ .+ X)) - lgamma(a + X₊) + lgamma(a) - len*lgamma(γ)
    
    return log_PX
end

end
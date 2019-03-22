module Misc

using Statistics, Clustering
using JSON
import DataStructures: PriorityQueue, peek
import SpecialFunctions: lbeta, lgamma
import Base: sum, maximum, minimum
import Statistics: mean
import LinearAlgebra: norm

export nanmean, nanmax, nansum, nanmin, sum, mean, maximum, minimum
export logmeanexp, logvarexp, logsumexp, lbeta
export KL, rmse, mae, hoyer
export discretize, categorize, cooccurrences
export finucan, SNR
export save_json, load_json

@inline sum(X::AbstractArray,dims) = dropdims(sum(X;dims=dims); dims=dims)
@inline mean(X::AbstractArray,dims) = dropdims(mean(X;dims=dims); dims=dims)
@inline maximum(X::AbstractArray,dims) = dropdims(maximum(X;dims=dims); dims=dims)
@inline minimum(X::AbstractArray,dims) = dropdims(minimum(X;dims=dims); dims=dims)

@inline _nanmean(X::AbstractArray) = mean(filter(!isnan,X))
nanmean(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmean(X) : mapslices(_nanmean, X; dims=dims)
nanmean(X::AbstractArray, dims) = dropdims(nanmean(X; dims=dims),dims=dims)

@inline _nansum(X::AbstractArray) = sum(filter(!isnan,X))
nansum(X::AbstractArray; dims=nothing) = dims == nothing ? _nansum(X) : mapslices(_nansum, X; dims=dims)
nansum(X::AbstractArray, dims) = dropdims(nansum(X; dims=dims),dims=dims)

@inline _nanmax(X::AbstractArray) = maximum(filter(!isnan,X))
nanmax(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmax(X) : mapslices(_nanmax, X; dims=dims)
nanmax(X::AbstractArray, dims) = dropdims(nanmax(X; dims=dims),dims=dims)

@inline _nanmin(X::AbstractArray) = minimum(filter(!isnan,X))
nanmin(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmin(X) : mapslices(_nanmin, X; dims=dims)
nanmin(X::AbstractArray, dims) = dropdims(nanmin(X; dims=dims),dims=dims)

@inline _logsumexp(X::AbstractArray) = begin x⁺ = maximum(X); log(sum(exp, X .- x⁺)) + x⁺ end
logsumexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logsumexp(X) : mapslices(_logsumexp, X; dims=dims)
logsumexp(X::AbstractArray, dims) = dropdims(logsumexp(X; dims=dims),dims=dims)

@inline _logmeanexp(X::AbstractArray) = _logsumexp(X) - log(length(X))
logmeanexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logmeanexp(X) : mapslices(_logmeanexp, X; dims=dims)
logmeanexp(X::AbstractArray, dims) = dropdims(logmeanexp(X; dims=dims),dims=dims)

@inline _logvarexp(X::AbstractArray; μ=_logmeanexp(X)) = begin EX = _logsumexp(X); log(var(exp.(X .- EX); mean=exp(μ - EX))) + 2.0 * EX end
logvarexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logvarexp(X) : mapslices(_logvarexp, X; dims=dims)
logvarexp(X::AbstractArray, dims) = dropdims(logvarexp(X; dims=dims),dims=dims)

@inline _lbeta(X::AbstractArray) =  sum(lgamma, X) - lgamma(sum(X))
lbeta(X::AbstractArray; dims=nothing) = dims == nothing ? _lbeta(X) : mapslices(_lbeta, X; dims=dims)
lbeta(X::AbstractArray, dims) = dropdims(lbeta(X; dims=dims),dims=dims)

@inline _lbeta(γ::Number,I::Integer) = I*lgamma(γ) - lgamma(I*γ)
lbeta(γ::Number, sz::Vararg{Integer}; dims=nothing) = dims == nothing ? _lbeta(γ,prod(sz)) : fill(_lbeta(γ,prod(sz[dims])), map(t -> t[1] in dims ? 1 : t[2], enumerate(sz))...)

    
function rmse(X::AbstractArray, Xᵖ::AbstractArray)
    return sqrt(nanmean((X .- Xᵖ) .^ 2 ))
end
    
function mae(X::AbstractArray, Xᵖ::AbstractArray)
    return nanmean(abs.(X .- Xᵖ))
end

function KL(X::AbstractArray, Xᵖ::AbstractArray)
    KL_div = 0.0
    for (x,xᵖ) in zip(X,Xᵖ)
        if !isnan(x) && !isnan(xᵖ)
            KL_div += (x ≈ 0.0) ? 0.0 : x * (log(x) - log(xᵖ)) + xᵖ - x
        end
    end
    return KL_div
end

function SNR(X::AbstractArray, Xᵖ::AbstractArray; base=10.0)
    return 2.0*base*(log(base,norm(X)) - log(base,norm(X .- Xᵖ)))
end

function hoyer(X::AbstractArray)
    N = length(X)
    return (sqrt(N) - sum(abs,X)/sqrt(sum(X .* X))) / (sqrt(N) - 1)
end

function discretize(X::AbstractArray{ℜ}, L::Ƶ; ϵ=1e-10) where {ℜ <: Real, Ƶ <: Integer}
    Xₙ = X .- minimum(X) .+ ϵ
    Xₙ ./= maximum(Xₙ) + ϵ
    return Ƶ.(floor.(L*Xₙ)) .+ 1 
end

function categorize(X::AbstractArray, L::Ƶ=2; cluster::Bool=true) where {Ƶ <: Integer}
    if eltype(X) <: AbstractFloat
    	if cluster
        	Mₓ = kmeans(reshape(X,1,:),L)
        	Categories = Dict(zip(sortperm(Mₓ.centers[:]),1:L))
        	return map(x -> Categories[x], reshape(Mₓ.assignments, size(X)))
        else
        	return discretize(X,L)
        end
    elseif eltype(X) <: Integer
            return X .- minimum(X) .+ 1
    else
        Labels = sort(unique(X))
        Categories = Dict(zip(Labels,1:length(Labels)))
        return map(x -> Categories[x], X)
    end
end

function cooccurrences(Tab::AbstractArray{Ƶ,2},dims) where {Ƶ <: Integer}
    T, N = size(Tab)
    X = zeros(Ƶ,dims...)
    for t in 1:T
        X[Tab[t,:]...] += 1
    end
    return X
end

function cooccurrences(Tab::AbstractArray{Ƶ,2}) where {Ƶ <: Integer}
	I = nanmax(Tab;dims=1)[1,:]
	return cooccurrences(Tab,I)
end

function finucan(N::ℜ, p::Array{ℜ}; ϵ::ℜ=1e-16) where {ℜ<:Real}
    K, k = length(p), 0
    p ./= sum(p) + ϵ
    Sᴿ = (N + K/2.0).*(p .+ ϵ)
    S = floor.(Sᴿ)
    N₀ = sum(S)
    
    F = Array{ℜ}(undef,K)
    
    if N₀ < N
        F .= Sᴿ .- S
        Q = PriorityQueue{Int,ℜ}([k => ((1.0 - F[k])./(S[k] + 1.0)) for k=1:K])
        while N₀ < N
            k, Qₖ = peek(Q)
            S[k] += 1.0
            F[k] -= 1.0
            Q[k] = (1 - F[k])/(S[k] + 1.0)
            N₀ += 1.0
        end
    elseif N₀ > N
        F = Sᴿ .- S
        Q = PriorityQueue{Int,ℜ}([k => (F[k]/(S[k] + ϵ)) for k=1:K])
        while N₀ > N
            k, Qₖ = peek(Q)
            S[k] -= 1.0
            F[k] += 1.0
            Q[k] = F[k]/(S[k] + ϵ)
            N₀ -= 1.0
        end
    end
    return S
end

function save_json(filename; variables...)
    json_str = JSON.json(Dict(variables));
    open(filename, "w") do f
        write(f, json_str)
    end
    return json_str
end

function load_json(filename)
    json_str = nothing
    open(filename, "r") do f
        json_str = read(f, String)
    end
    return JSON.parse(json_str)
end

end

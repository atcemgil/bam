module BackwardKernel

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions
import DataStructures: PriorityQueue, peek, enqueue!, dequeue!
import Base: iterate, length, sum, eltype

export EventQueue, sum, length, eltype, iterate

mutable struct EventQueue{ℜ<:Real} 
    T::Int
    X::Array{ℜ,2}
    pq::PriorityQueue{Tuple{Int,Int},ℜ,Base.Order.ForwardOrdering}
    function EventQueue(X_new::Array{ℜ,2}) where {ℜ<:Real}
        I::Int = size(X_new,1)
        T_new::Int = Int(round(sum(X_new)))
        
        pq_new = PriorityQueue([(i₁,i₂) => rand(Dirichlet([1.0,X_new[i₁,i₂]]))[1] 
                                           for i₁=1:I, i₂=1:I if X_new[i₁,i₂]>0])
        return new{ℜ}(T_new,copy(X_new),pq_new)
    end
end

Base.sum(L::EventQueue) = sum(L.X)
Base.length(L::EventQueue) = L.T
Base.eltype(L::EventQueue) = Tuple{Int,Int}

function Base.iterate(L::EventQueue{ℜ}, state::Ƶ=1) where {Ƶ<:Int,ℜ<:Real}
    τ::Ƶ = state
    i₁_τ::Ƶ, i₂_τ::Ƶ = 0, 0
    t::ℜ, t_next::ℜ = 0.0, 0.0
    if L.T < τ
        return nothing
    end
    t = peek(L.pq)[2]
    i₁_τ, i₂_τ = dequeue!(L.pq)
    L.X[i₁_τ, i₂_τ] -= 1.0
    if L.X[i₁_τ, i₂_τ] > 0.0
        t_next = t + (1.0 - t)*rand(Dirichlet([1.0, L.X[i₁_τ,i₂_τ]]))[1]
        L.pq[(i₁_τ, i₂_τ)] = t_next
    end
    return (i₁_τ, i₂_τ), state+1
end

end
module BackwardKernel

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions
import DataStructures: PriorityQueue, peek, enqueue!, dequeue!
import Base: iterate, length, sum, eltype

export EventQueue, sum, length, eltype, iterate

mutable struct EventQueue{ℜ<:Real} 
    T::Int
    X::Array{ℜ}
    pq::PriorityQueue{Tuple{Int,Int,Int},ℜ,Base.Order.ForwardOrdering}
    function EventQueue(X_new::Array{ℜ}) where {ℜ<:Real}
        I::Int, J::Int, K::Int = size(X_new)
        T_new::Int = Int(round(sum(X_new)))
        pq_new = PriorityQueue([(i,j,k)=>rand(Dirichlet([1.0,X_new[i,j,k]]))[1] for k=1:K, j=1:J, i=1:I if X_new[i,j,k]>0])
        return new{ℜ}(T_new,copy(X_new),pq_new)
    end
end

Base.sum(L::EventQueue) = sum(L.X)
Base.length(L::EventQueue) = L.T
Base.eltype(L::EventQueue) = Tuple{Int,Int,Int}

function Base.iterate(L::EventQueue{ℜ}, state::Ƶ=1) where {Ƶ<:Int,ℜ<:Real}
    τ::Ƶ = state
    i_τ::Ƶ, j_τ::Ƶ, k_τ::Ƶ = 0, 0, 0
    t::ℜ, t_next::ℜ = 0.0, 0.0
    if L.T < τ
        return nothing
    end
    t = peek(L.pq)[2]
    i_τ, j_τ, k_τ = dequeue!(L.pq)
    L.X[i_τ, j_τ, k_τ] -= 1.0
    if L.X[i_τ, j_τ, k_τ] > 0.0
        t_next = t + (1.0 - t)*rand(Dirichlet([1.0, L.X[i_τ,j_τ, k_τ]]))[1]
        L.pq[(i_τ, j_τ, k_τ)] = t_next
    end
    return (i_τ, j_τ, k_τ), state+1
end

end
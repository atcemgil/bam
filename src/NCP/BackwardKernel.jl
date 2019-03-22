module BackwardKernel

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions
import DataStructures: PriorityQueue, peek, enqueue!, dequeue!
import Base: iterate, length, sum, eltype
import Base.Iterators: product

export EventQueue, sum, length, eltype, iterate

mutable struct EventQueue{ℜ<:Real, N}
    T::Int
    X::Array{ℜ,N}
    pq::PriorityQueue{NTuple{N,Int},Float64,Base.Order.ForwardOrdering}
    function EventQueue(X::Array{ℜ,N}) where {ℜ<:Real,N}
        I = size(X)
        T::Int = Int(round(sum(X)))
        pq = PriorityQueue([i => rand(Dirichlet([1.0,X[i...]]))[1] 
                           for i ∈ product(map(Iₙ -> 1:Iₙ, I)...) if X[i...]>0])
        return new{ℜ,N}(T,copy(X),pq)
    end
end

Base.sum(L::EventQueue) = sum(L.X)
Base.length(L::EventQueue) = L.T
Base.eltype(L::EventQueue{ℜ,N}) where {ℜ<:Real, N} = NTuple{N,Integer}
Base.size(L::EventQueue) = size(L.X)

function Base.iterate(L::EventQueue, state::Ƶ=1) where {Ƶ<:Integer}
    if L.T < state
        return nothing
    end
    
    t = peek(L.pq)[2]
    i = dequeue!(L.pq)
    
    L.X[i...] -= 1
    if L.X[i...] > 0.0
        L.pq[i] = t + (1.0 - t)*rand(Dirichlet([1.0, L.X[i...]]))[1]
    end

    return i, state+1
end

end
module NCP_EM

include("../Misc.jl")

using .Misc
using Distributions, SpecialFunctions
import Base.Iterators: product

export standard_EM 

function standard_EM(X::Array{ℜ,N}, R::Ƶ; a=1.0, b=a/nansum(X), EPOCHS::Ƶ=1, ϵ=1e-16) where {ℜ<:Real, Ƶ<:Integer, N}
    I = size(X)
    α_I = (a/R) ./ I
    α_R = a/R
    
    S_R, S₊ = zeros(ℜ,R), sum(X)
    S_I = [Array{ℜ}(undef,R,Iₙ) for Iₙ in I]
    
    log_λ = log(max(S₊ + a - 1, ϵ)) - log(b + 1)
    log_θ_R = log.(rand(Dirichlet(R,α_R + 0.1)))
    log_θ_I = [log.(rand(Dirichlet(Iₙ,α_Iₙ + 0.1),R)) for (Iₙ,α_Iₙ) in zip(I,α_I)]
    
    s, log_p = Array{ℜ}(undef,R), Array{ℜ}(undef,R)   

    for eph=1:EPOCHS
        # initialization
        for n ∈ 1:N
            S_I[n] .= 0
        end
        S_R .= 0
        
        # expectation step
        for i=product(map(Iₙ -> 1:Iₙ, I)...) 
            log_p .= log_θ_R + sum(n -> log_θ_I[n][i[n],:],1:N)
            log_p .-= logsumexp(log_p)
            s .= X[i...] .* exp.(log_p)
            
            for (n,iₙ) ∈ enumerate(i)
                S_I[n][:,iₙ] .+= s
            end
            S_R .+= s
        end
                    
        # maximization step
        for (n,Iₙ) ∈ enumerate(I)
            log_θ_I[n] .= log.(max.(S_I[n] .+ α_I[n] .- 1, ϵ))'
            log_θ_I[n] .-= logsumexp(log_θ_I[n],dims=1)
        end
        log_θ_R .= log.(max.(S_R .+ α_R .- 1, ϵ))
        log_θ_R .-= logsumexp(S_R)
    end
    return log_λ, (log_θ_R, log_θ_I...)
end

end
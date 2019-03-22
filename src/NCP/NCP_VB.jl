module NCP_VB

include("../Misc.jl")
using .Misc
import Base.Iterators: product

using Distributions, SpecialFunctions
import Base.Iterators: product

export standard_VB

function standard_VB(X::Array{ℜ,N}, R::Ƶ; a=1.0, b=a/nansum(X), EPOCHS::Ƶ=1, ϵ=1e-16) where {ℜ<:Real, Ƶ<:Integer, N}
    I = size(X)
    α_I = (a/R) ./ I
    α_R = a/R
    
    S_R, S₊ = zeros(R), sum(X)
    S_I = [zeros(R,Iₙ) for Iₙ in I]
    
    log_λ = digamma(S₊ + a) - log(b + 1)
    log_θ_R = log.(rand(Dirichlet(R,α_R + 0.1)))
    log_θ_I = [log.(rand(Dirichlet(Iₙ,α_Iₙ + 0.1),R)) for (Iₙ,α_Iₙ) in zip(I,α_I)]
    
    log_C = a*log(b) - (a + S₊)*log(b + 1) + lgamma(a + S₊) - lgamma(a) - sum(lgamma, X .+ 1)
                                                    
    s, log_p = zeros(R), zeros(R)
    ELBO = fill(log_C,EPOCHS)          

    for eph=1:EPOCHS
        # initialization
        for n ∈ 1:N
            S_I[n] .= 0
        end
        S_R .= 0
        
        # expectation of S
        for i ∈ product(map(Iₙ -> 1:Iₙ, I)...) 
            log_p .= log_θ_R + sum(n -> log_θ_I[n][i[n],:],1:N)
            log_p .-= logsumexp(log_p)
            s .= X[i...] .* exp.(log_p)
            
            for (n,iₙ) ∈ enumerate(i)
                S_I[n][:,iₙ] .+= s
            end
            S_R .+= s
            
            ELBO[eph] -= sum(s .* log_p)
        end
                                 
        for (n,Iₙ) ∈ enumerate(I)
            log_θ_I[n] .= digamma.(S_I[n] .+ α_I[n])' .- digamma.(S_R .+ α_R)'
            ELBO[eph] += sum(lbeta(α_I[n] .+ S_I[n];dims=2)) - R*lbeta(α_I[n],Iₙ)
        end
        log_θ_R .= digamma.(S_R .+ α_R) .- digamma(S₊ + R*α_R)
        ELBO[eph] += lbeta(α_R .+ S_R) - lbeta(α_R,R)
    end
    return ELBO, log_λ, (log_θ_R, log_θ_I...)
end

end
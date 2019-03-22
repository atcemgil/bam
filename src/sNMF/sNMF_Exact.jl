module sNMF_Exact

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export generate, log_marginal


function generate(I::Ƶ, R::Ƶ, T::Ƶ; γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ = I*I*γ
    α_R, α_IR = fill(a/R,R), fill(2.0*a/(I*R),I,R)
    
    θ_R = rand(Dirichlet(α_R))
    θ_IR = reshape([θ_ir for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r]))],I,R)
                            
    θ_I₁I₂ = Array{ℜ}(undef,I,I)
    @einsum θ_I₁I₂[i₁,i₂] = θ_R[r] * θ_IR[i₁,r] * θ_IR[i₂,r]
    θ_flat = reshape(θ_I₁I₂, I*I)

    X::Array{ℜ} = reshape(rand(Multinomial(T,θ_flat)),I,I)
                    
    return X, (θ_R, θ_IR)
end

function log_marginal(X::Array{ℜ,2}, R::Ƶ; μ::ℜ=1.0, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ = size(X,1)
    X_flat = reshape(Int.(X),I*I)
    
    a::ℜ, b::ℜ = I*I*γ, γ/μ
    α_R, α_IR = fill(a/R,R), fill(2.0*a/(I*R),I,R)
    
    s, S_R, S_IR, S⁺::ℜ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), sum(X)  
    log_PX, log_PS = -Inf, -Inf
    i₁::Ƶ, i₂::Ƶ = 0, 0
    
    log_C = a*log(b) - (a + S⁺)*log(b + 1.0)  
    log_C += sum(lgamma.(2.0 .* α_R)) - sum(lgamma.(α_IR)) - sum(lgamma.(α_R)) 
    
    if R == 1 
        S_R .= a + S⁺
        S_IR[:,1] = sum(X,dims=1)[1,:] .+ sum(X,dims=2)[:,1] .+ α_IR[:,1]
        log_PS = log_C - sum(lgamma.(X .+ 1.0))
        log_PS += sum(lgamma.(S_R)) + sum(lgamma.(S_IR)) - sum(lgamma.(2.0 .* S_R))
        return log_PS
    end
    
    for S_div = product(map(X_i₁i₂ -> combinations(1:(X_i₁i₂+R-1),R-1), X_flat)...)
        S_R .= α_R
        S_IR .= α_IR 
        log_PS = log_C
        for (i₁i₂, part) = enumerate(S_div)
            i₁, i₂ = mod(i₁i₂-1,I)+1, div(i₁i₂-1,I)+1
            s[1] = part[1]-1
            for r=2:R-1
                s[r] = part[r] - part[r-1] - 1
            end
            s[R] = X[i₁,i₂] + R - 1 - part[R-1]
            
            S_R .+= s
            S_IR[i₁,:] .+= s
            S_IR[i₂,:] .+= s
            log_PS -= sum(lgamma.(s .+ 1.0))
        end
        log_PS += sum(lgamma.(S_R)) + sum(lgamma.(S_IR)) - sum(lgamma.(2.0 .* S_R))
        log_PX = logsumexp([log_PX,log_PS])
    end
    
    return log_PX
end
end
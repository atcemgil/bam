module CP_Exact

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export generate, log_marginal

function generate(I::Ƶ, J::Ƶ, K::Ƶ, R::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    λ = rand(Gamma(a,1.0/b))
    θ_R = rand(Dirichlet(α_R))
    θ_IR = reshape([θ_ir for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r]))],I,R)
    θ_JR = reshape([θ_jr for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r]))],J,R)
    θ_KR = reshape([θ_kr for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r]))],K,R)
                            
    θ_IJK = Array{ℜ}(undef,I,J,K)
    @einsum θ_IJK[i,j,k] = θ_R[r]*θ_IR[i,r]*θ_JR[j,r]*θ_KR[k,r]
    Λ_IJK = λ.*θ_IJK
    
    X::Array{ℜ} = map.(λ_ijk -> rand(Poisson(λ_ijk)), Λ_IJK)
    return X, λ, (θ_R, θ_IR, θ_JR, θ_KR)
end

function generate(I::Ƶ, J::Ƶ, K::Ƶ, R::Ƶ, T::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    λ = rand(Gamma(a,1.0/b))
    θ_R = rand(Dirichlet(α_R))
    θ_IR = reshape([θ_ir for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r]))],I,R)
    θ_JR = reshape([θ_jr for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r]))],J,R)
    θ_KR = reshape([θ_kr for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r]))],K,R)
                            
    θ_IJK = Array{ℜ}(undef,I,J,K)
    @einsum θ_IJK[i,j,k] = θ_R[r]*θ_IR[i,r]*θ_JR[j,r]*θ_KR[k,r]
    θ_flat = reshape(θ_IJK, I*J*K)

    X::Array{ℜ} = reshape(rand(Multinomial(T,θ_flat)),I,J,K)
    return X, (θ_R, θ_IR, θ_JR, θ_KR)
end


function log_marginal(X::Array{ℜ,3}, R::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    X_flat = reshape(Int.(X),I*J*K)
    
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    log_PX, log_PS = -Inf, -Inf
    i::Ƶ, j::Ƶ , k::Ƶ = 0, 0, 0
    
    log_C = a*log(b) - (a + S⁺)*log(b + 1.0) #+ lgamma(a + S⁺) - lgamma(a)  
    log_C += 2.0*sum(lgamma.(α_R)) - sum(lgamma.(α_IR)) - sum(lgamma.(α_JR)) - sum(lgamma.(α_KR))
    
    if R == 1 
        S_R .= S⁺
        S_IR[:,1] .= sum(X,dims=(2,3))[:,1,1]
        S_JR[:,1] .= sum(X,dims=(1,3))[1,:,1]
        S_KR[:,1] .= sum(X,dims=(1,2))[1,1,:]
        log_PS = log_C
        log_PS += sum(lgamma.(α_IR .+ S_IR)) + sum(lgamma.(α_JR .+ S_JR)) + sum(lgamma.(α_KR .+ S_KR)) - 2.0*sum(lgamma.(α_R .+ S_R))
        log_PS -= sum(lgamma.(X .+ 1.0))
        return log_PS
    end
    
    for S_div = product(map(X_ijk -> combinations(1:(X_ijk+R-1),R-1), X_flat)...)
        S_R .= α_R
        S_IR .= α_IR
        S_JR .= α_JR
        S_KR .= α_KR
        log_PS = 0.0
        for (ijk, part) = enumerate(S_div)
            i, j, k = mod(ijk-1,I)+1, mod(div(ijk-1,I),J)+1, div(ijk-1,I*J)+1
            s[1] = part[1]-1
            for r=2:R-1
                s[r] = part[r] - part[r-1] - 1
            end
            s[R] = X[i,j,k] + R-1 - part[R-1]
            
            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            log_PS -= sum(lgamma.(s .+ 1.0))
        end
        log_PS += sum(lgamma.(S_IR)) + sum(lgamma.(S_JR)) + sum(lgamma.(S_KR)) - 2.0*sum(lgamma.(S_R)) + log_C
        log_PX = logsumexp([log_PX,log_PS])
    end
    
    return log_PX
end


end
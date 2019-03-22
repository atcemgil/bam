module CP_EM

include("../Misc.jl")

using .Misc
using Distributions, SpecialFunctions

export standard_EM, dual_EM 

function standard_EM(X::Array{ℜ,3}, R::Ƶ ; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = rand(Dirichlet(α_R .+ 1.0/R))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R)
    log_θ_JR = reshape([log(θ_jr) for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R)
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R)
    
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), 0.0
                   
    X_full = similar(X)

    for eph=1:EPOCHS
        S_R .= 0.0
        S_IR .= 0.0
        S_JR .= 0.0
        S_KR .= 0.0
        S⁺ = 0.0
                                        
        for k=1:K, j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i,:] .+ log_θ_JR[j,:] .+ log_θ_KR[k,:]
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
            
            X_full[i,j,k] = isnan(X[i,j,k]) ? exp(log_ρ_ijk) : X[i,j,k]
            s .= X_full[i,j,k] .* q

            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            S⁺ += X_full[i,j,k]
        end
                                        
        log_λ = log(max(S⁺+a-1.0, ϵ)) - log(b+1.0)
        log_θ_R .= log.(max.(S_R.+α_R.-1.0, ϵ))
        log_θ_IR .= log.(max.(S_IR.+α_IR.-1.0, ϵ))
        log_θ_JR .= log.(max.(S_JR.+α_JR.-1.0, ϵ))
        log_θ_KR .= log.(max.(S_KR.+α_KR.-1.0, ϵ))

        log_θ_R .-= logsumexp(log_θ_R)
        log_θ_IR .-= logsumexp(log_θ_IR;dims=1)
        log_θ_JR .-= logsumexp(log_θ_JR;dims=1)
        log_θ_KR .-= logsumexp(log_θ_KR;dims=1)
    end
    return X_full, log_λ, (log_θ_R, log_θ_IR, log_θ_JR, log_θ_KR)
end

function dual_EM(X::Array{ℜ,3}, R::Ƶ; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = rand(Dirichlet(α_R .+ 1.0/R))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R)
    log_θ_JR = reshape([log(θ_jr) for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R)
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R)
    
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), 0.0
                   
    X_full = similar(X)

    for eph=1:EPOCHS
        S_R .= 0.0
        S_IR .= 0.0
        S_JR .= 0.0
        S_KR .= 0.0
        S⁺ = 0.0
                                        
        for k=1:K, j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i,:] .+ log_θ_JR[j,:] .+ log_θ_KR[k,:]
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
                                            
            X_full[i,j,k] = isnan(X[i,j,k]) ? exp(log_ρ_ijk) : X[i,j,k]
            s .= finucan(X_full[i,j,k],q)

            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            S⁺ += X_full[i,j,k]
        end                             
        log_λ = digamma(S⁺+a) - log(b+1.0)
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR.+α_IR) .- digamma.(S_R.+α_R)'
        log_θ_JR .= digamma.(S_JR.+α_JR) .- digamma.(S_R.+α_R)'
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R.+α_R)'
    end
    return X_full, S⁺, (S_R, S_IR, S_JR, S_KR)
end

end
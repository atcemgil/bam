module Datasets

include("Misc.jl")
using .Misc

import NPZ, CSV
import DataFrames: dropmissing!, names!

export DataDesc, ngram, load

struct DataDesc
    path::String
    features::Array{Symbol}
    dims::Array{Int}
    types::Array{DataType}
end

descriptions = Dict(
    "Abalone"       => DataDesc("data/abalone/abalone.data",
                         [:sex, :length, :diameter, :height, :whole, :shucked, :viscera, :shell, :rings],
                         [3,10,10,10,10,10,10,10,10],
                         [String,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64]),

    "Contraception" => DataDesc("data/contraception/cmc.data",
                         [:age, :education, :heducation, :children, :religion, :working, :hoccupation, :living, :exposure, :method],
                         [4,4,4,4,2,2,4,4,2,3],
                         [Float64,Int,Int,Float64,Int,Int,Int,Int,Int,Int]),

    "Citation"      => DataDesc("data/citation/citation.data",
                         [:source, :target],
                         [2555,2555],
                         [Int,Int]),

    "LastFM"        => DataDesc("data/lastfm/user_artists.dat",
                         [:user, :artist],
                         [1892,17632],
                         [String,String])
    )

function load(name::String)
    desc = descriptions[name]
    df = CSV.read(desc.path; delim=',', datarow=1, types=desc.types)
    dropmissing!(df; disallowmissing=true)
    
    T,N = size(df)
    tab = Array{Int,2}(undef, T, N)
    
    for n in 1:N
        tab[:,n] .= categorize(df[n],desc.dims[n])
    end
    
    return cooccurrences(tab,desc.dims)
end

function load(name::String,features::Array{Symbol})
    desc = descriptions[name]
    df = CSV.read(desc.path; delim=',', datarow=1, types=desc.types)
    dropmissing!(df; disallowmissing=true)
    names!(df,desc.features)

    T,N = size(df)
    tab = Array{Int,2}(undef, T, length(features))
    dims = Array{Int,1}(undef, length(features))
    
    for n in 1:length(features)
        i = findfirst(desc.features .== features[n])
        tab[:,n] .= categorize(df[i],desc.dims[i])
        dims[n] = desc.dims[i]
    end
    
    return cooccurrences(tab,dims)
end

function load(name::String,features::Array{String})
    return load(name, Symbol.(features))
end

function load(name::String,features::AbstractArray{Int})
    return load(name, descriptions[name].features[features])
end

function ngram(n::Int, T::Int)
    freq = NPZ.npzread("data/letters/transitions_$(n)L.npy");
    θ = freq ./ sum(freq)
    return finucan(Float64(T),θ)
end

end
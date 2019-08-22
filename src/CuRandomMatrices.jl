module CuRandomMatrices
export curand

using LinearAlgebra
using CuArrays

include("../../randommatrices/src/RandomMatrices.jl")
using ..RandomMatrices

include("ginibre.jl")
# include("circular.jl")
# include("wigner.jl")
# include("wishart.jl")
end
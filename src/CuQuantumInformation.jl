module CuQuantumInformation
using QuantumInformation
eval(Expr(:export, names(QuantumInformation)...))
using MatrixEnsembles
using CUDA
using LinearAlgebra
using DocStringExtensions

const ⊗ = kron

export ⊗, curand

include("utils.jl")
include("randomqobjects.jl")
end # module
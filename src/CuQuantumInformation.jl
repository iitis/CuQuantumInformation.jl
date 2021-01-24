module CuQuantumInformation
using QuantumInformation
eval(Expr(:export, names(QuantumInformation)...))
using MatrixEnsembles
using CUDA
using LinearAlgebra
using DocStringExtensions

export curand

CUDA.allowscalar(false)
include("utils.jl")
include("randomqobjects.jl")
end # module
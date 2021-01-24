using CuQuantumInformation
using CUDA
using Random

using LinearAlgebra
using Test

my_tests = [
    "utils.jl",
    # "randomqobjects.jl",
    ]
for my_test in my_tests
    include(my_test)
end

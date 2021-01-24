```@setup CuQuantumInformation
using CuQuantumInformation
using LinearAlgebra
```

# Random quantum objects

All the distributions defined in `QuantumInformation` can be sampled from via the `curand` method
```@repl CuQuantumInformation
h = HaarKet{2}(3)

ψ = curand(h)

norm(ψ)
```
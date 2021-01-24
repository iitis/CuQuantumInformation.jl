# CuQuantumInformation.jl

CuQuantumInformation.jl is an extension of [`QuantumInformation`](https://github.com/iitis/QuantumInformation.jl) which provides specialized methods for `CuArrays`. This package is intended to provide a high-level interface for shifting the workload to a GPU.

This package provides definitions which play nice with `CuArrays` and their requirements like `allowscalar(false)`. After installing you can use this package as
```julia
using CuQuantumInformation
```
This will import and reexport all functions from `QuantumInformation`.

The main part of `QuantumInformation` that required specialized implementation for `CuArrays` are random quantum objects.
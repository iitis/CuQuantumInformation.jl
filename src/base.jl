export ket, bra, ketbra, proj, res, unres, max_mixed, max_entangled,
    werner_state, permutesystems
using CUDA

function ket(::Type{CuVector{T}}, val::Int, dim::Int) where {T<:Number}
    dim > 0 ? () : throw(ArgumentError("Vector dimension has to be nonnegative"))
    1 <= val <= dim ? () : throw(ArgumentError("Label have to be smaller than vector dimension"))
    ψ = CUDA.zeros(T, dim)
    @inline function kernel(ψ, val)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state == val
            @inbounds ψ[state] = one(T)
        end
        return
    end

    threads, blocks = cudiv(length(ψ))
    @cuda threads=threads blocks=blocks kernel(ψ, val)
    ψ
end

"""
$(SIGNATURES)
- `val`: non-zero entry - label.
- `dim`: length of the vector.

Return complex column vector \$|val\\rangle\$ of unit norm describing quantum state.
"""
cuket(val::Int, dim::Int) = ket(CuVector{ComplexF64}, val, dim)


"""
$(SIGNATURES)
- `val`: non-zero entry - label.
- `dim`: length of the vector

Return Hermitian conjugate \$\\langle val| = |val\\rangle^\\dagger\$ of the ket with the same label.
"""
cubra(val::Int, dim::Int) = cuket(val, dim)'

function ketbra(::Type{CuMatrix{T}}, valk::Int, valb::Int, idim::Int, odim::Int) where {T<:Number}
    idim > 0 && odim > 0 ? () : throw(ArgumentError("Matrix dimension has to be nonnegative"))
    1 <= valk <= idim && 1 <= valb <= odim ? () : throw(ArgumentError("Ket and bra labels have to be smaller than operator dimension"))
    
    ρ = CUDA.zeros(T, odim, idim)
    CI = CartesianIndices(ρ)
    @inline function kernel(ρ, valk, valb)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if CI[state] == CartesianIndex(valk, valb)
            @inbounds ρ[state] = one(T)
        end
        return
    end
    threads, blocks = cudiv(length(ρ))
    @cuda threads=threads blocks=blocks kernel(ρ, valk, valb)
    ρ
end

cuketbra(valk::Int, valb::Int, idim::Int, odim::Int) = ketbra(CuMatrix{ComplexF64}, valk, valb, idim, odim)

"""
$(SIGNATURES)
- `valk`: non-zero entry - label.
- `valb`: non-zero entry - label.
- `dim`: length of the ket and bra vectors

# Return outer product \$|valk\\rangle\\langle vakb|\$ of states \$|valk\\rangle\$ and \$|valb\\rangle\$.
"""


"""
- `valk`: non-zero entry - label.
- `valb`: non-zero entry - label.
- `idim`: length of the ket vector
- `odim`: length of the bra vector

# Return outer product \$|valk\\rangle\\langle vakb|\$ of states \$|valk\\rangle\$ and \$|valb\\rangle\$.
"""




"""
$(SIGNATURES)
- `ρ`: input matrix.

Returns `vec(ρ.T)`. Reshaping maps
    matrix `ρ` into a vector row by row.
""" 
res(ρ::AbstractMatrix{<:Number}) = @cast x[(j, i)] := ρ[i, j] ## TODO reshape + permutedims solves the problem

unres(ϕ::AbstractVector{<:Number}, cols::Int) = @cast x[i, j] := ϕ[(j, i)] j:cols ## TODO

"""
$(SIGNATURES)
- `ϕ`: input matrix.

Return de-reshaping of the vector into a matrix.
"""
function unres(ρ::AbstractVector{<:Number})  ## TODO
    dim = size(ρ, 1)
    s = isqrt(dim)
    unres(ρ, s)
end


"""
$(SIGNATURES)
- `d`: length of the vector.

Return maximally mixed state \$\\frac{1}{d}\\sum_{i=0}^{d-1}|i\\rangle\\langle i |\$ of length \$d\$.
"""
max_mixed(d::Int) = I(d)/d

"""
$(SIGNATURES)
- `d`: length of the vector.

Return maximally entangled state \$\\frac{1}{\\sqrt{d}}\\sum_{i=0}^{\\sqrt{d}-1}|ii\\rangle\$ of length \$\\sqrt{d}\$.
"""
function max_entangled(d::Int)
    sd = isqrt(d)
    ρ = res(Diagonal{ComplexF64}(I, sd))
    renormalize!(ρ)
    ρ
end

"""
$(SIGNATURES)
- `d`: length of the vector.
- `α`: real number from [0, 1].

Returns [Werner state](http://en.wikipedia.org/wiki/Werner_state) given by
\$\\frac{\\alpha}{d}\\left(\\sum_{i=0}^{\\sqrt{d}-1}|ii\\rangle\\right)
\\left(\\sum_{i=0}^{\\sqrt{d}-1}\\langle ii|\\right)+
\\frac{1-\\alpha}{d}\\sum_{i=0}^{d-1}|i\\rangle\\langle i|\$.
"""
function werner_state(d::Int, α::Float64)
    α > 1 || α < 0 ? throw(ArgumentError("α must be in [0, 1]")) : ()
    α * proj(max_entangled(d)) + (1 - α) * max_mixed(d)
end



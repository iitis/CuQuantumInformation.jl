renormalize!(x::CuVector) = (x = x ./ CuArrays.norm(x))
renormalize!(x::CuMatrix) = (x = x ./ CuArrays.tr(x))
# this works only for positive matrices!!
# invsqrt
function invsqrt!(x::CuMatrix)
    F = CuArrays.svd!(x)
    S = 1 ./ sqrt.(F.S)
    (CuArrays.transpose(S) .* F.U) * F.U'
end

function invsqrt(x::CuMatrix)
    y = copy(x)
    invsqrt!(y)
end

# kron for CuArrays
function kron(a::CuMatrix{T1}, b::CuMatrix{T2}) where {T1<:Number, T2<:Number}
    dims = map(prod, zip(size(a), size(b)))
    T = promote_type(T1, T2)
    c = CuMatrix{T}(undef, dims...)
    gpu_call(c, (c, a, b)) do state, c, a, b
        (i, j) = @cartesianidx c
        (p, q) = size(b)
        k = (i - 1) ÷ p + 1
        l = (j - 1) ÷ q + 1
        m = i - ((i - 1) ÷ p) * p
        n = j - ((j - 1) ÷ q) * q
        @inbounds c[i, j] = a[k, l] * b[m, n]
        return
    end
    return c
end

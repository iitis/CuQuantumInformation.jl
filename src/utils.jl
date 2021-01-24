export invsqrt, invsqrt!, cudiv

@inline function cudiv(x::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_x, ceil(Int, x/threads_x)
end

renormalize!(x::CuVector) = (x[:] = x ./ norm(x))
renormalize!(x::CuMatrix) = (x[:] = x ./ tr(x))

function invsqrt!(x::CuMatrix)
    F = CUDA.svd(x)
    S = 1 ./ sqrt.(F.S)
    x[:] = (transpose(S) .* F.U) * F.U'
end

function invsqrt(x::CuMatrix)
    y = copy(x)
    invsqrt!(y)
end

function LinearAlgebra.kron(A::CuArray{T1}, B::CuArray{T2}) where {T1<:Number, T2<:Number}
    T = promote_type(T1, T2)
    ret = CUDA.zeros(T, (size(A) .* size(B))...)
    CI = CartesianIndices(ret)
    @inline function kernel(ret, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(ret)
            @inbounds idx = CI[state].I .- 1
            idx_A = idx .÷ size(B) .+ 1
            idx_B = idx .% size(B) .+ 1
            @inbounds ret[state] = A[idx_A...] * B[idx_B...]
        end
        return
    end
    threads, blocks = cudiv(length(ret))
    @cuda threads=threads blocks=blocks kernel(ret, A, B)
    return ret
end

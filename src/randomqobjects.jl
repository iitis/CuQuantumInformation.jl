function curand(h::HaarKet{1})
    ψ = CUDA.randn(h.d)
    renormalize!(ψ)
 end

 function curand(h::HaarKet{2})
     ψ = CUDA.randn(h.d) + 1im * CUDA.randn(h.d)
     renormalize!(ψ)
  end

function curand(hs::HilbertSchmidtStates)
    ρ = MatrixEnsembles.curand(hs.w)
    renormalize!(ρ)
end

function curand(c::ChoiJamiolkowskiMatrices)
    z = MatrixEnsembles.curand(c.w)
    y = ptrace(z, [c.odim, c.idim], [1])
    sy = invsqrt(y)
    #TODO: fix this, can be accelerated with a custom kernel
    onesy = CuMatrix(I(c.odim)) ⊗ sy
    DynamicalMatrix(onesy * z * onesy, c.idim, c.odim)
end


function curand(c::HaarPOVM{N}) where N
    V = MatrixEnsembles.curand(c.c)
    POVMMeasurement([V'*(cu(ketbra(ComplexF32, i, i, c.odim)) ⊗ cu(𝕀(N)))*V for i=1:c.odim])
end

function curand(c::VonNeumannPOVM)
    V = MatrixEnsembles.curand(c.c)
    POVMMeasurement([proj(V[:, i]) for i=1:c.d])
end

function curand(c::WishartPOVM)
    Ws = map(x->MatrixEnsembles.curand(x), c.c)
    S = sum(Ws)
    Ssq = invsqrt(S)
    POVMMeasurement([Ssq * W * Ssq for W=Ws])
end
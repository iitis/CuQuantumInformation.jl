Random.seed!(42)
CUDA.allowscalar(false)
@testset "randomqobjects" begin
    @testset "HaarKet" begin
        d = 10
        h1 = HaarKet{1}(d)
        h2 = HaarKet{2}(d)

        ϕ = curand(h1)
        ψ = curand(h2)

        @test length(ϕ) == d
        @test length(ψ) == d
        @test typeof(ϕ) == CuVector{Float32}
        @test typeof(ψ) == CuVector{ComplexF32}
        @test sum(abs2.(ϕ)) ≈ 1. atol=1e-6
        @test sum(abs2.(ψ)) ≈ 1. atol=1e-6

        h2 = HaarKet(d)
        ϕ = curand(h2)

        @test length(ϕ) == d
        @test typeof(ϕ) == CuVector{ComplexF32}
        @test sum(abs2.(ϕ)) ≈ 1. atol=1e-6
    end

    @testset "HilbertSchmidtStates" begin
        d = 10
        hs = HilbertSchmidtStates{1, 0.1}(d)
        ρ = curand(hs)

        @test size(ρ) == (d, d)
        @test typeof(ρ) == CuMatrix{Float32}
        @test norm(ρ - ρ') ≈ 0. atol=1e-6 # is close to hermitian
        @test tr(ρ) ≈ 1. atol=1e-6

        hs = HilbertSchmidtStates{2, 0.1}(d)
        ρ = curand(hs)

        @test size(ρ) == (d, d)
        @test typeof(ρ) == CuMatrix{ComplexF32}
        @test norm(ρ - ρ') ≈ 0. atol=1e-6 # is close to hermitian
        @test tr(ρ) ≈ 1. atol=1e-6

    end

    @testset "ChoiJamiolkowskiMatrices" begin
        idim = 5
        odim = 6
        c = ChoiJamiolkowskiMatrices{1, 0.1}(idim, odim)
        j = curand(c)
        @test ptrace(j.matrix, [odim, idim], [1]) ≈ I atol=1e-5
        @test tr(j.matrix) ≈ idim atol=1e-5
        @test typeof(j.matrix) == CuMatrix{Float32}

        c = ChoiJamiolkowskiMatrices{2, 0.1}(idim, odim)
        j = curand(c)
        @test ptrace(j.matrix, [odim, idim], [1]) ≈ I atol=1e-5
        @test tr(j.matrix) ≈ idim atol=1e-5
        @test typeof(j.matrix) == CuMatrix{ComplexF32}
    end

    @testset "HaarPOVMs" begin
        idim = 2
        odim = 3
        c = HaarPOVM(idim, odim)

        p = curand(c)
        @test norm(sum(p.matrices) - I) ≈ 0  atol=1e-5
    end

    @testset "VonNeumannPOVMs" begin
        d = 3
        c = VonNeumannPOVM(d)

        p = curand(c)
        @test norm(sum(p.matrices) - I) ≈ 0  atol=1e-5
        @test length(p.matrices) == d
    end

    @testset "WishartPOVMs" begin
        idim = 2
        odim = 3
        c = WishartPOVM(idim, odim)

        p = curand(c)
        @test norm(sum(p.matrices) - I) ≈ 0  atol=1e-5
    end
end

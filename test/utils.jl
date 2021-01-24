CUDA.allowscalar(false)
@testset "Utility functions" begin

ρ = [0.25 0.25im; -0.25im 0.75]
ρ_d = cu(ρ)

@testset "renormalize" begin
    v = CUDA.randn(10)
    renormalize!(v)

    @test norm(v) ≈ 1 atol=1e-6

    A = CUDA.randn(10, 10)
    renormalize!(A)

    @test tr(A) ≈ 1 atol=1e-6
end

@testset "invsqrt" begin
    A = inv(sqrt(ρ))
    B_d = invsqrt(ρ_d)
    B = Array(B_d)
    @test norm(A-B) ≈ 0 atol=1e-6
end

@testset "kron" begin
   ρ2 = ρ ⊗ ρ
   ρ2_d = ρ_d ⊗ ρ_d 
   ρ2_h = Array(ρ2_d)
   @test norm(ρ2 - ρ2_h) ≈ 0 atol=1e-6
end
end

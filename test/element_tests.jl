# The reference element. Everything downstream reads its tables and
# nothing else, so a wrong coefficient here is a wrong scheme everywhere —
# and one that still runs, still conserves an energy, and still converges,
# only to the wrong order or the wrong answer. These are checked against
# the paper's printed matrices and against the properties the paper's
# stability argument rests on.

using LinearAlgebra: I, norm

# `x ↦ x^k` and its derivative at the element's nodes on [-1, 1].
monomial(el, k) = [x^k for x in el.nodes]
dmonomial(el, k) = [k == 0 ? zero(x) : k * x^(k - 1) for x in el.nodes]

stencil_matrix(el::Element{T,P1,PW}) where {T,P1,PW} =
    [el.stencil[i][j] for i in 1:P1, j in 1:PW]
truncate_matrix(el::Element{T,P1}) where {T,P1} =
    [el.truncate[i][j] for i in 1:P1, j in 1:P1]

@testset "Gauss–Lobatto nodes are symmetric and integrate to degree 2P-1" begin
    # Guards the node and weight construction: a node off by an ulp
    # breaks the symmetry the derivative's antisymmetry inherits, and a
    # wrong weight breaks summation by parts.
    for P in 1:8
        el = Element(Float64, P)
        @test el.nodes[1] == -1 && el.nodes[end] == 1
        @test all(i -> el.nodes[i] == -el.nodes[P + 2 - i], 1:(P + 1))
        @test issorted(el.nodes)
        @test el.offsets[1] == 0 && el.offsets[end] == P + 1
        for k in 0:(2P - 1)
            exact = iseven(k) ? 2 / (k + 1) : 0.0
            @test sum(el.weights .* monomial(el, k)) ≈ exact atol = 1e-14
        end
        # ... and not one degree further: GLL is not Gauss.
        @test !isapprox(sum(el.weights .* monomial(el, 2P)), 2 / (2P + 1); atol=1e-14)
    end
end

@testset "The wide derivative is the paper's (Figs. 5, 6, 7)" begin
    # Guards the operator against a sign or a factor in the face terms,
    # which would leave a consistent derivative that is no longer the
    # paper's — and no longer summation by parts. The paper prints
    # `hₑ D`; the element stores `h D = hₑ D / (P+1)`.
    W2 = stencil_matrix(Element(Float64, 2)) .* 3
    @test W2 ≈ [-3 0 4 -1 0; 0 -1 0 1 0; 0 1 -4 0 3] atol = 1e-14
    W3 = stencil_matrix(Element(Float64, 3)) .* 4
    @test round.(W3; digits=2) == [-6 0 8.09 -3.09 1 0
                                   0 -1.62 0 2.24 -0.62 0
                                   0 0.62 -2.24 0 1.62 0
                                   0 -1 3.09 -8.09 0 6]
    W4 = stencil_matrix(Element(Float64, 4)) .* 5
    @test round.(W4; digits=2) == [-10 0 13.51 -5.33 2.82 -1 0
                                   0 -2.48 0 3.49 -1.53 0.52 0
                                   0 0.75 -2.67 0 2.67 -0.75 0
                                   0 -0.52 1.53 -3.49 0 2.48 0
                                   0 1 -2.82 5.33 -13.51 0 10]
end

@testset "The wide derivative has the paper's structure for every P" begin
    # The paper's §II F: the neighbours reach only the face rows, the
    # diagonal vanishes, and D is odd under x → -x (eq. 58).
    for P in 1:8
        W = stencil_matrix(Element(Float64, P))
        P1 = P + 1
        @test all(iszero, W[2:end, 1]) && all(iszero, W[1:(end - 1), end])
        @test W[1, 1] != 0 && W[end, end] != 0
        @test all(i -> abs(W[i, i + 1]) < 1e-12, 1:P1)
        @test W ≈ -reverse(W) atol = 1e-12
    end
end

@testset "Across a continuous face the wide derivative is exact to degree P" begin
    # When the neighbours' face values equal the element's own, the face
    # terms vanish and what is left is the element's pseudospectral
    # derivative — exact for every polynomial the element can hold. On
    # [-1, 1] with `hₑ = 2`, `h = 2/(P+1)`, so the stored stencil divided
    # by `h` is the derivative per unit `ξ`.
    for P in 1:8
        el = Element(Float64, P)
        W = stencil_matrix(el)
        h = 2 / (P + 1)
        for k in 0:P
            u = monomial(el, k)
            window = [u[1]; u; u[end]]
            @test W * window ./ h ≈ dmonomial(el, k) atol = 1e-11 * max(1, P^2)
        end
    end
end

@testset "The wide derivative is summation by parts on a periodic line" begin
    # The property the paper's stability rests on (§II E, ξ = 1): with
    # the quadrature weights `w`, `w D + (w D)ᵀ = 0` when there is no
    # outer boundary. It holds only for the central fetch — a different
    # face average would break it — so this pins the face terms down from
    # a second direction.
    for P in 1:6, K in (1, 3)
        el = Element(Float64, P)
        Dm, _, w = global_operators(el, K, 0.1)
        WD = w .* Dm
        @test norm(WD + WD') <= 1e-12 * norm(WD)
    end
end

@testset "Truncation is the orthogonal projector off the top Legendre mode" begin
    # Guards the projector: it must keep every polynomial of degree < P,
    # remove L_P, be idempotent, and be self-adjoint in the quadrature
    # inner product — the last is what keeps the energy an invariant
    # when the right-hand side of `v` is projected.
    for P in 1:8
        el = Element(Float64, P)
        Π = truncate_matrix(el)
        w = collect(el.weights)
        for k in 0:(P - 1)
            @test Π * monomial(el, k) ≈ monomial(el, k) atol = 1e-12
        end
        LP = [TreeWaveOLDG.legendre_all(P, x)[P + 1] for x in el.nodes]
        @test norm(Π * LP) < 1e-12
        @test Π * Π ≈ Π atol = 1e-12
        @test w .* Π ≈ (w .* Π)' atol = 1e-12
    end
end

@testset "The tables are built once in BigFloat and rounded once" begin
    # Guards the precision rule: each type's coefficients are the
    # BigFloat ones rounded to that type, not the Float64 ones converted,
    # so a Float32 table is correctly rounded and a wider type gets every
    # digit it can hold.
    for P in (3, 5)
        big = Element(BigFloat, P)
        for T in (Float32, Float64)
            el = Element(T, P)
            @test el.nodes == T.(big.nodes)
            @test stencil_matrix(el) == T.(stencil_matrix(big))
            @test truncate_matrix(el) == T.(truncate_matrix(big))
        end
    end
    @test_throws ArgumentError Element(Float64, 0)
end

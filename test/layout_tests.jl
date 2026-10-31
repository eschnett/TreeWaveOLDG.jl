# The layout: where each owned point is, and how a sum over points
# becomes an integral. Both are TreeAMR's element centering's, reached
# through this package's names, so an error here would sample initial data
# and exact solutions at the wrong places — and the convergence tests
# would then measure the sampling, not the scheme.

@testset "Nodes sit where the element says, and face nodes coincide" begin
    # Guards the node positions against an off-by-one in the element or
    # block offset, and the two face nodes of a face against being given
    # positions an ulp apart, which would make "continuous across the
    # face" a statement about roundoff.
    for (D, P, E) in ((1, 3, 2), (2, 2, 2), (3, 1, 2))
        forest = wave_forest(Float64, Val(D), P, E; roots=2)
        el = Element(Float64, P)
        fs = FieldSet(forest, D; G=1, centering=elementcentered(D), P=P)
        fill_at_nodes!((x, v) -> x[v], fs, el)
        N = forest.N
        h = minimum_spacing(forest)
        for b in 1:nblocks(fs)
            origin = block_origin(forest, blockkey(fs, b))
            for I in CartesianIndices(ntuple(_ -> N, D))
                x = node_coordinates(fs, el, b, Tuple(I))
                @test all(d -> fs.work[(Tuple(I) .+ 1)..., d, b] == x[d], 1:D)
                for d in 1:D
                    q, i = divrem(I[d] - 1, P + 1)
                    ξ = el.nodes[i + 1]
                    @test x[d] ≈ origin[d] + (q + (1 + ξ) / 2) * (P + 1) * h
                end
            end
            # The high face node of each element and the low face node of
            # the next, inside the block, along dimension 1.
            for q in 1:(E - 1)
                I0 = ntuple(d -> d == 1 ? q * (P + 1) : 1, D)
                I1 = ntuple(d -> d == 1 ? q * (P + 1) + 1 : 1, D)
                @test node_coordinates(fs, el, b, I0) == node_coordinates(fs, el, b, I1)
            end
        end
    end
end

@testset "The quadrature norm integrates what the element can represent" begin
    # Guards the weights' scaling to an element `(P+1)h` wide and their
    # tensor product: the mean of `x^k` over the unit box is exact as
    # long as the element's quadrature is, and a constant has norm one.
    for (D, P) in ((1, 3), (2, 2), (2, 3))
        forest = wave_forest(Float64, Val(D), P, 2; roots=2)
        el = Element(Float64, P)
        fs = FieldSet(forest, 1; G=1, centering=elementcentered(D), P=P)
        u = statevector(fs)
        fill_at_nodes!((x, v) -> 1.0, fs, el)
        gather!(u, fs)
        @test quadrature_norm(fs, el, u) ≈ 1 atol = 1e-14
        @test quadrature_norm(fs, el, u; p=1) ≈ 1 atol = 1e-14
        k = P - 1                        # |x^k|² has degree 2P-2 ≤ 2P-1
        fill_at_nodes!((x, v) -> prod(x)^k, fs, el)
        gather!(u, fs)
        @test quadrature_norm(fs, el, u) ≈ sqrt(1 / (2k + 1))^D atol = 1e-13
        @test quadrature_norm(fs, el, u; p=Inf) ≈ 1
    end
end

@testset "A block that splits an element is refused, with the reason" begin
    # Guards against running on a forest whose blocks do not hold whole
    # elements: the wide derivative would read a face that is not one.
    el = Element(Float64, 2)
    @test_throws "not a whole number of elements" WaveProblem(
        Forest((2,); N=4, periodic=(true,), extents=((0.0, 1.0),)), el)
end

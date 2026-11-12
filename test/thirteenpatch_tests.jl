# The thirteen-patch ball: the seven-patch ball out to a middle sphere
# `|x| = Rm`, and six shells from there to `|x| = R1` whose radial
# coordinate is exactly radial, glued to the inner ones by the identity
# and to each other by the same quarter turns (`ThirteenPatchBall`). On
# the outer shells `√g` is a radial factor times an angular one, so the
# truncation there is a rank-one radial pair times an angular projection
# per plane, with the angular tables shared by every radial block and all
# six shells. The promises are the seven-patch ball's: the energy is
# invariant across every glue, it changes through a radiative sphere by
# the boundary term alone, and a plane wave converges at order `P`. Uses
# `ZeroData`, `ownedvars`, `energy_rate` and `radiative_rate` from
# `sevenpatch_tests.jl`, `scrambled`, `node_weights` and `state_of` from
# `refined_tests.jl`, and `metric_inner` and `statevar` from
# `annulus_tests.jl`.

using LinearAlgebra: det, cross, dot, norm

# The thirteen-patch ball's map without its factorization, so that the
# problem builds per-element tables on the outer shells too.
struct Unfactored{G} <: FullGeometry
    geometry::G
end
TreeWaveOLDG.geometry_frame(u::Unfactored, x, p) = geometry_frame(u.geometry, x, p)
TreeWaveOLDG.physical_coordinates(u::Unfactored, x, p) =
    physical_coordinates(u.geometry, x, p)
TreeWaveOLDG.check_geometry(u::Unfactored, forest) =
    TreeWaveOLDG.check_geometry(u.geometry, forest)
TreeWaveOLDG.affine_patch(u::Unfactored, p) = TreeWaveOLDG.affine_patch(u.geometry, p)
TreeWaveOLDG.outer_radius(u::Unfactored) = TreeWaveOLDG.outer_radius(u.geometry)

# Zero `u` and `v` in the outermost elements of the outer shells, of `m′`
# roots radially, so that nothing crosses the outer sphere under zero
# Dirichlet data.
function zero_outer_layer!(p::WaveProblem, u, m′)
    forest = p.fs.forest
    ua = statearray(u, p.fs)
    N = forest.N
    he = (TreeWaveOLDG.order(p.element) + 1) / N # an element's logical width
    for b in 1:nblocks(p.fs)
        q = patch(forest, blockkey(p.fs, b))
        q >= 8 || continue
        k, s = TreeWaveOLDG.SEVENPATCH_SHELLS[q - 7]
        for I in CartesianIndices((N, N, N))
            xk = coordinates(p.fs, b, Tuple(I))[k]
            (s == 1 ? xk >= m′ - he - 1e-12 : xk <= he + 1e-12) && (ua[I, :, b] .= 0)
        end
    end
    return u
end

@testset "The thirteen-patch ball's frame is its map's, and it refuses what it does not map" begin
    # `S[d] = √g ∇ξ^d` is the cofactor of the map's Jacobian, on all thirteen
    # patches. The inner shells run from the cube's faces to the middle
    # sphere, where the outer shells start at the same points, so the
    # identity glue joins them; the outer shells end on the outer sphere.
    # On an outer shell `|x|` is a function of `ξ_k` alone and the angles
    # of the other two alone, so `S[k]` is radial and the other two are
    # tangential; a map blended like the inner shells' fails that by `O(1)`.
    forest, geo = thirteenpatch_forest(Float64, 2, 2; roots=(2, 1, 2), Lc=1, Rm=2,
                                       R1=3.5)
    @test TreeWaveOLDG.TreeAMR.npatches(forest) == 13
    x = (0.3, 0.7, 0.9)
    ε = 1e-6
    for q in 1:13
        J, S = geometry_frame(geo, x, q)
        Jm = hcat([collect((physical_coordinates(geo, ntuple(j -> x[j] + (j == e) * ε, 3),
                                                 q) .-
                            physical_coordinates(geo, ntuple(j -> x[j] - (j == e) * ε, 3),
                                                 q)) ./ (2ε)) for e in 1:3]...)
        Sfd = abs(det(Jm)) .* inv(Jm)
        @test all(abs(S[d][i] - Sfd[d, i]) < 1e-8 * max(1, J) for d in 1:3, i in 1:3)
        @test J > 0
    end
    for (k, s) in TreeWaveOLDG.SEVENPATCH_SHELLS
        qi = TreeWaveOLDG.inner_shell(k, s)
        qo = TreeWaveOLDG.outer_shell(k, s)
        at(ξk) = ntuple(j -> j == k ? ξk : 0.6, 3)
        inner, middle = at(s == 1 ? 0.0 : 1.0), at(s == 1 ? 1.0 : 0.0)
        @test physical_coordinates(geo, inner, qi) ==
              ntuple(j -> j == k ? Float64(s) : -0.4, 3)
        @test sqrt(sum(abs2, physical_coordinates(geo, middle, qi))) ≈ 2 rtol = 1e-15
        start, outer = at(s == 1 ? 0.0 : 2.0), at(s == 1 ? 2.0 : 0.0)
        @test physical_coordinates(geo, start, qo) == physical_coordinates(geo, middle, qi)
        @test sqrt(sum(abs2, physical_coordinates(geo, outer, qo))) ≈ 3.5 rtol = 1e-15
        for ξ in ((0.3, 0.7, 0.9), (1.9, 0.1, 1.2))
            X = collect(physical_coordinates(geo, ξ, qo))
            ρ = s == 1 ? ξ[k] / 2 : 1 - ξ[k] / 2
            @test norm(X) ≈ 2 + 1.5ρ rtol = 1e-14
            c = [j == k ? s : 2 * ξ[j] / 2 - 1 for j in 1:3]
            @test X ≈ norm(X) .* c ./ norm(c) rtol = 1e-14
            J, S = geometry_frame(geo, ξ, qo)
            @test norm(cross(collect(S[k]), X)) < 1e-13 * norm(S[k]) * norm(X)
            @test all(abs(dot(collect(S[d]), X)) < 1e-13 * norm(S[d]) * norm(X)
                      for d in 1:3 if d != k)
        end
    end
    @test TreeWaveOLDG.outer_radius(geo) == 3.5
    el = Element(Float64, 2)
    wave = PlaneWave((1.0, 0.0, 0.0), 0.0)
    seven, geo7 = sevenpatch_forest(Float64, 2, 2; roots=(2, 2))
    @test_throws "thirteenpatch_forest" WaveProblem(seven, el; geometry=geo, boundary=wave)
    @test_throws "thirteenpatch_forest" WaveProblem(
        forest, el; boundary=wave, geometry=ThirteenPatchBall(1.0, 2.0, 3.5, (2, 1, 1)))
    @test_throws "sevenpatch_forest" WaveProblem(forest, el; geometry=geo7, boundary=wave)
    @test_throws "odd" thirteenpatch_forest(Float64, 2, 3)
    @test_throws "√3 Lc" thirteenpatch_forest(Float64, 2, 2; Lc=1, Rm=1.5, R1=3)
    @test_throws "must exceed the middle radius" thirteenpatch_forest(Float64, 2, 2;
                                                                      Rm=2, R1=2)
    @test_throws "(n, m, m′)" thirteenpatch_forest(Float64, 2, 2; roots=(2, 2))
    @test_throws "positive number of roots" thirteenpatch_forest(Float64, 2, 2;
                                                                 roots=(2, 1, 0))
    @test_throws "reaches the cube's faces" thirteenpatch_forest(
        Float64, 2, 2; roots=(4, 1, 1), refined=((0, 2), (0, 2), (0, 2)))
end

@testset "On the outer shells √g is the product of the factors the ball declares" begin
    # The factored truncation is `√g`-orthogonal only if the radial and the
    # angular factor multiply to the frame's `√g`, and its angular tables
    # are shared across the six shells only if the angular factor is the
    # same function on each. A factor missing `r′`, the angles' scale
    # `(2/n)²`, or a power of `|c|`, or a factor taken along the wrong
    # dimension, differs by `O(1)`. The cube and the inner shells do not
    # factorize.
    forest, geo = thirteenpatch_forest(Float64, 2, 2; roots=(3, 1, 2), Lc=1, Rm=2,
                                       R1=3.5)
    @test all(q -> TreeWaveOLDG.factored_volume(geo, q) == 0, 1:7)
    for ξ in ((0.3, 0.7, 0.9), (2.9, 0.1, 1.2), (1.5, 1.5, 1.5))
        for q in 8:13
            k = TreeWaveOLDG.factored_volume(geo, q)
            @test k == TreeWaveOLDG.SEVENPATCH_SHELLS[q - 7][1]
            @test TreeWaveOLDG.angular_class(geo, q) == 8
            ξ⊥ = Tuple(ξ[j] for j in 1:3 if j != k)
            J = first(geometry_frame(geo, ξ, q))
            @test TreeWaveOLDG.radial_volume(geo, ξ[k], q) *
                  TreeWaveOLDG.angular_volume(geo, ξ⊥, q) ≈ J rtol = 1e-14
        end
    end
end

@testset "On the outer shells the factored truncation is the per-element projection" begin
    # The radial pair times the angular projection is the `√g`-orthogonal
    # projection onto `Q_{P-1}`, so it equals the per-element one built
    # from the same map without its factorization, to rounding; an angular
    # table taken from another block's position, a radial pair from the
    # wrong element along `k`, or a plane read across the wrong dimension
    # does not. The forty-eight outer blocks share four angular tables. The
    # truncation is idempotent, self-adjoint in the `√g`-weighted inner
    # product, and leaves `u` alone.
    for P in (2, 3)
        el = Element(Float64, P)
        forest, geo = thirteenpatch_forest(Float64, P, 2; roots=(2, 1, 2), Lc=1, Rm=2,
                                           R1=3.5)
        p = WaveProblem(forest, el; geometry=geo, boundary=ZeroData())
        pe = WaveProblem(forest, el; geometry=Unfactored(geo), boundary=ZeroData())
        t, te = p.metric.truncation, pe.metric.truncation
        outer = [patch(forest, blockkey(p.fs, b)) >= 8 for b in 1:nblocks(p.fs)]
        @test count(outer) == 48
        @test all((t.factored.slot .!= 0) .== outer)
        @test all((t.slot .!= 0) .== (te.slot .!= 0) .& .!outer)
        @test size(t.factored.aweights, 3) == 4
        @test size(t.weights, 3) == 24
        n = length(statevector(p.fs))
        a, b = statevector(p.fs), statevector(p.fs)
        a .= scrambled(n, 0.1)
        b .= scrambled(n, 0.6)
        Πa, Πb = truncate_state!(copy(a), p), truncate_state!(copy(b), p)
        @test isapprox(Πa, truncate_state!(copy(a), pe); rtol=1e-14)
        va, vb = statevar(p, a, 2), statevar(p, b, 2)
        Πva, Πvb = statevar(p, Πa, 2), statevar(p, Πb, 2)
        scale = sqrt(metric_inner(p, va, va) * metric_inner(p, vb, vb))
        @test abs(metric_inner(p, Πva, vb) - metric_inner(p, va, Πvb)) < 1e-14 * scale
        @test truncate_state!(copy(Πa), p) ≈ Πa rtol = 1e-13
        @test statevar(p, Πa, 1) == statevar(p, a, 1)
        @test metric_inner(p, Πva, Πva) <= metric_inner(p, va, va)
    end
end

@testset "The factored tables built in Float64x2 are BigFloat's, bit for bit" begin
    # As for the shells' per-element tables: a radial pair or an angular
    # Cholesky factor whose build loses a digit to `Float64`, in a constant
    # of the map or in `projector_mode`'s solve, rounds some entry
    # differently.
    for P in (2, 3)
        el = Element(Float64, P)
        forest, geo = thirteenpatch_forest(Float64, P, 2; roots=(2, 1, 2), Lc=1, Rm=2,
                                           R1=3.5)
        fast = TreeWaveOLDG.element_truncation(geo, forest, el, TreeWaveOLDG.CPU())
        slow = TreeWaveOLDG.element_truncation(geo, forest, el, TreeWaveOLDG.CPU(),
                                               BigFloat)
        f, s = fast.factored, slow.factored
        @test f.slot == s.slot && f.aslot == s.aslot && f.radial == s.radial
        @test f.amodes == s.amodes
        @test f.rmodes == s.rmodes
        @test f.rduals == s.rduals
        @test f.aweights == s.aweights
        @test f.achol == s.achol
        @test fast.chol == slow.chol
    end
end

@testset "TreeAMR fills the thirteen-patch ball's sets across every glue" begin
    # Every patch face but the outer sphere's is glued, by the identity
    # between the layers and by quarter turns within each; a glue missing
    # from either layer leaves an outer face inside the ball.
    forest, geo = thirteenpatch_forest(Float64, 2, 2; roots=(2, 1, 1))
    p = WaveProblem(forest, Element(Float64, 2); geometry=geo,
                    boundary=RadiativeBoundary())
    @test occursin("turned across a glue", sprint(show, p.cartschedule))
    @test length(TreeWaveOLDG.thirteenpatch_glue()) == 2 * (6 + 6) + 2 * 6 * 4
    @test length(p.boundary.faces) == 6 * 2^2
    @test all(f -> patch(forest, blockkey(p.fs, Int(f[1]))) >= 8, p.boundary.faces)
end

@testset "The energy is invariant across the thirteen-patch ball's glues" begin
    # As on the seven-patch ball, now across the inner–outer identity glue
    # and the outer layer's quarter turns too, and with the factored
    # truncation on the outer shells: a truncation that is not
    # `√g`-orthogonal there leaves a rate of the size of its terms. The
    # state vanishes in the outer shells' outermost elements and the
    # Dirichlet data are zero, so nothing crosses the outer sphere.
    for P in (2, 3), truncate in (false, true)
        el = Element(Float64, P)
        forest, geo = thirteenpatch_forest(Float64, P, 2; roots=(2, 1, 2))
        p = WaveProblem(forest, el; geometry=geo, boundary=ZeroData(), truncate=truncate)
        u = statevector(p.fs)
        u .= scrambled(length(u), 0.2)
        zero_outer_layer!(p, u, 2)
        truncate_state!(u, p)
        rate, scale = energy_rate(p, u)
        @test scale > 1
        @test abs(rate) < 1e-13 * scale
    end
end

@testset "Through the thirteen-patch ball's radiative sphere the energy changes by its boundary term alone" begin
    # The outer sphere is the outer shells' face, where `S[k]` is radial;
    # the energy's rate is `-Σ w |S| (κ v² + u v/R)` there, to rounding.
    for P in (2, 3), (κ, ir) in ((1.0, true), (0.0, false)), truncate in (false, true)
        el = Element(Float64, P)
        forest, geo = thirteenpatch_forest(Float64, P, 2; roots=(2, 1, 1))
        p = WaveProblem(forest, el; geometry=geo, truncate=truncate,
                        boundary=RadiativeBoundary(; κ=κ, inverse_r=ir))
        @test p.boundary.invR == (ir ? 1 / 3 : 0)
        u = statevector(p.fs)
        u .= scrambled(length(u), 0.3)
        truncate_state!(u, p)
        rate, scale = energy_rate(p, u)
        @test scale > 1
        @test abs(rate - radiative_rate(p, u, κ)) < 1e-13 * scale
        κ > 0 && @test radiative_rate(p, u, κ) < -1e-3 * scale
    end
end

@testset "A plane wave converges at order P on the thirteen-patch ball" begin
    # The wave crosses every glue of both layers and is the Dirichlet data
    # on the outer sphere. The bounds are the measured rates (CODE.md, "The
    # thirteen-patch ball" under "Measured results") with room either side.
    for P in (2, 3)
        rs = [sevenpatch_errors(; P=P, E=2, roots=(2s, s, s), Rm=2, tend=1/4)
              for s in (1, 2)]
        rate = convergence_rate([r.h for r in rs], [r.l2 for r in rs])
        @test P - 0.3 < rate < P + 0.4
    end
end

@testset "The energy is invariant on the thirteen-patch ball with a refined cube" begin
    # Refinement inside the cube, whose map is affine, is the seven-patch
    # ball's; the outer layer changes nothing there, so the energy argument
    # holds with both coarse-fine faces and the factored truncation.
    P = 2
    el = Element(Float64, P)
    forest, geo = thirteenpatch_forest(Float64, P, 2; roots=(4, 1, 1), refined=true)
    @test maxlevel(forest) == 1
    @test all(k -> patch(forest, k) == 1 || level(k) == 0, forest.leaves)
    p = WaveProblem(forest, el; geometry=geo, boundary=ZeroData())
    u = statevector(p.fs)
    u .= scrambled(length(u), 0.7)
    zero_outer_layer!(p, u, 1)
    truncate_state!(u, p)
    rate, scale = energy_rate(p, u)
    @test scale > 1
    @test abs(rate) < 1e-13 * scale
end

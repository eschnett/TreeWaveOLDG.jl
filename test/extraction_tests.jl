# Waveforms (INSPIRAL-PLAN.md, M8e): the field's spherical-harmonic modes
# on spheres, from points mapped to the forest, read from the solution's
# element polynomials, and summed by a Gauss–Legendre times trapezoid
# rule. Each part is held to what it promises on its own: the rule to
# roundoff on the modes it integrates exactly, the ball's inverse map to
# roundoff against the forward map, and the reading to the interpolation
# order `P + 1` against the function it was filled from.

using LinearAlgebra: I, norm
using StaticArrays: SVector
using TreeWaveOLDG: sphere_rule, mode_index, at_physical, statevector, gather!

# A smooth function that is no polynomial of any degree, and its rate,
# for both variables of a state.
extraction_f(x) = sin(0.7x[1] + 0.3) * cos(0.5x[2]) * exp(0.2x[3]) + 0.3x[1] * x[3]
extraction_fill(x, var) = var == 1 ? extraction_f(x) : 2extraction_f(x)

function extraction_state(wave)
    fill_at_nodes!(at_physical(wave, extraction_fill), wave.fs, wave.element)
    u = statevector(wave.fs)
    gather!(u, wave.fs)
    return u
end

@testset "The spherical harmonics are the closed forms, and the rule's modes are exact" begin
    # A wrong recurrence coefficient or phase passes orthonormality on
    # some modes and not others; the closed forms of ℓ ≤ 2 pin the phase
    # and the normalization, the rule's orthonormality every mode up to
    # lmax, and a sum of `r^ℓ Y_ℓm` that its quadrature must recover.
    θ, φ = 0.7, 1.9
    Y = spherical_harmonics(3, θ, φ)
    @test Y[mode_index(0, 0)] ≈ 1 / sqrt(4π)
    @test Y[mode_index(1, 0)] ≈ sqrt(3 / 4π) * cos(θ)
    @test Y[mode_index(1, 1)] ≈ -sqrt(3 / 8π) * sin(θ) * cis(φ)
    @test Y[mode_index(2, 0)] ≈ sqrt(5 / 16π) * (3cos(θ)^2 - 1)
    @test Y[mode_index(2, 1)] ≈ -sqrt(15 / 8π) * sin(θ) * cos(θ) * cis(φ)
    @test Y[mode_index(2, 2)] ≈ sqrt(15 / 32π) * sin(θ)^2 * cis(2φ)
    @test harmonic_modes(2) == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    lmax, nθ = 8, 9
    table, dirs = sphere_rule(Float64, lmax, nθ)
    @test length(dirs) == 2nθ^2
    @test all(d -> norm(d) ≈ 1, dirs)
    # The rule integrates `Y_ℓm Ȳ_ℓ′m′` exactly for `ℓ + ℓ′ < 2nθ`.
    Ys = reduce(hcat, [spherical_harmonics(lmax, acos(d[3]), atan(d[2], d[1]))
                       for d in dirs])
    @test maximum(abs, table * transpose(Ys) - I) < 1e-13
    # A real field's modes: `Φ = Σ c_ℓm Y_ℓm + conj`, recovered.
    c = Dict((l, m) => complex(sin(3l + m), cos(l - 2m)) for (l, m) in harmonic_modes(lmax))
    field(d) = (y = spherical_harmonics(lmax, acos(d[3]), atan(d[2], d[1]));
                sum(2real(c[(l, m)] * y[mode_index(l, m)]) for (l, m) in harmonic_modes(lmax)))
    modes = table * field.(dirs)
    for (l, m) in harmonic_modes(lmax)
        # `2Re(c Y_ℓm)` is `c Y_ℓm + c̄ Ȳ_ℓm`, and `Ȳ_ℓm = (−1)^m Y_{ℓ,−m}`, so the
        # mode `(ℓ, m)` of it is `c` for `m > 0` and `c + c̄` for `m = 0`.
        want = m == 0 ? 2real(c[(l, m)]) : c[(l, m)]
        @test abs(modes[mode_index(l, m)] - want) < 1e-13
    end
end

@testset "The ball's inverse map finds every point, graded or not" begin
    # A wrong face, sign or radial inversion puts a point in the wrong
    # patch or at the wrong depth; the forward map of what is found must
    # give the point back, inside the patch's box, everywhere: in the
    # cube, in every shell, on the cube's faces, edges and corners, and on
    # the sphere.
    for grading in (0, 1.5)
        geo = SevenPatchBall(1.0, 3.0, (4, 3); grading=grading)
        boxes = [p == 1 ? (4, 4, 4) : ntuple(d -> d == (p - 2) ÷ 2 + 1 ? 3 : 4, 3)
                 for p in 1:7]
        pts = [SVector{3}(randn(3)) for _ in 1:4000]
        pts = [x / norm(x) * 3 * rand()^(1 / 3) for x in pts]
        append!(pts, [SVector(1.0, 0.3, -0.2), SVector(1.0, 1.0, 0.5),
                      SVector(-1.0, -1.0, -1.0), SVector(0.0, 0.0, 3.0),
                      SVector(3, 3, 3) / sqrt(3), SVector(0.0, -2.0, 0.0),
                      SVector(0.0, 0.0, 0.0)])
        worst, outside, patches = 0.0, 0, Set{Int}()
        for x in pts
            pp = ball_patch_point(geo, x)
            push!(patches, pp.p)
            y = SVector(physical_coordinates(geo, pp.ξ, pp.p))
            worst = max(worst, norm(y - x))
            outside += any(d -> !(0 <= pp.ξ[d] <= boxes[pp.p][d]), 1:3)
        end
        @test worst < 1e-13
        @test outside == 0
        @test patches == Set(1:7)
    end
    geo = SevenPatchBall(1.0, 3.0, (4, 3))
    @test ball_patch_point(geo, (0.5, -0.25, 1.0)).p == 1
    @test ball_patch_point(geo, (0.0, 0.0, 2.0)).p == 7
    @test ball_patch_point(geo, (-2.0, 0.0, 0.0)).p == 2
    @test_throws "beyond the ball's sphere" ball_patch_point(geo, (0.0, 3.1, 0.0))
end

@testset "On the refined ball the extraction converges at order P + 1" begin
    # The reading is the element's own polynomial at a located point: a
    # point in the wrong element, patch or level, or a wrong logical
    # position, reads another polynomial, and the error stops falling. On
    # spheres inside the cube's refined region, across its coarse-fine
    # faces, across the cube–shell faces and in the shells, the values
    # converge at order `P + 1` against the function at the points, and
    # each mode's error is bounded by `√(4π)` times theirs.
    P = 3
    radii = [0.3, 0.7, 1.3, 2.2]
    errs, moderrs = Float64[], Float64[]
    for s in (1, 2)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(4s, 2s), Lc=1.0, R1=3.0,
                                        refined=((s, 3s), (s, 3s), (s, 3s)))
        wave = WaveProblem(forest, Element(Float64, P); geometry=geo,
                           boundary=RadiativeBoundary())
        u = extraction_state(wave)
        ex = SphereExtraction(wave, radii; lmax=4)
        @test length(ex.points) == length(radii) * 2ex.nθ^2
        e1 = sphere_values(ex, wave, u) - [extraction_f(x) for x in ex.points]
        e2 = sphere_values(ex, wave, u; var=2) - [2extraction_f(x) for x in ex.points]
        push!(errs, max(maximum(abs, e1), maximum(abs, e2) / 2))
        m = sphere_modes(ex, wave, u) - sphere_modes(ex, [extraction_f(x) for x in ex.points])
        push!(moderrs, maximum(abs, m))
        @test moderrs[end] <= sqrt(4π) * errs[end]
        @test size(m) == (length(harmonic_modes(4)), length(radii))
    end
    @test log2(errs[1] / errs[2]) > P + 0.5
end

@testset "On a brick the extraction reads physical points, and converges" begin
    # Without a geometry the points are the forest's own positions; a
    # brick's sphere about its centre reads as the ball's does.
    P = 3
    errs = Float64[]
    for s in (1, 2)
        forest = wave_forest(Float64, Val(3), P, 2; roots=4s, L=4.0)
        wave = WaveProblem(forest, Element(Float64, P))
        u = extraction_state(wave)
        ex = SphereExtraction(wave, [0.5, 1.5]; lmax=3, centre=(2, 2, 2))
        e = sphere_values(ex, wave, u) - [extraction_f(x) for x in ex.points]
        push!(errs, maximum(abs, e))
    end
    @test log2(errs[1] / errs[2]) > P + 0.5
end

@testset "The extraction refuses what it cannot read" begin
    # A geometry without an inverse map would read points at the wrong
    # place; a radius beyond the sphere is no point of the ball.
    forest, geo = annulus_forest(Float64, 2, 2)
    wave = WaveProblem(forest, Element(Float64, 2); geometry=geo)
    @test_throws "brick or the seven-patch ball" SphereExtraction(wave, [0.5])
    forest, geo = sevenpatch_forest(Float64, 2, 2; roots=(2, 1), Lc=1.0, R1=3.0)
    wave = WaveProblem(forest, Element(Float64, 2); geometry=geo,
                       boundary=RadiativeBoundary())
    @test_throws "beyond the ball's sphere" SphereExtraction(wave, [3.5])
    @test_throws "must exceed lmax" SphereExtraction(wave, [1.0]; lmax=4, nθ=4)
end

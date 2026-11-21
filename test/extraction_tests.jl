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
        # The gather is TreeAMR's own element polynomial at the same points,
        # its kernel's nodes and weights summed in another order.
        xs = [ball_patch_point(geo, x) for x in ex.points]
        for var in 1:2
            ref = interpolate(wave.fs, u, xs, ElementPolynomial(); vars=var:var).values
            @test maximum(abs, vec(ref) - sphere_values(ex, wave, u; var)) <
                  1e-13 * maximum(abs, ref)
        end
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

# A quadratic, which the element polynomials hold exactly at `P ≥ 2`, and
# its gradient.
extraction_q(x) = 0.3x[1]^2 - 0.2x[1] * x[2] + 0.5x[3] + 0.1x[3]^2 + 1
extraction_∇q(x) = SVector(0.6x[1] - 0.2x[2], -0.2x[1], 0.5 + 0.2x[3])

function extraction_state(wave, f)
    fill_at_nodes!(at_physical(wave, (x, var) -> f(x)), wave.fs, wave.element)
    u = statevector(wave.fs)
    gather!(u, wave.fs)
    return u
end

@testset "The gradient is the element polynomial's, physical in the ball's cube" begin
    # Guards `sphere_gradients` (M6b's flux): the derivative weights must
    # be the reading's polynomial's, divided by the block's spacing and,
    # in the cube, by the map's scale; a weight taken along the wrong
    # dimension, a lost spacing or a logical gradient fails the
    # quadratic, which the elements hold exactly, and a point in the
    # wrong element stops the smooth function's error from falling at
    # order `P`.
    P = 3
    errs = Float64[]
    for s in (1, 2)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(4s, 2s), Lc=1.0, R1=3.0,
                                        refined=((s, 3s), (s, 3s), (s, 3s)))
        wave = WaveProblem(forest, Element(Float64, P); geometry=geo,
                           boundary=RadiativeBoundary())
        ex = SphereExtraction(wave, [0.3, 0.7, 0.95]; lmax=2, gradient=true)
        g = sphere_gradients(ex, wave, extraction_state(wave, extraction_q))
        @test maximum(j -> norm(g[j] - extraction_∇q(ex.points[j])), eachindex(g)) < 1e-12
        g = sphere_gradients(ex, wave, extraction_state(wave, extraction_f))
        ∇f(x) = TreeWaveOLDG.ForwardDiff.gradient(extraction_f, x)
        push!(errs, maximum(j -> norm(g[j] - ∇f(ex.points[j])), eachindex(g)))
    end
    @test log2(errs[1] / errs[2]) > P - 0.5
    # On a brick, the same quadratic.
    forest = wave_forest(Float64, Val(3), P, 2; roots=4, L=4.0)
    wave = WaveProblem(forest, Element(Float64, P))
    ex = SphereExtraction(wave, [0.5, 1.5]; lmax=2, centre=(2, 2, 2), gradient=true)
    g = sphere_gradients(ex, wave, extraction_state(wave, extraction_q))
    @test maximum(j -> norm(g[j] - extraction_∇q(ex.points[j])), eachindex(g)) < 1e-12
    @test_throws "gradient = true" sphere_gradients(SphereExtraction(wave, [0.5]; lmax=2,
                                                                     centre=(2, 2, 2)),
                                                    wave, statevector(wave.fs))
end

@testset "The flux through a sphere is -∮ Π ∂_rΦ" begin
    # Guards `sphere_flux`: for `Φ = g(r²)` and `Π = h(r²)` the flux through
    # the sphere of radius `R` is `-4πR² h(R²) 2R g′(R²)`, which the rule
    # integrates exactly from the exact values, and the grid's converges
    # to at order `P` or better; a wrong sign, a lost `R²`, a radial
    # derivative taken along the wrong direction or the rule's weights not
    # summing to `4π` breaks it.
    g(q) = exp(-4q) * cos(3q)
    dg(q) = -exp(-4q) * (4cos(3q) + 3sin(3q))
    h(q) = sin(2q) + 0.3
    state(x, var) = (q = x[1]^2 + x[2]^2 + x[3]^2; var == 1 ? g(q) : h(q))
    exact(R) = -4π * R^2 * h(R^2) * 2R * dg(R^2)
    P = 3
    radii = [0.4, 0.6, 0.9]
    errs = Float64[]
    for s in (1, 2)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(4s, 2s), Lc=1.0, R1=3.0,
                                        refined=((s, 3s), (s, 3s), (s, 3s)))
        wave = WaveProblem(forest, Element(Float64, P); geometry=geo,
                           boundary=RadiativeBoundary())
        fill_at_nodes!(at_physical(wave, state), wave.fs, wave.element)
        u = statevector(wave.fs)
        gather!(u, wave.fs)
        ex = SphereExtraction(wave, radii; lmax=2, gradient=true)
        Π = [state(x, 2) for x in ex.points]
        ∇Φ = [TreeWaveOLDG.ForwardDiff.gradient(y -> state(y, 1), x) for x in ex.points]
        @test sphere_flux(ex, Π, ∇Φ) ≈ exact.(radii) rtol = 1e-13
        push!(errs, maximum(abs, sphere_flux(ex, wave, u) - exact.(radii)))
    end
    @test log2(errs[1] / errs[2]) > P - 0.5
    @test errs[2] < 1e-3 * maximum(abs, exact.(radii))
end

@testset "The extraction refuses what it cannot read" begin
    # A geometry without an inverse map would read points at the wrong
    # place; a radius beyond the sphere is no point of the ball; a sphere
    # leaving a periodic brick would read the field of its images, which
    # TreeAMR's fold code does not mark, as if it were there.
    forest = wave_forest(Float64, Val(3), 2, 2; roots=4, L=4.0)
    wave = WaveProblem(forest, Element(Float64, 2))
    @test_throws "periodic image" SphereExtraction(wave, [3.0]; lmax=2, centre=(2, 2, 2))
    @test SphereExtraction(wave, [1.5]; lmax=2, centre=(2, 2, 2)) isa SphereExtraction
    forest, geo = annulus_forest(Float64, 2, 2)
    wave = WaveProblem(forest, Element(Float64, 2); geometry=geo)
    @test_throws "brick or the seven-patch ball" SphereExtraction(wave, [0.5])
    forest, geo = sevenpatch_forest(Float64, 2, 2; roots=(2, 1), Lc=1.0, R1=3.0)
    wave = WaveProblem(forest, Element(Float64, 2); geometry=geo,
                       boundary=RadiativeBoundary())
    @test_throws "beyond the ball's sphere" SphereExtraction(wave, [3.5])
    @test_throws "cube only" SphereExtraction(wave, [1.5]; gradient=true)
    @test_throws "must exceed lmax" SphereExtraction(wave, [1.0]; lmax=4, nθ=4)
end

@testset "The plane reads the field where the forest holds it, and converges" begin
    # Guards `PlaneExtraction` (`bin/orbit_movie.jl`'s frames): its grid
    # has to be the cell centres it says, its points beyond the ball's
    # sphere or outside a brick have to be left out as `NaN` and the rest
    # read, through the cube, its refined levels and the shells, as
    # TreeAMR's element polynomial does, converging at order `P + 1`.
    P = 3
    errs = Float64[]
    for s in (1, 2)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(4s, 2s), Lc=1.0, R1=3.0,
                                        refined=((s, 3s), (s, 3s), (s, 3s)))
        wave = WaveProblem(forest, Element(Float64, P); geometry=geo,
                           boundary=RadiativeBoundary())
        u = extraction_state(wave)
        n = 24
        ex = PlaneExtraction(wave, (0.1, -0.2, 0.3), 2.9, n)
        @test ex.points[1] ≈ SVector(0.1 - 2.9 * (n - 1) / n, -0.2 - 2.9 * (n - 1) / n, 0.3)
        @test ex.points[2][1] - ex.points[1][1] ≈ 2 * 2.9 / n
        @test ex.points[n + 1][2] - ex.points[1][2] ≈ 2 * 2.9 / n
        @test ex.inside == [norm(x) < 3.0 for x in ex.points]
        @test 0 < count(ex.inside) < n^2
        xs = [ball_patch_point(geo, x) for x in ex.points[ex.inside]]
        for var in 1:2
            vals = plane_values(ex, wave, u; var)
            @test size(vals) == (n, n)
            @test isnan.(vec(vals)) == .!ex.inside
            ref = interpolate(wave.fs, u, xs, ElementPolynomial(); vars=var:var).values
            @test maximum(abs, vec(ref) - vals[ex.inside]) < 1e-13 * maximum(abs, ref)
        end
        vals = plane_values(ex, wave, u)
        push!(errs, maximum(abs(vals[j] - extraction_f(ex.points[j]))
                            for j in eachindex(vals) if ex.inside[j]))
    end
    @test log2(errs[1] / errs[2]) > P + 0.5
    # On a brick the points outside it are left out.
    forest = wave_forest(Float64, Val(3), P, 2; roots=4, L=4.0)
    wave = WaveProblem(forest, Element(Float64, P))
    u = extraction_state(wave)
    ex = PlaneExtraction(wave, (2.0, 2.0, 2.0), 3.0, 12)
    @test ex.inside == [all(0 .<= x .<= 4) for x in ex.points]
    vals = plane_values(ex, wave, u)
    @test maximum(abs(vals[j] - extraction_f(ex.points[j]))
                  for j in eachindex(vals) if ex.inside[j]) < 1e-2
    forest, geo = annulus_forest(Float64, 2, 2)
    wave = WaveProblem(forest, Element(Float64, 2); geometry=geo)
    @test_throws "brick or the seven-patch ball" PlaneExtraction(wave, (0, 0, 0), 1.0, 4)
end

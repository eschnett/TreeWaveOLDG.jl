# The four-patch annulus: TreeAMR's ring of patches glued by the identity
# (its M17), mapped to `r0 ≤ r ≤ r1` in polar coordinates by
# `PolarAnnulus`, with solid walls at both radii. The scheme's promises
# are the flat box's, in the metric: the discrete energy
# `½ Σ w (√g v² + Σ_d F_d²/A_d)` is an exact invariant of the
# semi-discrete system, across the glued faces and the walls, with the
# truncation orthogonal in the `√g`-weighted inner product; and a smooth
# mode converges at order `P`. A glue that hands over the wrong face, a
# flux that is not `A_d D_d u`, or a truncation in the flat weights each
# break one of them. Uses `scrambled`, `owned`, `node_weights` and
# `state_of` from `refined_tests.jl`.

using StaticArrays: SVector

# `Σ w √g a b` over the owned nodes of `p`'s state set.
function metric_inner(p::WaveProblem{T,D}, a, b) where {T,D}
    w = node_weights(p.fs, p.element)
    J = selectdim(p.metric.coef, D + 1, 1)
    return sum(w .* J .* a .* b)
end

# Variable `var` of the state vector `u`, `(N…, block)`.
statevar(p::WaveProblem{T,D}, u, var) where {T,D} =
    copy(selectdim(statearray(u, p.fs), D + 1, var))

@testset "On the ring without a metric the right-hand side is the brick's, bit for bit" begin
    # Without a geometry the ring of four patches is, in logical
    # coordinates, the brick of four times the roots along the angle,
    # periodic there and walled along the radius. A glue that fills a ghost
    # from the wrong patch, the wrong face or the wrong plane, in `u`'s set
    # or in a gradient set with its parity, makes the two differ.
    for P in (2, 3), truncate in (false, true)
        el = Element(Float64, P)
        ring, _ = annulus_forest(Float64, P, 2; roots=(2, 2))
        brick = Forest{Float64}((8, 2); N=2(P + 1), periodic=(true, false),
                                reflecting=((false, false), (true, true)))
        pr = WaveProblem(ring, el; truncate=truncate)
        pb = WaveProblem(brick, el; truncate=truncate)
        # Block `b` of the ring is the brick's block at its origin moved by
        # its patch's place along the angle.
        borigin = Dict(o => b for (b, o) in enumerate(block_origins(brick, Float64)))
        patches = block_patches(ring)
        tobrick = [borigin[(o[1] + 2(patches[b] - 1), o[2])]
                   for (b, o) in enumerate(block_origins(ring, Float64))]
        ur = statevector(pr.fs)
        ur .= scrambled(length(ur), 0.3)
        ub = statevector(pb.fs)
        ra, ba = statearray(ur, pr.fs), statearray(ub, pb.fs)
        for b in eachindex(tobrick)
            selectdim(ba, 4, tobrick[b]) .= selectdim(ra, 4, b)
        end
        truncate_state!(ur, pr)
        truncate_state!(ub, pb)
        dur, dub = similar(ur), similar(ub)
        wave_rhs!(dur, ur, pr, 0.0)
        wave_rhs!(dub, ub, pb, 0.0)
        dra, dba = statearray(dur, pr.fs), statearray(dub, pb.fs)
        @test all(selectdim(dra, 4, b) == selectdim(dba, 4, tobrick[b])
                  for b in eachindex(tobrick))
        @test wave_energy(pr, ur) ≈ wave_energy(pb, ub) rtol = 1e-14
    end
end

@testset "The metric's coefficients at the nodes are the annulus's" begin
    # The kernels read `√g` and `A_d` from the coefficient set, and the
    # energy and the truncation evaluate the geometry itself; a coefficient
    # set filled at the wrong points or of the wrong patch makes them
    # disagree. And `√g` must be the product of its one-dimensional
    # factors, which the truncation is built from.
    el = Element(Float64, 3)
    forest, geo = annulus_forest(Float64, 3, 2; roots=(2, 1), r0=0.5, r1=2.0)
    p = WaveProblem(forest, el; geometry=geo)
    c = p.metric.coef
    worst = 0.0
    for b in 1:nblocks(p.fs), I in CartesianIndices((8, 8))
        x = node_coordinates(p.fs, el, b, Tuple(I))
        q = patch(forest, blockkey(p.fs, b))
        J, A = geometry_terms(geo, x, q)
        @test (c[I, 1, b], c[I, 2, b], c[I, 3, b]) == (J, A...)
        Jf = TreeWaveOLDG.jacobian_factor(geo, Val(1), x[1], q) *
             TreeWaveOLDG.jacobian_factor(geo, Val(2), x[2], q)
        worst = max(worst, abs(Jf / J - 1))
        # The physical position is on the annulus, at the patch's angle.
        X = physical_coordinates(geo, x, q)
        r, θ = hypot(X...), mod(atan(X[2], X[1]), 2π)
        @test r ≈ 0.5 + 1.5 * x[2] rtol = 1e-14
        x[1] < 2 && @test θ ≈ (q - 1) * π / 2 + π / 4 * x[1] atol = 1e-13
    end
    @test worst < 1e-15
    # The largest logical speed is `√g¹¹ = 1/(r θ′)` at the inner wall.
    @test p.metric.speed ≈ 1 / (0.5 * π / 4) rtol = 1e-14
    @test oldg_timestep(p; cfl=1) ≈ (1 / 8) / 4 / p.metric.speed rtol = 1e-14
end

@testset "On the annulus the truncation is the √g-orthogonal projection onto Q_{P-1}" begin
    # The energy is invariant under a truncation only if the projector is
    # self-adjoint in the inner product the energy is written in, which on
    # the annulus weights by `√g = r θ′ r′`. The element's projector is
    # self-adjoint in the flat weights only, so the problem builds one per
    # line from the radial factor of `√g`; a table built from the wrong
    # line, or in the flat weights, fails the adjointness.
    for P in (2, 3, 4)
        el = Element(Float64, P)
        forest, geo = annulus_forest(Float64, P, 2; roots=(2, 2), r0=0.25, r1=2.0)
        p = WaveProblem(forest, el; geometry=geo)
        flat = WaveProblem(forest, el)
        n = length(statevector(p.fs))
        a, b = statevector(p.fs), statevector(p.fs)
        a .= scrambled(n, 0.1)
        b .= scrambled(n, 0.6)
        Πa, Πb = truncate_state!(copy(a), p), truncate_state!(copy(b), p)
        va, vb = statevar(p, a, 2), statevar(p, b, 2)
        Πva, Πvb = statevar(p, Πa, 2), statevar(p, Πb, 2)
        scale = sqrt(metric_inner(p, va, va) * metric_inner(p, vb, vb))
        # Self-adjoint in `√g`, and so not in the flat weights.
        @test abs(metric_inner(p, Πva, vb) - metric_inner(p, va, Πvb)) < 1e-14 * scale
        # A projection, onto `Q_{P-1}`: the flat projector, which is the
        # identity on `Q_{P-1}` and on nothing else, leaves its result.
        @test truncate_state!(copy(Πa), p) ≈ Πa rtol = 1e-14
        @test truncate_state!(copy(Πa), flat) ≈ Πa rtol = 1e-13
        # `u` is left alone.
        @test statevar(p, Πa, 1) == statevar(p, a, 1)
        # The kinetic energy cannot grow under it.
        @test metric_inner(p, Πva, Πva) <= metric_inner(p, va, va)
        # Along the angle `√g` is constant, so there the tables are the
        # element's own.
        @test all(isapprox(collect(p.metric.truncation.modes[1][:, e, b]),
                           collect(el.topmode); rtol=1e-13, atol=1e-14)
                  for e in 1:2, b in 1:nblocks(p.fs))
    end
end

@testset "The energy is an exact invariant across the glued faces and the walls" begin
    # `dE/dt = Σ w √g v v̇ + Σ_d Σ w F_d (D_d v)` with `F_d = A_d D_d u`
    # vanishes for any state when the wide derivative is summation by parts
    # across every glued face and wall, and the right-hand side is
    # `v̇ = Σ_d D_d F_d/√g`. A ghost across the glue from the wrong patch, a
    # flux that is not `A_d D_d u`, or a divergence not divided by `√g`
    # leaves a rate of the size of its terms. Truncating `v` changes
    # nothing, since `Π_J` is self-adjoint in `√g`.
    for P in (2, 3), truncate in (false, true)
        el = Element(Float64, P)
        forest, geo = annulus_forest(Float64, P, 2; roots=(2, 1), r0=0.5, r1=1.5)
        p = WaveProblem(forest, el; geometry=geo, truncate=truncate)
        u = statevector(p.fs)
        u .= scrambled(length(u), 0.2)
        truncate_state!(u, p)
        du = similar(u)
        wave_rhs!(du, u, p, 0.0)
        F = map(owned, p.grads)
        v, vdot = statevar(p, u, 2), statevar(p, du, 2)
        TreeWaveOLDG.fill_gradient!(p, state_of(p, v))
        G = map(owned, p.grads)
        w = node_weights(p.fs, el)
        A = ntuple(d -> selectdim(p.metric.coef, 3, 1 + d), 2)
        kinetic = metric_inner(p, v, vdot)
        potential = sum(d -> sum(w .* F[d] .* G[d] ./ A[d]), 1:2)
        scale = metric_inner(p, abs.(v), abs.(vdot)) +
                sum(d -> sum(w .* abs.(F[d] .* G[d]) ./ A[d]), 1:2)
        @test scale > 1
        @test abs(kinetic + potential) < 1e-13 * scale
    end
end

@testset "Over a run on the annulus the energy changes by RK4's loss alone" begin
    # The semi-discrete energy is invariant, so what a run loses is the
    # integrator's, which never gains energy at a stable step and falls
    # at least as RK4's dissipation, `dt⁴` per unit time, when `dt` halves.
    k = annulus_wavenumber(2, 1, 1, 2)
    losses = map((1 // 2, 1 // 4)) do cfl
        el = Element(Float64, 3)
        forest, geo = annulus_forest(Float64, 3, 2; roots=(2, 1))
        p = WaveProblem(forest, el; geometry=geo)
        u0 = initial_state(p, annulus_exact(geo, 2, k, 0.0))
        tend = π / k
        dt = tend / ceil(tend / oldg_timestep(p; cfl=cfl))
        integ = solve!(wave_integrator(p, u0, 0.0, tend; dt=dt))
        E0, E1 = wave_energy(p, u0), wave_energy(p, integ.u)
        @test E1 <= E0
        (E0 - E1) / E0
    end
    @test losses[1] < 1e-8
    @test losses[1] / losses[2] > 16
end

@testset "The rotating Bessel mode converges at order P on the annulus" begin
    # The mode turns a quarter of the way round, so every point of it
    # crosses a glued face, and it has `∂_r u = 0` on both walls. A glue,
    # a metric or a wall that is wrong to `O(1)` breaks the rate; the
    # bounds are the measured rates (CODE.md, "The annulus" under
    # "Measured results") with room either side.
    for P in (2, 3)
        rs = [annulus_errors(; P=P, E=2, roots=(2s, s)) for s in (1, 2, 4)]
        rate = convergence_rate([r.h for r in rs], [r.l2 for r in rs])
        @test P - 0.3 < rate < P + 0.4
        @test all(r -> r.energy <= r.energy0, rs)
    end
end

@testset "The Bessel mode is the annulus's Neumann mode" begin
    # The exact solution must have `∂_r u = 0` on both walls, or the
    # convergence would measure the wall's error instead of the scheme's;
    # and `R(kr0) = 1` by its scaling.
    for (m, n) in ((0, 1), (1, 1), (2, 1), (3, 2))
        k = annulus_wavenumber(m, n, 0.5, 2.0)
        @test k > 0
        mode = annulus_exact(PolarAnnulus(0.5, 2.0, (2, 1)), m, k, 0.0).f
        dR(r) = mode.scale * (TreeWaveOLDG.dbesselj(m, k * r) * mode.yr0 -
                              TreeWaveOLDG.dbessely(m, k * r) * mode.jr0)
        @test abs(dR(0.5)) < 1e-12
        @test abs(dR(2.0)) < 1e-12
        @test mode((0.0, 0.0), 1, 1) ≈ 1 rtol = 1e-12
    end
    # The roots are ordered and distinct.
    @test annulus_wavenumber(1, 1, 1, 2) < annulus_wavenumber(1, 2, 1, 2)
end

@testset "A geometry refuses a forest it does not map" begin
    # The map is the patches' as `annulus_forest` builds them; anything
    # else would be evolved in a metric that is not its own.
    el = Element(Float64, 2)
    forest, geo = annulus_forest(Float64, 2, 2; roots=(2, 1))
    brick = wave_forest(Float64, Val(2), 2, 2; roots=2)
    @test_throws "annulus_forest" WaveProblem(brick, el; geometry=geo)
    @test_throws "annulus_forest" WaveProblem(forest, el;
                                              geometry=PolarAnnulus(1.0, 2.0, (2, 2)))
    @test_throws "odd" annulus_forest(Float64, 2, 3)
    @test_throws "0 < r0 < r1" annulus_forest(Float64, 2, 2; r0=2, r1=1)
    refined, geo2 = annulus_forest(Float64, 2, 2; roots=(2, 1))
    refine!(refined, refined.leaves[1:1])
    balance!(refined)
    # The annulus's map is nowhere affine, so it takes no refinement.
    @test_throws "whose map is affine" WaveProblem(refined, el; geometry=geo2)
    # The inspiral needs a brick in flat space.
    glue = reduce(vcat, [[(from=(q, 1, 2), to=(mod1(q + 1, 4), 1, 1), map=(1, 2, 3)),
                          (from=(mod1(q + 1, 4), 1, 1), to=(q, 1, 2), map=(1, 2, 3))]
                         for q in 1:4])
    ring3 = Forest{Float64}([(roots=(2, 2, 2),) for _ in 1:4]; N=6, glue=glue,
                            reflecting=((false, false), (true, true), (true, true)))
    wave = WaveProblem(ring3, el)
    c = SVector(1.0, 1.0, 1.0)
    @test_throws "a forest of patches" InspiralProblem(
        wave, (ParticleModel(; Q=0.1, ℓ=0.01, M=1.0),),
        Window(c, 0.5, 0.75; continuity=3), uniform_motion((c,), (SVector(0.0, 0.0, 0.0),)))
end

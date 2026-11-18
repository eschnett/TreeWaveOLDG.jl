# Prescribed paths (INSPIRAL-PLAN.md, M4): hyperbolic motion and the slow
# circle, whose accelerations and their derivatives are known exactly,
# and the retarded field of a point charge that the hyperbolic run starts
# from and is compared with. A path whose derivatives were not those of
# its positions would give the source a wrong `ä` that no test of the
# field would see; a wrong retarded time, a wrong `γ` or a wrong sign in
# the field would make every comparison with it meaningless.

using IMEXRungeKutta: solve!
using TreeWaveOLDG: truncate_state!
using LinearAlgebra: dot, norm
using StaticArrays

const TWP = TreeWaveOLDG

# d/dt of f at t, by nested dual numbers.
ddt(f, t) = TWP.ForwardDiff.derivative(f, t)

@testset "The paths' derivatives are those of their positions" begin
    # Guards `CirclePath` and `HyperbolicPath`: `v`, `a`, `ȧ`, `ä`, `a⃛`
    # and `a⃜` must be the derivatives of `z`, and the motions those asked
    # for: a circle of the given radius and speed with `ȧ = -Ω²v`, and
    # hyperbolic motion at rest at `t = 0` with `a = g`, `ä = -3g³`.
    for motion in (circular_motion((1.0, 2.0, 3.0), 0.5, 0.05; phase=0.3),
                   circular_motion((1.0, 2.0, 3.0), 0.2, 0.3; normal=(1, 1, 0)),
                   hyperbolic_motion((1.0, 2.0, 3.0), 0.7; direction=(1, 2, 2)))
        path = motion.path
        for t in (0.0, 0.4, 1.3)
            tr = path(t)[1]
            @test ddt(s -> path(s)[1].z, t) ≈ tr.v atol = 1e-15
            @test ddt(s -> path(s)[1].v, t) ≈ tr.a atol = 1e-15
            @test ddt(s -> path(s)[1].a, t) ≈ tr.ȧ atol = 1e-15
            @test ddt(s -> path(s)[1].ȧ, t) ≈ tr.ä atol = 1e-14
            @test ddt(s -> path(s)[1].ä, t) ≈ tr.a⃛ atol = 1e-14
            @test ddt(s -> path(s)[1].a⃛, t) ≈ tr.a⃜ atol = 1e-14
        end
    end
    circle = circular_motion((1.0, 2.0, 3.0), 0.5, 0.05).path
    for t in (0.0, 2.0)
        tr = circle(t)[1]
        @test norm(tr.z - SA[1.0, 2.0, 3.0]) ≈ 0.5
        @test norm(tr.v) ≈ 0.05
        @test tr.ȧ ≈ -(0.05 / 0.5)^2 * tr.v
        @test tr.z[3] == 3.0
    end
    @test circle(0.0)[1].z ≈ SA[1.5, 2.0, 3.0]
    @test circle(0.0)[1].v[2] > 0
    g = 0.7
    tr = hyperbolic_motion((1.0, 2.0, 3.0), g).path(0.0)[1]
    @test tr.z == SA[1.0, 2.0, 3.0] && tr.v == zero(tr.v)
    @test tr.a ≈ SA[0, 0, g] && tr.ȧ == zero(tr.ȧ) && tr.ä ≈ SA[0, 0, -3g^3]
    # Proper acceleration g: γ³|a| = g along the motion.
    tr = hyperbolic_motion((1.0, 2.0, 3.0), g).path(0.8)[1]
    @test norm(tr.a) / (1 - dot(tr.v, tr.v))^1.5 ≈ g
end

@testset "Along the exact circle, the source is the residual of the windowed particle field" begin
    # Guards `a⃜` in `Trajectory` and `trajectory_at`: the order-3 term
    # reads `ä` through the rest-frame `C`, so the source's second time
    # derivative needs the path's sixth derivative. Without it the source
    # was the residual along the path's Taylor polynomial, off the true
    # circle's by 3% of the order-3 correction at `v = 0.2`, and the
    # circle's radial force converged to -0.024 F_RR rather than zero
    # (CODE.md, "Prescribed paths" under "Measured results").
    ddt2(f, t) = ddt(s -> ddt(f, s), t)
    ∇²(f, x) = sum(ddt2(u -> f(x + u * e), zero(eltype(x)))
                   for e in (SA[1, 0, 0], SA[0, 1, 0], SA[0, 0, 1]))
    setprecision(BigFloat, 256) do
        motion = circular_motion(SA[big(0), big(0), big(0)], big"0.5", big"0.2")
        m = ParticleModel{BigFloat}(sqrt(big"0.005"), big"1e-3", big"1.0")
        w = Window(SA[big(0), big(0), big(0)], big"1.25", big"2.5"; continuity=4)
        tr₀ = motion.path(big(0))
        WΨ(ξ, t) = w(ξ) * particle_field(m, ξ, motion.path(t)[1]; order=3)
        for x in (SA[big"0.6", big"0.1", big"0.05"], SA[big"0.0", big"0.5", big"0.3"],
                  SA[big"1.0", big"0.6", big"0.3"], SA[big"1.6", big"0.8", big"0.4"])
            exact = ddt2(t -> WΨ(x, t), big(0)) - ∇²(ξ -> WΨ(ξ, big(0)), x) +
                    TWP.particle_density(m, x, tr₀[1].z, tr₀[1].v)
            S = particle_source((m,), w, x, tr₀; order=3)
            @test S ≈ exact rtol = 1e-40
        end
    end
end

@testset "The retarded field is the field of a point charge" begin
    # Guards `retarded_field`: for a particle at rest or in uniform
    # motion it must be the boosted Coulomb field `-Q/4π|Λ(v)(x - z(t))|`,
    # at `t = 0` of hyperbolic motion the time-symmetric
    # `-Q/(4πr√(1 + gr cos θ + g²r²/4))` with `∂ₜΦ = 0`; at any time it
    # must solve the wave equation away from the particle, and its second
    # value must be its time derivative.
    Q = 0.07
    for v in (SA[0.0, 0.0, 0.0], SA[0.2, -0.1, 0.3])
        path = uniform_motion((SA[0.1, 0.2, 0.3],), (v,)).path
        for x in (SA[0.5, 0.2, -0.1], SA[-1.0, 2.0, 0.7]), t in (0.0, 0.6)
            tr = path(t)[1]
            y = TWP.boost(v, x - tr.z)
            Φ, Π = retarded_field(path, 1, Q, x, t)
            @test Φ ≈ -Q / (4π * norm(y)) rtol = 1e-13
            @test Π ≈ ddt(s -> -Q / (4π * norm(TWP.boost(v, x - path(s)[1].z))), t) rtol = 1e-11
        end
    end
    g = 0.3
    path = hyperbolic_motion(SA[0.0, 0.0, 0.0], g; direction=(0, 1, 0)).path
    for (r, θ) in ((0.05, 0.3), (1.0, 1.2), (3.0, 2.5), (0.5, π / 2))
        x = r * SA[sin(θ), cos(θ), 0.0]
        Φ, Π = retarded_field(path, 1, Q, x, 0.0)
        @test Φ ≈ -Q / (4π * r * sqrt(1 + g * r * cos(θ) + g^2 * r^2 / 4)) rtol = 1e-14
        # `∂ₜΦ` vanishes here, so `Π` is roundoff, mostly of the retarded
        # time, which lies about `r` back: it grows with `r` (1.2e-15 of
        # `Φ` at `r = 3`, under 3e-17 at `r ≤ 1`).
        @test abs(Π) < 1e-15 * max(r, 1) * abs(Φ)
    end
    # Behind the particle by more than 1/g no past light cone reaches.
    @test_throws ArgumentError retarded_field(path, 1, Q, SA[0.0, -4.0, 0.0], 0.0)
    for path in (path, circular_motion(SA[0.0, 0.0, 0.0], 0.5, 0.3).path)
        x, t, h = SA[0.3, 0.7, -0.2], 0.4, 1e-3
        Φ(x, t) = retarded_field(path, 1, Q, x, t)[1]
        e = (SA[h, 0, 0], SA[0, h, 0], SA[0, 0, h])
        ∇² = sum((Φ(x + d, t) - 2Φ(x, t) + Φ(x - d, t)) / h^2 for d in e)
        ∂ₜ² = (Φ(x, t + h) - 2Φ(x, t) + Φ(x, t - h)) / h^2
        @test abs(∂ₜ² - ∇²) < 1e-4 * abs(∇²)
        @test retarded_field(path, 1, Q, x, t)[2] ≈ (Φ(x, t + h) - Φ(x, t - h)) / 2h rtol = 1e-6
    end
end

@testset "The smoothstep rises from 0 to 1 with its first n derivatives zero at both ends" begin
    # Guards `smoothstep`, of which M8f's ramps are made: a wrong
    # coefficient leaves a derivative at a joint that should vanish, and
    # the joint then sends out what the run measures as the joint's cost;
    # a wrong mirror image above `x = 1/2` or a wrong integral puts the
    # spun-up particle somewhere else than its circle.
    @test smoothstep(1, 0.3) ≈ 3 * 0.3^2 - 2 * 0.3^3
    @test smoothstep(2, 0.3) ≈ 6 * 0.3^5 - 15 * 0.3^4 + 10 * 0.3^3
    @test smoothstep(0, 0.3) == 0.3 && smoothstep(0, 0.3, 1) == 1
    xs, ws = gauss_nodes_weights(8)
    setprecision(BigFloat, 256) do
        for n in 0:7
            S(x, d=0) = smoothstep(n, x, d)
            @test S(big(0)) == 0 && S(big(1)) == 1 && S(big"0.5") == 1 // 2
            @test S(big(-1)) == 0 && S(big(2)) == 1 && S(big(2), 1) == 0
            @test S(big(1), -1) == 1 // 2 && S(big(3), -1) == 5 // 2
            # The `n + 1`-th derivative does not vanish: `(n + 1)! C(2n + 1, n)`
            # at 0. The first `n` do: at 0 exactly, at 1 from the left as the
            # `(n + 1 - d)`-th power of the distance, times at most that.
            Sn1 = factorial(n + 1) * binomial(2n + 1, n)
            @test S(big(0), n + 1) == Sn1
            for d in 1:n
                @test S(big(0), d) == 0
                @test abs(S(1 - big"1e-20", d)) < Sn1 * big"1e-20"^(n + 1 - d)
            end
            # The derivatives and the integral against nested duals and
            # an 8-point Gauss rule, exact for degree 15, both sides of 1/2.
            for x in (big"0.2", big"0.5", big"0.7", big"0.95")
                f = S
                for d in 1:3
                    g = f
                    f = y -> ddt(g, y)
                    @test S(x, d) ≈ f(x) rtol = 1e-60 atol = 1e-60
                end
                I = x / 2 * sum(w * S(x * (ξ + 1) / 2) for (ξ, w) in zip(xs, ws))
                @test S(x, -1) ≈ I rtol = 1e-60
            end
        end
    end
    @test_throws ArgumentError smoothstep(8, 0.5)
    @test_throws ArgumentError smoothstep(-1, 0.5)
end

@testset "The ramped circle's jets are its positions' derivatives, and its joints as smooth as asked" begin
    # Guards `ramped_circular_motion` (INSPIRAL-PLAN.md's M8f): its `v`
    # through `a⃜` must be the derivatives of its `z` on every piece, or the
    # source is that of another path; at each joint `a` and its first
    # `n - 1` derivatives must be continuous and the `n`-th jump by what
    # the step's `n + 1`-th derivative says, or the runs measure another
    # smoothness than the one they name. Without a past and a departure it
    # must be the circle bit for bit, so that M8d's runs stay as they were.
    setprecision(BigFloat, 256) do
        rc, v, T₀, Δ, ε, Δ₂ = big"0.5", big"0.2", big(3), big(2), big"1e-3", big(5)
        Ω = v / rc
        for n in (0, 2, 4)
            path = ramped_circular_motion(SA[big(1), big(2), big(3)], rc, v; phase=big"0.3",
                                          past=T₀, ramp=Δ, n, depart=ε, onset=Δ₂,
                                          n2=n).path
            jet(tr) = (tr.z, tr.v, tr.a, tr.ȧ, tr.ä, tr.a⃛, tr.a⃜)
            for t in big.((-6, -4.5, -3.5, -1, 0.5, 3, 7))
                tr = path(t)[1]
                J = jet(tr)
                for k in 1:6
                    @test norm(ddt(s -> jet(path(s)[1])[k], t) - J[k + 1]) < big"1e-60"
                end
                @test norm(tr.v) <= v
            end
            # The joints: at rest, onto the circle, the departure, its end;
            # the jump in `a⁽ⁿ⁾ = z⁽ⁿ⁺²⁾` that the step's derivative gives.
            Sn1 = factorial(n + 1) * binomial(2n + 1, n)
            for (tj, jump) in ((-T₀ - Δ, rc * Ω / Δ^(n + 1) * Sn1),
                               (-T₀, rc * Ω / Δ^(n + 1) * Sn1),
                               (big(0), ε / Δ₂^(n + 1) * Sn1),
                               (Δ₂, ε / Δ₂^(n + 1) * Sn1))
                l, r = jet(path(tj - big"1e-40")[1]), jet(path(tj)[1])
                for k in 1:(n + 2)
                    @test norm(l[k] - r[k]) < big"1e-35"
                end
                @test norm(l[n + 3] - r[n + 3]) ≈ jump rtol = 1e-30
            end
            # At rest before the spin-up, on the circle where it ends.
            @test norm(path(big(-100))[1].v) == 0
            @test norm(path(-T₀ - Δ)[1].z - SA[big(1), big(2), big(3)]) ≈ rc
        end
    end
    circle = circular_motion((1.0, 2.0, 3.0), 0.5, 0.2; phase=0.3).path
    for path in (ramped_circular_motion((1.0, 2.0, 3.0), 0.5, 0.2; phase=0.3).path,
                 ramped_circular_motion((1.0, 2.0, 3.0), 0.5, 0.2; phase=0.3,
                                        past=20.0).path)
        @test all(t -> path(t)[1] === circle(t)[1], (-10.0, 0.0, 0.3, 7.0))
    end
    @test isbits(ramped_circular_motion((1.0, 2.0, 3.0), 0.5, 0.2; past=1.0).path)
    @test_throws ArgumentError ramped_circular_motion((0, 0, 0), 0.5, 0.2; past=-1)
    @test_throws ArgumentError ramped_circular_motion((0, 0, 0), 0.5, 0.2; depart=0.9)
    @test_throws ArgumentError ramped_circular_motion((0, 0, 0), 0.5, 0.2; n=8)
end

@testset "The spun-up past's retarded field is the circle's after -T and the static charge's before" begin
    # Guards the M8f data: where the retarded time lies after `-T`, the
    # spun-up past's retarded field must be the circle forever's, which
    # M8d and M8e measured clean; before `-T - Δ` it must be the field of
    # the charge at rest where the spin-up starts; between, it must still
    # solve the wave equation.
    Q, T₀, Δ = 0.07, 30.0, 10.0
    centre = SA[0.1, 0.2, 0.3]
    circle = circular_motion(centre, 0.5, 0.2).path
    path = ramped_circular_motion(centre, 0.5, 0.2; past=T₀, ramp=Δ, n=4).path
    for (x, t) in ((SA[6.0, 7.0, 2.0], 0.0), (SA[-12.0, 3.0, 9.0], 0.0),
                   (SA[0.4, 0.2, 0.5], 0.0), (SA[20.0, -5.0, 3.0], 5.0))
        Φ, Π = retarded_field(path, 1, Q, x, t)
        Φc, Πc = retarded_field(circle, 1, Q, x, t)
        @test Φ ≈ Φc rtol = 1e-14
        @test Π ≈ Πc rtol = 1e-12
    end
    zrest = path(-100.0)[1].z
    for (x, t) in ((SA[50.0, 20.0, 10.0], 0.0), (SA[-30.0, 50.0, -40.0], 2.0))
        Φ, Π = retarded_field(path, 1, Q, x, t)
        @test Φ ≈ -Q / (4π * norm(x - zrest)) rtol = 1e-14
        @test abs(Π) < 1e-15 * abs(Φ)
    end
    # About 35 away the retarded time lies in the spin-up.
    x, t, h = SA[30.0, 15.0, 10.0], 0.0, 1e-2
    Φ(x, t) = retarded_field(path, 1, Q, x, t)[1]
    @test -T₀ - Δ < t - norm(x - centre) < -T₀
    e = (SA[h, 0, 0], SA[0, h, 0], SA[0, 0, h])
    ∇² = sum((Φ(x + d, t) - 2Φ(x, t) + Φ(x - d, t)) / h^2 for d in e)
    ∂ₜ² = (Φ(x, t + h) - 2Φ(x, t) + Φ(x, t - h)) / h^2
    @test abs(∂ₜ² - ∇²) < 1e-3 * abs(∇²)
    @test retarded_field(path, 1, Q, x, t)[2] ≈ (Φ(x, t + h) - Φ(x, t - h)) / 2h rtol = 1e-4
end

@testset "At t = 0 of hyperbolic motion the remainder is the series beyond the orders carried" begin
    # Guards the hyperbolic run's initial remainder, `Φ_ret - Ψ̂_pt`: at
    # `t = 0` the retarded field is `-(Q/4πr) Σₖ (-gr/2)^k P_k(cos θ)`, and
    # the particle field carried to `order` holds its terms `k ≤ order`,
    # so the difference is the rest of the series, regular at the particle,
    # up to the roundoff of the two `-Q/4πr` that cancel. The point model
    # must be a point at these radii.
    Q, g = 0.07, 0.3
    z = SA[0.1, 0.2, 0.3]
    path = hyperbolic_motion(z, g; direction=(0, 1, 0)).path
    m = point_model(ParticleModel(; Q, ℓ=1e-3, M=1.0))
    @test m.ℓ == 1e-12
    # Legendre polynomials by the recurrence.
    function legendre(k, c)
        p₀, p₁ = one(c), c
        k == 0 && return p₀
        for n in 1:(k - 1)
            p₀, p₁ = p₁, ((2n + 1) * c * p₁ - n * p₀) / (n + 1)
        end
        return p₁
    end
    tr = path(0.0)[1]
    for order in (2, 3), (r, θ) in ((0.01, 0.3), (0.2, 1.2), (1.0, 2.5))
        x = z + r * SA[sin(θ), cos(θ), 0.0]
        rest = -Q / (4π * r) * sum((-g * r / 2)^k * legendre(k, cos(θ)) for k in (order + 1):40)
        δ = retarded_field(path, 1, Q, x, 0.0)[1] - particle_field(m, x, tr; order)
        # Roundoff of the two `-Q/4πr`, which cancel.
        @test δ ≈ rest rtol = 1e-9 atol = 10eps() * Q / (4π * r)
    end
end

# The suite's nested forest about a particle in hyperbolic motion: three
# cubes about the centre, the finest of half-width 2, `roots·E` elements
# across each (`roots = 4` gives `hₑ = 1/2`, `8` gives `1/4`), one
# particle of `ℓ = 0.01` off the element corners accelerating along `y`
# with `g = 0.2`, the window `R₁ = 1`, `R₂ = 1.75`, and the retarded field
# in the ball of radius 1.5 about the centre, blended out by 1.9.
function hyperbolic_box(roots; g=0.2, order=3)
    w₀, levels = 2.0, 3
    L = 2^levels * w₀
    forest = wave_forest(Float64, Val(3), 3, 2; roots=roots, L=L,
                         refined=nested_cubes(Val(3), w₀, levels))
    wave = WaveProblem(forest, Element(Float64, 3))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(c, 1.0, 1.75; continuity=4)
    motion = hyperbolic_motion(c + SVector(0.11, 0.07, 0.13), g; direction=(0, 1, 0))
    return InspiralProblem(wave, (m,), w, motion; order), Window(c, 1.5, 1.9; continuity=4)
end

@testset "In hyperbolic motion the total field is the retarded one, and the force falls" begin
    # Guards the split for an accelerated particle, with everything an
    # accelerated particle adds to M3's uniform motion: the near field, its
    # time derivatives and the source's bracket. Started from the retarded
    # field, `Φ_r + Φ_p` has to stay the retarded field away from the
    # particle, to an error that falls with `hₑ`, and the remainder's
    # force, zero at `t = 0` in the continuum, has to fall too. The rates
    # are measured ("Prescribed paths" under "Measured results" in
    # `CODE.md`); the coarse forest is not in the asymptotic range.
    T = 0.25                            # v = 0.05 at g = 0.2
    results = map((4, 8)) do roots
        p, wx = hyperbolic_box(roots)
        u0 = inspiral_initial_state(p; retarded=wx)
        dt = oldg_timestep(p.wave; cfl=1 // 2)
        integ = solve!(inspiral_integrator(p, u0, 0.0, T; dt=dt))
        ue = inspiral_initial_state(p; t=integ.t, retarded=wx)
        c = p.window.centre
        z = p.motion.path(integ.t)[1].z
        (; err=ball_norm(p, integ.u .- ue, c, 1.0; hole=(z, 0.25)),
         force=norm(particle_forces(p, integ.u, integ.t)[1]),
         force0=norm(particle_forces(p, u0, 0.0)[1]))
    end
    coarse, fine = results
    @test coarse.err / fine.err > 5
    @test coarse.force / fine.force > 10
    @test fine.force < 1e-5
    # At t = 0 the remainder is the time-symmetric field beyond order 3,
    # whose force is that of a term of order 5.
    @test coarse.force0 < 1e-8 && fine.force0 < 1e-8
end

@testset "A retarded window past every light cone is refused" begin
    # Guards the fill's check for nodes without a retarded time: the
    # callback, a kernel argument on a device, cannot throw the error
    # itself and marks them `NaN` instead, which must not reach the state.
    # Behind the particle by more than `1/g = 5` no past light cone of
    # hyperbolic motion reaches, and a window of radius 7 about the box's
    # centre takes nodes there.
    p, _ = hyperbolic_box(4)
    c = p.window.centre
    @test_throws "no retarded time found" inspiral_initial_state(
        p; retarded=Window(c, 7.0, 7.5; continuity=4))
end

@testset "On a circle the remainder's force is radiation reaction" begin
    # Guards the sign and the size of the force on an accelerated
    # particle, which no uniform motion can: on a circle the remainder
    # holds the radiation-reaction field `-(Q/12π)ȧ·(x - z)`, whose force
    # `(Q²/12π)ȧ` opposes the motion. Started from the retarded field, the
    # coarse forest gets its sign and its size, at `v = 0.3`, where the
    # relativistic corrections are `γ⁴ ≈ 1.2`; its convergence is measured
    # at `v = 0.05` ("Prescribed paths" under "Measured results" in
    # `CODE.md`).
    p, _ = hyperbolic_box(8)
    c = p.window.centre
    m = p.models[1]
    motion = circular_motion(c + SVector(0.011, 0.007, 0.013), 0.25, 0.3)
    p = InspiralProblem(p.wave, p.models, p.window, motion)
    wx = Window(c, 1.5, 1.9; continuity=4)
    u0 = inspiral_initial_state(p; retarded=wx)
    tr = motion.path(0.0)[1]
    rr = m.Q^2 / 12π * dot(tr.ȧ, tr.v) / norm(tr.v)
    @test rr < 0
    F = particle_forces(p, u0, 0.0)[1]
    @test 0.5 < dot(F, tr.v) / norm(tr.v) / rr < 2
    dt = oldg_timestep(p.wave; cfl=1 // 2)
    integ = solve!(inspiral_integrator(p, u0, 0.0, 0.125; dt=dt))
    tr = motion.path(integ.t)[1]
    F = particle_forces(p, integ.u, integ.t)[1]
    @test 0.5 < dot(F, tr.v) / norm(tr.v) / rr < 2
end

# The force of the exact remainder `W_x(Φ_ret - Ψ̂_pt)` as the pairing reads
# it at `t`, without evolving it, and the same with the lab slice's order 1
# in its point form, the t′² term of Φ_C, put back into the remainder, as
# the particle field did not carry it before M4.
function exact_reading(p, wx, t)
    u = inspiral_initial_state(p; t, retarded=wx)
    tr = p.motion.path(t)[1]
    point = point_model(p.models[1])
    T₂(x, var) = begin
        X = SVector{3}(x)
        w = wx(X)
        (iszero(w) || norm(X - tr.z) < 1e-6) && return 0.0
        var == 1 && return w * lab_slice(point, Val(1), X, tr)
        return w * TWP.ForwardDiff.derivative(s -> lab_slice(point, Val(1), X,
                                                             TWP.trajectory_at(tr, s)), 0.0)
    end
    fill_at_nodes!(T₂, p.wave.fs, p.wave.element)
    back = zero(u)
    fb = field_state(back, p)
    TWP.TreeAMR.zerofill!(TWP.statearray(fb, p.wave.fs), TWP.get_backend(p.wave.fs.work))
    TWP.gather!(fb, p.wave.fs)
    truncate_state!(fb, p.wave)
    return particle_forces(p, u, t)[1], particle_forces(p, u .+ back, t)[1], tr
end

@testset "With the lab slice carried, the exact remainder exerts no force across the motion" begin
    # Guards the lab slice in the particle field, whose order 1 is, for a
    # point, the t′² term of Φ_C. Without
    # it the remainder holds an `r⁰` structure proportional to `v²a`, which
    # the grid reads as a force along the acceleration that grows as the
    # elements shrink (M4's first series, "Prescribed paths" in CODE.md).
    # Read without evolving it, the exact remainder of hyperbolic motion at
    # `v = 0.05` exerts almost no force along the acceleration, and that
    # of the circle almost none radially, at both resolutions, while with
    # the term put back the radial reading grows with resolution.
    for roots in (4, 8)
        p, wx = hyperbolic_box(roots)
        F, Fback, tr = exact_reading(p, wx, 0.25)
        â = tr.a / norm(tr.a)
        @test abs(dot(F, â)) < 0.01 * abs(dot(Fback, â))
    end
    radial = map((4, 8)) do roots
        p, wx = hyperbolic_box(roots)
        c = p.window.centre
        motion = circular_motion(c + SVector(0.011, 0.007, 0.013), 0.25, 0.3)
        p = InspiralProblem(p.wave, p.models, p.window, motion)
        F, Fback, tr = exact_reading(p, wx, 0.0)
        ρ̂ = -tr.a / norm(tr.a)
        rr = p.models[1].Q^2 / 12π * norm(tr.ȧ)
        (; with=dot(F, ρ̂) / rr, without=dot(Fback, ρ̂) / rr)
    end
    coarse, fine = radial
    @test abs(coarse.with) < 0.1 && abs(fine.with) < 0.1
    @test abs(fine.with) < abs(coarse.with)
    @test abs(fine.without) > 2 * abs(coarse.without)
end

@testset "Uniform motion's far data are the default's at a = 0 and the tangent's field on a circle" begin
    # Guards `far = :uniform` (INSPIRAL-PLAN.md, M8d): data that beyond the
    # window are the field of a charge that has always moved on its
    # tangent. At `a = 0` they must be the default's `(1 - W)Ψ̂` bit for
    # bit, since every near-field and lab-slice term vanishes there. On a
    # circle `Φ_r` must be the boosted static field of the particle's `z`
    # and `v`, and `Π_r` its derivative along the tangent line, a finite
    # difference in `t`, not `𝒟Ψ̂₀`, which carries the acceleration through
    # `v̇` and is not the time derivative of any solution.
    p, wx = hyperbolic_box(4)
    c = p.window.centre
    uniform = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(0.2, 0.1, 0.0),))
    q = InspiralProblem(p.wave, p.models, p.window, uniform)
    @test inspiral_initial_state(q; far=:uniform) == inspiral_initial_state(q)
    @test inspiral_initial_state(q; t=0.3, far=:uniform) ==
          inspiral_initial_state(q; t=0.3)
    @test_throws ArgumentError inspiral_initial_state(q; far=:retarded)
    @test_throws ArgumentError inspiral_initial_state(q; far=:uniform, retarded=wx)

    m = p.models[1]
    motion = circular_motion(c + SVector(0.011, 0.007, 0.013), 0.25, 0.3)
    q = InspiralProblem(p.wave, p.models, p.window, motion)
    t = 0.4
    tr = motion.path(t)[1]
    f = TWP.RemainderData(q, (tr,), t; far=:uniform)
    Φ₀(x, s) = TWP.boosted_static(m, x, tr.z + s * tr.v, tr.v)
    δ = 1e-4
    for x in (c + SVector(3.0, 0.5, -0.4), c + SVector(-1.0, 2.7, 1.3),
              c + SVector(0.2, -0.3, 4.1))
        @test norm(x - c) > p.window.R₂
        @test f(x, 1) == Φ₀(x, 0.0)
        fd = (Φ₀(x, δ) - Φ₀(x, -δ)) / 2δ
        @test f(x, 2) ≈ fd rtol = 1e-7
        𝒟Ψ̂₀ = TWP.particle_rate(m, x, tr; order=0)
        @test abs(f(x, 2) - 𝒟Ψ̂₀) > 1e-3 * abs(fd)
    end
    # Inside the window's inner ball the remainder is zero.
    @test f(tr.z + SVector(0.1, 0.0, 0.0), 1) == 0
    @test f(tr.z + SVector(0.1, 0.0, 0.0), 2) == 0
end

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
    # Guards `CirclePath` and `HyperbolicPath`: `v`, `a`, `ȧ`, `ä` and
    # `a⃛` must be the derivatives of `z`, and the motions those asked
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
        @test abs(Π) < 1e-15 * abs(Φ)
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
# it at `t`, without evolving it, and the same with the t′² term put back
# into the remainder, as the particle field did not carry it before M4.
function exact_reading(p, wx, t)
    u = inspiral_initial_state(p; t, retarded=wx)
    tr = p.motion.path(t)[1]
    point = point_model(p.models[1])
    T₂(x, var) = begin
        X = SVector{3}(x)
        w = wx(X)
        (iszero(w) || norm(X - tr.z) < 1e-6) && return 0.0
        var == 1 && return w * lab_slice(point, X, tr)
        return w * TWP.ForwardDiff.derivative(s -> lab_slice(point, X,
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

@testset "With the t′² term carried, the exact remainder exerts no force across the motion" begin
    # Guards the t′² term of Φ_C's lab slice in the particle field. Without
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

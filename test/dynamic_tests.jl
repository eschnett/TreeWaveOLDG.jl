# Dynamic particles (INSPIRAL-PLAN.md, M6a): the analytic parts of the
# equation of motion, each held to a closed form where one exists, and
# the solve of the particles' jets from the state. See "The equations of
# motion" in `CODE.md`.

using LinearAlgebra: dot, norm
using StaticArrays: SVector
using TreeWaveOLDG: self_force_addot, self_force_a2a, self_potential_a2, lorentz_factor,
                    near_field, field_force

@testset "The own near field's force is M1's self-force, and -δm γ a moving" begin
    # Guards the quadrature over the support and its map to the lab: at
    # rest the own near field must give M1's generated coefficients, and
    # moving its order-1 force must be `-δm γ a` in every direction, the
    # covariant renormalization `M₀ + δm = M` in `mγ a`, which a wrong
    # boost of the points, a wrong volume element `1/γ`, a lost lab slice
    # or a lost rate term `-v∫ρ ġ` breaks.
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    δm = self_inertia(m)
    z = SVector(0.1, 0.2, 0.3)
    a, ȧ, ä = SVector(0.3, -0.2, 0.1), SVector(0.05, 0.02, -0.04), SVector(0.01, 0.03, 0.02)
    rule = SupportRule(Float64)
    rest = Trajectory(; z, v=zero(z), a, ȧ, ä, a⃛=SVector(0.002, 0.0, 0.001))
    F1, P1 = own_force(m, rest, rule; order=1)
    @test norm(F1 + δm * a) <= 1e-14 * δm * norm(a)
    @test abs(P1) <= 1e-14 * m.Q * norm(a)
    F2, P2 = own_force(m, rest, rule; order=2)
    @test norm(F2 - F1) <= 1e-14 * norm(F1)
    @test P2 ≈ m.Q * self_potential_a2(m.Q, m.ℓ) * dot(a, a) rtol = 1e-12
    # The order-3 term alone, whose force is far below order 1's.
    h3(x) = (near_field(m, Val(3), x, rest), 0.0)
    F3, _ = field_force(m, rest, rule, h3)
    @test F3 ≈ self_force_addot(m.Q, m.ℓ) * ä + self_force_a2a(m.Q, m.ℓ) * dot(a, a) * a rtol = 1e-12
    fine = SupportRule(Float64; n_r=24, n_θ=12)
    for v in (SVector(0.2, 0.0, 0.0), SVector(0.0, 0.0, 0.2), SVector(0.1, -0.1, 0.1))
        tr = Trajectory(; z, v, a, ȧ, ä)
        F1, _ = own_force(m, tr, rule; order=1)
        @test F1 ≈ -δm * lorentz_factor(v) * a rtol = 1e-13
        F3, P3 = own_force(m, tr, rule; order=3)
        F3f, P3f = own_force(m, tr, fine; order=3)
        @test F3 ≈ F3f rtol = 1e-13
        @test P3 ≈ P3f rtol = 1e-12
    end
end

@testset "Another particle's field pushes as its value at the particle" begin
    # Guards the other particle's average: outside a particle at rest its
    # field `-Q_b/4πr` is harmonic, so its average over the spherical
    # density is its value at the centre and the force is `-Q∇Φ_b(z)`,
    # which a wrong sign of `∫(∇ρ) g` or a wrong weight breaks.
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    mb = ParticleModel(; Q=-0.1, ℓ=1e-2, M=2.0)
    z, zb = SVector(0.1, 0.2, 0.3), SVector(0.9, -0.3, 0.5)
    tr, trb = Trajectory(; z, v=zero(z)), Trajectory(; z=zb, v=zero(zb))
    F, P = other_force(m, tr, mb, trb, SupportRule(Float64))
    r = z - zb
    Φ = -mb.Q / (4π * norm(r))
    @test F ≈ -m.Q * (mb.Q / (4π * norm(r)^3)) * r rtol = 1e-12
    @test P ≈ m.Q * Φ rtol = 1e-12
end

# A constant external force, for the tests.
struct ConstantForce{T}
    F::SVector{3,T}
end
(f::ConstantForce)(z, v) = f.F

@testset "A pushed particle at rest accelerates as M a = F" begin
    # Guards the equation of motion's assembly: from rest in its own
    # static field, under a constant force, the solve must give
    # `M a = F` with the physical mass, its own near field's `-δm a`
    # making up `M - M₀`, and no rate; a lost own force, a bare mass in
    # its place, a wrong sign of the external force or of the grid's
    # part breaks it.
    L, P = 4.0, 3
    forest = wave_forest(Float64, Val(3), P, 2; roots=4, L=L)
    wave = WaveProblem(forest, Element(Float64, P))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    past = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(0.0, 0.0, 0.0),))
    F = SVector(0.01, -0.02, 0.005)
    dyn = Dynamic((ConstantForce(F),), past)
    p = InspiralProblem(wave, (m,), Window(c, 1.0, 1.75; continuity=P + 1), dyn)
    u = inspiral_initial_state(p)
    tr = TreeWaveOLDG.trajectories(dyn, p, u, 0.0)[1]
    @test tr.a ≈ F / m.M rtol = 1e-8
    @test norm(tr.ȧ) <= 1e-12 * norm(F)
    @test dyn.scratch.last.residual <= 1e-14 * norm(F)
    @test particle_state(u, p)[1][1] == tr.z
    # The hand-over: from rest, `(1 - S) M a + S (M a - F) = 0` gives
    # `a = S F/M`, and its derivatives in time those of `S`, which a
    # blend whose time derivatives were lost would not; the remainder, a
    # particle's at rest, answers the accelerating jets at `10⁻⁶`.
    onset = 2.0
    dyn = Dynamic((ConstantForce(F),), past; onset)
    p = InspiralProblem(wave, (m,), Window(c, 1.0, 1.75; continuity=P + 1), dyn)
    @test TreeWaveOLDG.trajectories(dyn, p, u, 0.0)[1] === past.path(0.0)[1]
    t = 0.6
    tr = TreeWaveOLDG.trajectories(dyn, p, u, t)[1]
    S(d) = smoothstep(4, t / onset, d) / onset^d
    @test tr.a ≈ S(0) * F / m.M rtol = 1e-4
    @test tr.ȧ ≈ S(1) * F / m.M rtol = 1e-4
    @test tr.ä ≈ S(2) * F / m.M rtol = 1e-4
    # The Jacobian found during the ramp is the blend's, and the chord
    # iteration converges with it after the take-over too, so without
    # finding it again there the solve would keep it for good, and
    # `bin/dynamic.jl` would record the blend's conditioning.
    @test dyn.scratch.last.fresh && dyn.scratch.last.σ == smoothstep(4, t / onset)
    TreeWaveOLDG.trajectories(dyn, p, u, onset + 0.1)
    @test dyn.scratch.last.fresh && dyn.scratch.last.σ == 1
    TreeWaveOLDG.trajectories(dyn, p, u, onset + 0.2)
    @test !dyn.scratch.last.fresh && dyn.scratch.last.σ == 1
end

@testset "A spring-held particle keeps its circle, slowed by the remainder's force" begin
    # Guards the solve on the circle M4 measured: from the circle's
    # retarded field, a particle held by the spring that keeps it there
    # must get the circle's jets, but for the remainder's tangential force
    # `F_t`, radiation reaction, which slows it, `Mγ a·τ̂ = F_t`; a lost
    # grid part, a grid part read at the wrong `t`, a wrong `γ` or an own
    # field that pushes along the motion breaks it.
    w₀, levels = 2.0, 3
    L = 2^levels * w₀
    forest = wave_forest(Float64, Val(3), 3, 2; roots=4, L=L,
                         refined=nested_cubes(Val(3), w₀, levels))
    wave = WaveProblem(forest, Element(Float64, 3))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(c, 1.0, 1.75; continuity=4)
    F, circle = spring_circle(m, c + SVector(0.011, 0.007, 0.013), 0.25, 0.3)
    dyn = Dynamic((F,), circle)
    p = InspiralProblem(wave, (m,), w, dyn)
    u = inspiral_initial_state(p; retarded=Window(c, 1.5, 1.9; continuity=4))
    tr = TreeWaveOLDG.trajectories(dyn, p, u, 0.0)[1]
    tc = circle.path(0.0)[1]
    τ = tc.v / norm(tc.v)
    Ft = dot(particle_forces(p, u, 0.0)[1], τ)
    rr = m.Q^2 / 12π * norm(tc.ȧ)
    @test 0.5 < -Ft / rr < 2
    @test m.M * lorentz_factor(tc.v) * dot(tr.a, τ) ≈ Ft rtol = 1e-3
    # Across the motion it leaves the circle only by the remainder's force
    # and the own near field's beyond `-δm γ a`, the circle's spring
    # holding neither.
    Fg = particle_forces(p, u, 0.0)[1]
    Fo = own_force(m, tc, dyn.rule)[1] + self_inertia(m) * lorentz_factor(tc.v) * tc.a
    across(f) = f - dot(f, τ) * τ
    @test m.M * lorentz_factor(tc.v) * across(tr.a - tc.a) ≈ across(Fg + Fo) rtol = 1e-2
    @test tr.ȧ ≈ tc.ȧ rtol = 1e-2
    @test tr.ä ≈ tc.ä rtol = 1e-2
end

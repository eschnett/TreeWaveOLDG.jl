# The coupled problem: the remainder field stepped with particles on a
# prescribed path. The right-hand side has to be the wave equation's minus
# the projected source and nothing else, the limiter has to reach the
# field inside the longer state vector, and the remainder of a particle at
# rest or in uniform motion has to stay where it starts: anything else is
# an error of the split, the window or the transfer that every later
# milestone would inherit.

using IMEXRungeKutta: IMEXProblem, RK4, init, solve!, step!
using LinearAlgebra: norm
using StaticArrays

# A coarse 3D box of side 4 around one particle of size `ℓ = 0.01`, off
# the element corners, moving uniformly with velocity `(v, 0, 0)`, the
# window `R₁ = 1`, `R₂ = 1.75`: `roots = 4` gives `hₑ = 1/2`, `roots = 8`
# gives `hₑ = 1/4`.
function inspiral_box(roots; v=0.0, P=3, order=3)
    L = 4.0
    forest = wave_forest(Float64, Val(3), P, 2; roots=roots, L=L)
    wave = WaveProblem(forest, Element(Float64, P))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(c, 1.0, 1.75; continuity=P + 1)
    motion = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(v, 0.0, 0.0),))
    return InspiralProblem(wave, (m,), w, motion; order=order)
end

@testset "The inspiral's right-hand side is wave_rhs! minus the projected source" begin
    # Guards the assembly: the field part of `du` must be exactly what
    # `wave_rhs!` and `project_source!` give on the field part of `u`, the
    # particles must move as their path says, and nothing may be written
    # to `u` or carried from one call to the next, which RK4 relies on.
    p = inspiral_box(4; v=0.2)
    u = inspiral_initial_state(p)
    u0 = copy(u)
    du1, du2 = similar(u), similar(u)
    inspiral_rhs!(du1, u, p, 0.1)
    inspiral_rhs!(du2, 2 .* u, p, 0.3)
    inspiral_rhs!(du2, u, p, 0.1)
    @test u == u0
    @test du1 == du2

    ref = similar(field_state(u, p))
    wave_rhs!(ref, field_state(u, p), p.wave, 0.1)
    trajs = p.motion.path(0.1)
    S = ParticleSource(p.models, p.window, trajs)
    project_source!(ref, p.wave, p.tables, S, map(ParticleSupport, p.models, trajs);
                    ball=(p.window.centre, p.window.R₂))
    @test field_state(du1, p) == ref
    # The source does something: the window's terms reach the nodes.
    wave_rhs!(ref, field_state(u, p), p.wave, 0.1)
    @test norm(field_state(du1, p) - ref, Inf) > 1e-3

    (z, v), = particle_state(u, p)
    @test z == p.motion.path(0.0)[1].z
    @test v == trajs[1].v
    (ż, v̇), = particle_state(du1, p)
    @test ż == v
    @test v̇ == trajs[1].a
end

@testset "Every inspiral stage and step has v truncated, and only through the limiter" begin
    # Guards `inspiral_stage!` and the partition: the limiter must find
    # the field inside the longer state vector and leave the particles
    # alone, and without it the stages carry `v`'s top mode. The stage
    # arithmetic by block owner, with the particles on thread 1, must be
    # bitwise the broadcast.
    p = inspiral_box(4; v=0.2)
    u0 = inspiral_initial_state(p)
    @test untruncated(field_state(u0, p), p.wave) < 1e-13
    dt = oldg_timestep(p.wave; cfl=1 // 2)
    for truncated in (true, false)
        worst = Ref(0.0)
        function rhs!(du, u, p, t)
            worst[] = max(worst[], untruncated(field_state(u, p), p.wave))
            inspiral_rhs!(du, u, p, t)
            return nothing
        end
        limiters = truncated ?
                   (stage_limiter=inspiral_stage!, step_limiter=inspiral_stage!) : (;)
        integ = init(IMEXProblem(rhs!, nothing, u0, (0.0, 2dt), p), RK4(); dt=dt,
                     limiters...)
        solve!(integ)
        steps = untruncated(field_state(integ.u, p), p.wave)
        if truncated
            @test worst[] < 1e-13
            @test steps < 1e-13
            owner = solve!(inspiral_integrator(p, u0, 0.0, 2dt; dt=dt))
            @test owner.u == integ.u
            @test particle_state(owner.u, p)[1][1] ≈ p.motion.path(2dt)[1].z rtol = 1e-15
        else
            @test worst[] > 1e-6
            @test steps > 1e-6
        end
    end
    part = inspiral_partition(p, u0)
    @test length(part) == Threads.nthreads()
    @test sort(reduce(vcat, map(r -> reduce(vcat, collect.(r)), part))) == eachindex(u0)
end

@testset "A particle at rest or in uniform motion keeps its remainder, and no force" begin
    # Guards the window, the source's split and the transfer together:
    # `(1 - W)Φ_C(z + vt)` is the exact remainder of a uniformly moving
    # particle, so the run may depart from it only by the window's
    # discretization error, which has to fall with `hₑ`, and the
    # remainder's force on the particle, zero in the continuum, has to
    # fall too. A wrong sign in the source, a source projected with the
    # wrong weights, or a particle field that is not the uniformly moving
    # one would leave an error that does not fall. The errors are taken in
    # the ball of radius `R₂` that the box's faces have not reached
    # (`ball_norm`). The rates are measured ("Measured results",
    # "Uniform motion" in `CODE.md`); the coarse box is not yet in the
    # asymptotic range, where they are `P`.
    T = 0.25
    for v in (0.0, 0.2)
        results = map((4, 8)) do roots
            p = inspiral_box(roots; v=v)
            u0 = inspiral_initial_state(p)
            dt = oldg_timestep(p.wave; cfl=1 // 2)
            integ = solve!(inspiral_integrator(p, u0, 0.0, T; dt=dt))
            ue = inspiral_initial_state(p; t=integ.t)
            c = p.window.centre
            (; err=ball_norm(p, integ.u .- ue, c, p.window.R₂),
             field=ball_norm(p, ue, c, p.window.R₂),
             force=norm(particle_forces(p, integ.u, integ.t)[1]),
             force0=norm(particle_forces(p, u0, 0.0)[1]),
             z=particle_state(integ.u, p)[1][1] - p.motion.path(integ.t)[1].z)
        end
        coarse, fine = results
        @test coarse.force0 == 0 && fine.force0 == 0
        @test coarse.err < coarse.field
        @test coarse.err / fine.err > 3
        @test coarse.force < 1e-4
        @test coarse.force / fine.force > 100
        @test all(r -> norm(r.z, Inf) < 1e-13, results)
    end
end

@testset "The ball norm is the quadrature norm restricted to the ball" begin
    # Guards `ball_norm`'s node positions, taken from the block origins
    # through the fold's tag: a ball that holds the whole box gives
    # `quadrature_norm`, and one that holds a single node gives its value.
    p = inspiral_box(4)
    u = inspiral_initial_state(p)
    fu = field_state(u, p)
    @test ball_norm(p, u, p.window.centre, 10.0) ≈ quadrature_norm(p.wave.fs, p.wave.element, fu)
    @test ball_norm(p, u, p.window.centre, 10.0; p=Inf) == norm(fu, Inf)
    x = node_coordinates(p.wave.fs, p.wave.element, 5, (3, 2, 4))
    ua = TreeWaveOLDG.statearray(fu, p.wave.fs)
    @test ball_norm(p, u, x, 1e-6; vars=1) ≈ abs(ua[3, 2, 4, 1, 5])
end

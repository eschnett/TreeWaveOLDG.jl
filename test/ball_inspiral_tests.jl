# The particles in the seven-patch ball's cube (INSPIRAL-PLAN.md, M8c):
# the transfer works in the cube's physical coordinates through its affine
# map, so on the cube's finest level it has to do what it does on nested
# boxes of the same physical layout, element for element, and it has to
# refuse a ball or a support that leaves the cube. See "Particles in the
# ball" in `CODE.md`.

using IMEXRungeKutta: solve!
using LinearAlgebra: dot, norm
using StaticArrays

const TWB = TreeWaveOLDG

# One particle of `ℓ = 0.01` near the origin, moving uniformly with
# velocity `(v, 0, 0)`, in the window `R₁ = 1`, `R₂`, on the wave problem
# `wave`, as `nested_tests.jl`'s `particle_problem` has it about the
# origin.
function origin_problem(wave; v=0.2, P=3, R₂=1.75, offset=(0.11, 0.07, 0.13),
                        tables=nothing)
    centre = SVector(0.0, 0.0, 0.0)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(centre, 1.0, R₂; continuity=P + 1)
    motion = uniform_motion((centre + SVector(offset),), (SVector(v, 0.0, 0.0),))
    return InspiralProblem(wave, (m,), w, motion; tables=tables)
end

# `levels` nested cubes about the origin, the finest of half-width 2, in
# the seven-patch ball's cube of half-width `Lc = 2^(levels-1)·2` and
# `roots` roots across, with shells of `m` radial roots to a Sommerfeld
# sphere at `R1 = 3Lc`, where the shells beside the cube's corners no
# longer set the step. `roots = 4` gives the finest `hₑ = 1/2`, and the
# cube's map has the scale `2Lc/roots = 4`.
function ball_problem(roots; levels=3, m=4, R1=24.0, P=3, kwargs...)
    Lc = 2.0^(levels - 1) * 2
    forest, geo = sevenpatch_forest(Float64, P, 2; roots=(roots, m), Lc=Lc, R1=R1,
                                    refined=nested_cubes(Val(3), roots / 2^levels,
                                                         levels))
    wave = WaveProblem(forest, Element(Float64, P); geometry=geo,
                       boundary=RadiativeBoundary())
    return origin_problem(wave; P, kwargs...)
end

# The same nested cubes as a periodic box about the origin, of side
# `2^levels·2`: the nested boxes of M3b, shifted so that their physical
# coordinates are the ball's.
function centred_boxes(roots; levels=3, P=3, kwargs...)
    L = 2.0^levels * 2
    forest = Forest{Float64}(ntuple(_ -> roots, 3); N=2(P + 1),
                             periodic=(true, true, true),
                             extents=ntuple(_ -> (-L / 2, L / 2), 3))
    regions = [map(r -> (r[1] - L / 2, r[2] - L / 2), region)
               for region in nested_cubes(Val(3), 2.0, levels)]
    TWB.refine_regions!(forest, regions, roots)
    return origin_problem(WaveProblem(forest, Element(Float64, P)); P, kwargs...)
end

# The finest level's blocks of `p`'s particle frame by their physical
# origin.
function finest_blocks(p)
    forest = p.wave.fs.forest
    inpatch = TWB.frame_blocks(forest, p.frame)
    return Dict(Tuple(TWB.frame_origin(forest, p.frame, blockkey(p.wave.fs, b))) => b
                for b in 1:nblocks(p.wave.fs)
                if inpatch[b] && level(blockkey(p.wave.fs, b)) == maxlevel(forest))
end

@testset "In the ball's cube the source and the pairing are the nested boxes'" begin
    # Guards the transfer through the cube's affine map: on the cube's
    # finest level, at the scale 4, the projection and the pairing have to
    # be the nested boxes' at the same physical positions bit for bit,
    # since the mapped block origins and spacings are the boxes' exactly;
    # the initial remainder, filled at `physical_coordinates`, to rounding.
    # A distance, a rule or an element box taken in the cube's logical
    # coordinates, or a pairing missing the Jacobian, would be off by
    # powers of 4. The far rule starts at `rfar = 3hₑ`, so that all three
    # rules run inside the ball.
    tables = TransferTables(Float64, 3; graded=((1, 14), (2, 10)), rfar=3)
    p = ball_problem(4; tables)
    q = centred_boxes(4; tables)
    @test p.frame.patch == 1 && p.frame.scale == 4 && p.frame.origin == fill(-8.0, 3)
    @test q.frame.patch == 0
    u, uq = inspiral_initial_state(p), inspiral_initial_state(q)
    trajs = p.motion.path(0.1)
    S = ParticleSource(p.models, p.window, trajs)
    sp, sq = zero(field_state(u, p)), zero(field_state(uq, q))
    supports = map(ParticleSupport, p.models, trajs)
    ball = (p.window.centre, p.window.R₂)
    project_source!(sp, p.wave, p.tables, S, supports; ball=ball)
    project_source!(sq, q.wave, q.tables, S, supports; ball=ball)
    @test norm(sq, Inf) > 1e-3
    # A field that is not zero under the particle, for the pairing: the
    # projection of a smooth function into both variables.
    z = trajs[1].z
    smooth = x -> exp(-sum(abs2, x - z))
    gp, gq = zero(sp), zero(sq)
    for var in (1, 2)
        project_source!(gp, p.wave, p.tables, smooth, supports; ball, var, factor=var)
        project_source!(gq, q.wave, q.tables, smooth, supports; ball, var, factor=var)
    end

    fp, fq = p.wave.fs, q.wave.fs
    blocks = finest_blocks(p)
    @test length(blocks) == length(finest_blocks(q)) == 64
    sa_p, sa_q = TWB.statearray(sp, fp), TWB.statearray(sq, fq)
    ua_p = TWB.statearray(field_state(u, p), fp)
    ua_q = TWB.statearray(field_state(uq, q), fq)
    # Every block the ball meets is one of the finest level's, and the
    # source is nowhere else.
    touched = Set(b for b in 1:nblocks(fp) if any(!iszero, sa_p[:, :, :, :, b]))
    @test touched ⊆ Set(values(blocks))
    for (origin, b) in finest_blocks(q)
        a = blocks[origin]
        @test sa_p[:, :, :, :, a] == sa_q[:, :, :, :, b]
        @test ua_p[:, :, :, :, a] ≈ ua_q[:, :, :, :, b] rtol = 1e-14
    end
    # The pairing, of the projected smooth field bit for bit, and of the
    # initial remainder to its rounding.
    m, tr = p.models[1], trajs[1]
    ρ(x) = particle_density(m, x, tr.z, tr.v)
    ∇ρ(x) = TWB.ForwardDiff.gradient(ρ, x)
    s = supports[1]
    for var in (1, 2), K in (ρ, ∇ρ), continuous in (false, true)
        a = pairing(p.wave, p.tables, gp, K, s; var, continuous)
        @test a == pairing(q.wave, q.tables, gq, K, s; var, continuous)
        @test norm(a) > 1e-3
        @test pairing(p.wave, p.tables, field_state(u, p), K, s; var, continuous) ≈
              pairing(q.wave, q.tables, field_state(uq, q), K, s; var, continuous) rtol = 1e-13
    end
    @test particle_forces(p, u, 0.1)[1] ≈ particle_forces(q, uq, 0.1)[1] rtol = 1e-13
    # The ball's shells hold the remainder `(1 - W)Ψ̂` at their nodes'
    # physical positions.
    m0, tr0 = p.models[1], p.motion.path(0.0)[1]
    geo = p.wave.metric.geometry
    for b in (1, nblocks(fp)), I in ((1, 1, 1), (3, 6, 8))
        key = blockkey(fp, b)
        patch(fp.forest, key) == 1 && continue
        ξ = node_coordinates(fp, p.wave.element, b, I)
        x = SVector(physical_coordinates(geo, ξ, patch(fp.forest, key)))
        @test norm(x) > 8
        exact = (1 - p.window(x)) * particle_field(m0, x, tr0; order=p.order)
        @test ua_p[I..., 1, b] ≈ exact rtol = 1e-14
    end
    # And the ball norm sees the same nodes and weights inside the cube.
    for r in (0.6, 1.9)
        @test ball_norm(p, u, p.window.centre, r) ≈ ball_norm(q, uq, q.window.centre, r) rtol = 1e-13
        @test ball_norm(p, u, p.window.centre, r; p=Inf) ≈
              ball_norm(q, uq, q.window.centre, r; p=Inf) rtol = 1e-14
        # The mean of `v` is a sum that cancels, so its rounding is
        # relative to the mean of `|v|`.
        @test ball_mean(p, u, p.window.centre, r; vars=2) ≈
              ball_mean(q, uq, q.window.centre, r; vars=2) atol = 1e-13 *
              ball_norm(q, uq, q.window.centre, r; vars=2, p=1)
    end
    # Outside the cube the norm and the mean see the shells' nodes at their
    # physical positions with `√g`: a constant has itself as its mean.
    one_ = fill!(similar(u), 1.0)
    @test ball_mean(p, one_, p.window.centre, 20.0) ≈ 1 rtol = 1e-14
    @test ball_norm(p, one_, p.window.centre, 20.0; p=3) ≈ 1 rtol = 1e-14
end

@testset "The particles stay in the ball's cube" begin
    # Guards the refusal that replaces "the inspiral runs on a brick": the
    # transfer works in the cube alone and looks across no glue, so a
    # window's ball or a particle's support that leaves the cube, where
    # the shells' map is not affine, must be refused by `InspiralProblem`,
    # `project_source!` and the pairing, naming the patch; and a
    # geometry that is not a full one is refused. The thirteen-patch
    # ball's cube is the same frame.
    forest, geo = sevenpatch_forest(Float64, 3, 2; roots=(4, 1), Lc=2, R1=4)
    wave = WaveProblem(forest, Element(Float64, 3); geometry=geo,
                       boundary=RadiativeBoundary())
    f = TWB.ParticleFrame(wave)
    @test (f.patch, f.scale, f.origin) == (1, 1.0, SVector(-2.0, -2.0, -2.0))
    forest13, geo13 = thirteenpatch_forest(Float64, 3, 2; roots=(4, 1, 1), Lc=2, Rm=4,
                                           R1=6)
    wave13 = WaveProblem(forest13, Element(Float64, 3); geometry=geo13,
                         boundary=RadiativeBoundary())
    @test TWB.ParticleFrame(wave13) == f
    @test_throws "inside patch 1" origin_problem(wave; R₂=2.5)
    @test_throws "inside patch 1" origin_problem(wave; offset=(1.995, 0.0, 0.0))
    p = origin_problem(wave; offset=(1.985, 0.0, 0.0))
    fu = field_state(inspiral_initial_state(p), p)
    m = p.models[1]
    c = p.window.centre
    for (z, refused) in ((SVector(1.985, 0.0, 0.0), false),
                         (SVector(1.995, 0.0, 0.0), true),
                         (SVector(0.0, -1.995, 0.0), true))
        s = ParticleSupport(z, zero(z), m.ℓ)
        ρ = x -> particle_density(m, x, z, zero(z))
        if refused
            @test_throws "inside patch 1" pairing(p.wave, p.tables, fu, ρ, s)
            @test_throws "inside patch 1" project_source!(
                zero(fu), p.wave, p.tables, x -> 1.0, (s,); ball=(c, 1.0))
        else
            @test pairing(p.wave, p.tables, fu, ρ, s) isa Float64
            @test project_source!(zero(fu), p.wave, p.tables, x -> 1.0, (s,);
                                  ball=(c, 1.0)) isa AbstractVector
        end
    end
    @test_throws "outside the domain of patch 1" reconstruct(p.wave, p.tables, fu, (2.5, 0.0, 0.0))
    @test reconstruct(p.wave, p.tables, fu, (1.5, 0.0, 0.0)) isa Float64
    @test_throws "forest has patches" TWB.ParticleFrame(wave.fs)
end

@testset "In the ball a moving particle keeps the nested boxes' remainder" begin
    # Guards the inspiral in the ball as a whole: the cube's metric terms,
    # the shells and the transfer through the cube's map have to evolve a
    # uniformly moving particle's remainder as the nested boxes do, inside
    # their clean time, to the rounding by which the two right-hand sides
    # differ, in the error and the force alike ("Particles in the ball" in
    # `CODE.md`). With the sphere at `3Lc` the finest cube sets the step,
    # the boxes' own.
    p, q = ball_problem(4), centred_boxes(4)
    dt = oldg_timestep(q.wave; cfl=1 // 2)
    @test oldg_timestep(p.wave; cfl=1 // 2) ≈ dt rtol = 1e-12
    results = map((p, q)) do r
        u0 = inspiral_initial_state(r)
        integ = solve!(inspiral_integrator(r, u0, 0.0, 0.125; dt=dt))
        ue = inspiral_initial_state(r; t=integ.t)
        (; err=ball_norm(r, integ.u .- ue, r.window.centre, r.window.R₂),
         field=ball_norm(r, ue, r.window.centre, r.window.R₂),
         force=particle_forces(r, integ.u, integ.t)[1])
    end
    ball, boxes = results
    @test ball.err < ball.field
    @test ball.err ≈ boxes.err rtol = 1e-10
    @test ball.force ≈ boxes.force rtol = 1e-12
end

@testset "In the ball a circle steps from uniform motion's far data" begin
    # Guards `far = :uniform` on the ball (INSPIRAL-PLAN.md, M8d): the
    # ball has to take the data whose far field is that of the tangent
    # line and evolve them, and beyond the window they are far smaller
    # than the order-3 particle field's, whose near-field terms grow with
    # the distance, most of all in `Π_r`. Near the particle the two starts
    # are the same remainder, so for the first steps the force is the
    # default start's (measured on this forest: `Π_r` beyond `R₂` 90 times
    # smaller, the force along the motion after 16 steps 4 % apart).
    p = ball_problem(4)
    c = p.window.centre
    m = p.models[1]
    motion = circular_motion(c + SVector(0.011, 0.007, 0.013), 0.25, 0.3)
    p = InspiralProblem(p.wave, p.models, p.window, motion)
    dt = oldg_timestep(p.wave; cfl=1 // 2)
    hole = (c, p.window.R₂)
    results = map((:particle, :uniform)) do far
        u0 = inspiral_initial_state(p; far)
        integ = solve!(inspiral_integrator(p, u0, 0.0, 16dt; dt=dt))
        tr = motion.path(integ.t)[1]
        F = particle_forces(p, integ.u, integ.t)[1]
        (; far0=ball_norm(p, u0, c, 5.0; hole), farΠ0=ball_norm(p, u0, c, 5.0; vars=2, hole),
         far=ball_norm(p, integ.u, c, 5.0; hole), finite=all(isfinite, integ.u),
         Ft=dot(F, tr.v) / norm(tr.v), rr=m.Q^2 / 12π * dot(tr.ȧ, tr.v) / norm(tr.v))
    end
    particle, uniform = results
    @test uniform.finite
    @test uniform.far0 < particle.far0 / 4
    @test uniform.farΠ0 < particle.farΠ0 / 50
    @test uniform.far < 2 * uniform.far0
    @test uniform.Ft ≈ particle.Ft rtol = 0.1
    @test 0 < uniform.Ft / uniform.rr < 1
end

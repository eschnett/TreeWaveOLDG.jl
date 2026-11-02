# Nested boxes (INSPIRAL-PLAN.md, M3b): a static forest of cubes about the
# centre, each level's twice as wide as the next finer one's, with the
# particles' transfer on the finest level. The transfer measures its
# distances and chooses its rules in the finest element width, so on the
# finest level it has to do exactly what it does on a uniform forest at
# that spacing, and anywhere else it has to refuse; the norms and the
# initial remainder have to take each block's own spacing.

using IMEXRungeKutta: solve!
using LinearAlgebra: norm
using StaticArrays

const TWN = TreeWaveOLDG

# `levels` nested cubes about the centre of a box of side `2^levels · w₀`,
# the finest of half-width `w₀ = 2`, `roots·E` elements across each level:
# `roots = 4` gives the finest `hₑ = 1/2`, `roots = 8` gives `1/4`. One
# particle of size `ℓ = 0.01` about the centre, off the element corners,
# moving uniformly with velocity `(v, 0, 0)`, and the window `R₁ = 1`,
# `R₂ = 1.75`, as `inspiral_box` in `inspiral_tests.jl` has them.
function nested_box(roots; levels=3, v=0.2, P=3, R₂=1.75, offset=(0.11, 0.07, 0.13),
                    tables=nothing)
    w₀ = 2.0
    L = 2^levels * w₀
    forest = wave_forest(Float64, Val(3), P, 2; roots=roots, L=L,
                         refined=nested_cubes(Val(3), w₀, levels))
    return particle_problem(forest, L / 2; v, P, R₂, offset, tables)
end

# The finest cube of `nested_box` alone, as a uniform periodic box at its
# spacing, with the same coordinates, so that the same elements sit at
# the same positions bit for bit.
function finest_box(roots; levels=3, kwargs...)
    w₀ = 2.0
    c = 2^(levels - 1) * w₀
    forest = Forest{Float64}((roots, roots, roots); N=8, periodic=(true, true, true),
                             extents=ntuple(_ -> (c - w₀, c + w₀), 3))
    return particle_problem(forest, c; kwargs...)
end

function particle_problem(forest, c; v=0.2, P=3, R₂=1.75, offset=(0.11, 0.07, 0.13),
                          tables=nothing)
    wave = WaveProblem(forest, Element(Float64, P))
    centre = SVector(c, c, c)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(centre, 1.0, R₂; continuity=P + 1)
    motion = uniform_motion((centre + SVector(offset),), (SVector(v, 0.0, 0.0),))
    return InspiralProblem(wave, (m,), w, motion; tables=tables)
end

# Whether `a` and `b` agree to an ulp of the larger.
ulp_equal(a, b) = all(abs.(a .- b) .<= eps.(max.(abs.(a), abs.(b))))

@testset "wave_forest builds nested cubes level by level" begin
    # Guards the forest of nested boxes: each region has to refine the
    # leaves of the level below it and no other, so that level `k` covers
    # exactly its cube with as many elements across as every other level;
    # a region that refined every leaf in it would make a finer level
    # inside the next finer cube. `refined = true` and a single region
    # must keep their two-level meaning.
    for D in (2, 3), levels in (1, 2, 3, 4)
        w₀ = 0.5
        L = 2^levels * w₀
        regions = nested_cubes(Val(D), w₀, levels)
        @test length(regions) == levels - 1
        forest = wave_forest(Float64, Val(D), 3, 2; roots=4, L=L, refined=regions)
        @test maxlevel(forest) == levels - 1
        @test isbalanced(forest)
        for key in forest.leaves
            k = level(key)
            # The block lies inside its level's cube and outside the next
            # finer one's.
            ext = block_extent(forest, key)
            half = maximum(d -> max(abs(ext[d][1] - L / 2), abs(ext[d][2] - L / 2)), 1:D)
            @test half <= 2^(levels - 1 - k) * w₀
            inner = 2.0^(levels - 2 - k) * w₀    # the next finer cube's half-width
            k < levels - 1 && @test any(d -> ext[d][1] >= L / 2 + inner ||
                                                 ext[d][2] <= L / 2 - inner, 1:D)
            # Every level has `roots·E = 8` elements across its cube.
            @test spacing(forest, key) * 4 * 8 == 2^(levels - k) * w₀
        end
    end
    two_level(refined) = wave_forest(Float64, Val(2), 3, 2; roots=4, refined=refined)
    half = ((0.25, 0.75), (0.25, 0.75))
    @test two_level(true).leaves == two_level([half]).leaves
    @test two_level(true).leaves == two_level(half).leaves
    @test wave_forest(Float64, Val(2), 3, 2; roots=4, refined=[]).leaves ==
          wave_forest(Float64, Val(2), 3, 2; roots=4).leaves
    @test_throws "no block centre of level 1" wave_forest(
        Float64, Val(2), 3, 2; roots=4,
        refined=[((0.25, 0.75), (0.25, 0.75)), ((0.9, 0.95), (0.9, 0.95))])
    @test_throws "a vector of such tuples" wave_forest(Float64, Val(2), 3, 2;
                                                       refined=[(0.25, 0.75)])
end

@testset "On the finest level the source and the pairing are the uniform forest's" begin
    # Guards the transfer on nested boxes: on a three-level forest whose
    # finest level holds the source's ball, the projection, the pairing
    # and the initial remainder have to be those of a uniform forest at
    # the finest spacing, element for element, to an ulp. A rule chosen in
    # another level's element width, a distance measured in it, or a block
    # origin taken from the wrong level would move them by far more. The
    # far rule starts at `rfar = 3hₑ` here, so that all three rules run
    # inside the ball.
    tables = TransferTables(Float64, 3; graded=((1, 14), (2, 10)), rfar=3)
    p = nested_box(4; tables=tables)
    q = finest_box(4; tables=tables)
    @test maxlevel(p.wave.fs.forest) == 2
    u, uq = inspiral_initial_state(p), inspiral_initial_state(q)
    du, duq = zero(u), zero(uq)
    inspiral_rhs!(du, u, p, 0.1)
    inspiral_rhs!(duq, uq, q, 0.1)
    # The source alone, without the wave equation's part.
    trajs = p.motion.path(0.1)
    S = ParticleSource(p.models, p.window, trajs)
    sp, sq = zero(field_state(u, p)), zero(field_state(uq, q))
    supports = map(ParticleSupport, p.models, trajs)
    ball = (p.window.centre, p.window.R₂)
    project_source!(sp, p.wave, p.tables, S, supports; ball=ball)
    project_source!(sq, q.wave, q.tables, S, supports; ball=ball)
    @test norm(sq, Inf) > 1e-3

    fp, fq = p.wave.fs, q.wave.fs
    blocks = Dict(block_origin(fp.forest, blockkey(fp, b)) => b for b in 1:nblocks(fp))
    sa_p, sa_q = TWN.statearray(sp, fp), TWN.statearray(sq, fq)
    ua_p = TWN.statearray(field_state(u, p), fp)
    ua_q = TWN.statearray(field_state(uq, q), fq)
    for b in 1:nblocks(fq)
        a = blocks[block_origin(fq.forest, blockkey(fq, b))]
        @test level(blockkey(fp, a)) == 2
        @test ulp_equal(sa_p[:, :, :, :, a], sa_q[:, :, :, :, b])
        @test ulp_equal(ua_p[:, :, :, :, a], ua_q[:, :, :, :, b])
    end
    for s in supports, var in (1, 2)
        m, tr = p.models[1], trajs[1]
        ρ(x) = particle_density(m, x, tr.z, tr.v)
        ∇ρ(x) = TWN.ForwardDiff.gradient(ρ, x)
        @test ulp_equal(pairing(p.wave, p.tables, field_state(u, p), ρ, s; var=var),
                        pairing(q.wave, q.tables, field_state(uq, q), ρ, s; var=var))
        @test ulp_equal(pairing(p.wave, p.tables, field_state(u, p), ∇ρ, s; var=var),
                        pairing(q.wave, q.tables, field_state(uq, q), ∇ρ, s; var=var))
    end
    @test ulp_equal(particle_forces(p, u, 0.1)[1], particle_forces(q, uq, 0.1)[1])

    # On the coarser levels the initial remainder is `(1 - W)Ψ̂` at each
    # block's own nodes.
    m, tr = p.models[1], p.motion.path(0.0)[1]
    for b in (1, nblocks(fp) ÷ 3), I in ((1, 1, 1), (3, 6, 8))
        level(blockkey(fp, b)) < 2 || continue
        x = SVector(node_coordinates(fp, p.wave.element, b, I))
        exact = (1 - p.window(x)) * particle_field(m, x, tr; order=p.order)
        @test ua_p[I..., 1, b] ≈ exact rtol = 1e-14
    end
end

@testset "The transfer refuses a ball or a support that reaches a coarser level" begin
    # Guards the refusal that replaces the uniform forest's: a source's
    # ball or a particle's support that reaches a block coarser than the
    # finest would be integrated by rules chosen for the finest element
    # width, silently too coarse, so `InspiralProblem`, `project_source!`
    # and the pairing must refuse it and name the level, and must accept
    # one that stays on the finest level, which ends at `2` from the
    # centre.
    @test_throws "a block of level 1 reaches to 2.0 from the ball's centre" nested_box(
        4; R₂=2.5)
    @test_throws "reaches a block of level 1" nested_box(4; offset=(1.995, 0.0, 0.0))
    p = nested_box(4; offset=(1.985, 0.0, 0.0))
    u = inspiral_initial_state(p)
    fu = field_state(u, p)
    c = p.window.centre
    m = p.models[1]
    ρ(z) = x -> particle_density(m, x, z, zero(z))
    for (z, refused) in ((c + SVector(1.985, 0.0, 0.0), false),
                         (c + SVector(1.995, 0.0, 0.0), true),
                         (c - SVector(0.0, 1.995, 0.0), true))
        s = ParticleSupport(z, zero(z), m.ℓ)
        if refused
            @test_throws "reaches a block of level 1" pairing(p.wave, p.tables, fu, ρ(z),
                                                              s)
            @test_throws "reaches a block of level 1" project_source!(
                zero(fu), p.wave, p.tables, x -> 1.0, (s,); ball=(c, 1.0))
        else
            @test pairing(p.wave, p.tables, fu, ρ(z), s) isa Float64
            @test project_source!(zero(fu), p.wave, p.tables, x -> 1.0, (s,);
                                  ball=(c, 1.0)) isa AbstractVector
        end
    end
    s = ParticleSupport(c, zero(c), m.ℓ)
    @test_throws "a block of level 1 reaches" project_source!(
        zero(fu), p.wave, p.tables, x -> 1.0, (s,); ball=(c, 2.01))
    @test project_source!(zero(fu), p.wave, p.tables, x -> 1.0, (s,);
                          ball=(c, 1.99)) isa AbstractVector
end

@testset "Inside the finest level the ball norm is the uniform forest's" begin
    # Guards `ball_norm` on a refined forest: each node's weight has to be
    # its own block's, and its position its own block's, so a ball inside
    # the finest level sees exactly the uniform forest's nodes and weights.
    p, q = nested_box(4), finest_box(4)
    u, uq = inspiral_initial_state(p), inspiral_initial_state(q)
    c = p.window.centre
    for r in (0.6, 1.9), vars in (1:2, 1:1)
        @test ball_norm(p, u, c, r; vars) ≈ ball_norm(q, uq, c, r; vars) rtol = 1e-14
        @test ball_norm(p, u, c, r; vars, p=Inf) == ball_norm(q, uq, c, r; vars, p=Inf)
    end
end

@testset "On nested boxes a moving particle keeps its remainder, and no force" begin
    # Guards the inspiral on nested boxes as a whole: the coarse levels,
    # their coarse-fine faces and the finest level's transfer together
    # have to leave the exact remainder `(1 - W)Φ_C(z + vt)` of a
    # uniformly moving particle in the window's ball, to an error that
    # falls with the finest `hₑ` (every level's spacing halved together),
    # and the remainder's force near zero, as on M3's uniform box. The
    # rates are measured on this coarse forest, as in `inspiral_tests.jl`
    # ("Nested boxes" in `CODE.md`).
    T = 0.25
    results = map((4, 8)) do roots
        p = nested_box(roots; v=0.2)
        u0 = inspiral_initial_state(p)
        dt = oldg_timestep(p.wave; cfl=1 // 2)
        integ = solve!(inspiral_integrator(p, u0, 0.0, T; dt=dt))
        ue = inspiral_initial_state(p; t=integ.t)
        c = p.window.centre
        (; err=ball_norm(p, integ.u .- ue, c, p.window.R₂),
         field=ball_norm(p, ue, c, p.window.R₂),
         force=norm(particle_forces(p, integ.u, integ.t)[1]),
         z=particle_state(integ.u, p)[1][1] - p.motion.path(integ.t)[1].z)
    end
    coarse, fine = results
    @test coarse.err < coarse.field
    @test coarse.err / fine.err > 3
    @test coarse.force < 1e-4
    @test coarse.force / fine.force > 100
    @test all(r -> norm(r.z, Inf) < 1e-13, results)
end

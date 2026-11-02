# A particle in uniform motion, INSPIRAL-PLAN.md's M3 and M3b, at the
# thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/uniform_motion.jl [--hdiv=8] [--L=6] \
#         [--levels=1] [--w0=L/2] [--tend=0.5] [--every=0.125] [--v=0.2] \
#         [--order=3] [--cfl=0.5] [--cost] [--reps=3]
#
# Like `bin/benchmark.jl`, this runs against the package environment;
# `bin/symmetry/uniform_motion.sh` runs M3's series on a Symmetry node, and
# `bin/symmetry/nested_motion.sh` M3b's.
#
# With `--levels=1`, the default, the box is M3's: uniform, periodic, of
# side `L`. With `--levels=n > 1` it is M3b's nested boxes
# (`nested_cubes`): `n` cubes about the centre, the finest of half-width
# `w0` (by default `L/2`, so that the finest level is M3's box) at
# `hₑ = 1/hdiv`, each coarser one twice as wide at twice the element
# width, inside a periodic box of side `2^n w0`. Every level then has
# `2 w0 hdiv` elements across, which has to be a multiple of `4E` for the
# cubes' faces to fall on block faces; `E` is 4 where that fits, else 2.
# The window's ball has to lie on the finest level, which the problem
# checks when it is built.
#
# One particle of the first run of INSPIRAL.md ("Parameters for a first
# run"), `Q² = 0.005`, `ℓ = 10⁻³`, `M = 1`, starts where the binary's first
# particle would, `(0.5, 0, 0)` from the centre of a periodic box of side
# `L` (plus `(1.23, 0.71, 0.93)·10⁻²`, off the element corners at every
# `hdiv`), and moves along `y` at `v`, so that it crosses element faces.
# The window is the first run's, `R₁ = 1.25`, `R₂ = 2.5`, of continuity
# `P + 1`, about the box's centre; `P = 3`, `hₑ = 1/hdiv`, the near field
# carried to `order`. The remainder starts as `(1 - W)Ψ̂`, which for
# uniform motion is the exact remainder `(1 - W)Φ_C(z + vt)` at every `t`
# in an unbounded domain. The box is not unbounded: the initial data are
# not periodic, and their error enters at the faces and travels inward.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads hdiv L levels w0 v t quantity value
#
# with `L` the outermost box's side. At `t = 0`: `nodes`, and on nested
# boxes `nodes_level<k>` for each level `k` (0 the root), `steps`, `dt`.
# Every `every` in time, the error against the exact remainder,
# `(Φ_r, Π_r)` together, in quadrature-weighted L2 over the ball of radius
# `R₂` about the centre (`err_R2`, in M3's box clean of the faces until
# about `(L/2 - R₂)/2`), of radius `R₁` (`err_R1`), and of radius 1/2
# about the particle (`err_z`); the same of `Φ_r` alone (`errphi_R2`),
# the L∞ error in the `R₂` ball (`errinf_R2`), the exact remainder's L2
# in it (`field_R2`), the remainder's force on the particle (`force`, its
# norm, and `Fx`, `Fy`, `Fz`), and how far the particle's state is from
# its path (`dz`).
#
# With `--cost`, at the end, in seconds per call, the least of `reps`
# after a warm-up: `rhs`, one `inspiral_rhs!`; its parts `wave`
# (`wave_rhs!`) and `source` (`project_source!`, `ParticleSource`
# included); `truncate`; `forces`, one `particle_forces`; and the step,
# `step`. And bytes per node: `bytes_state` (the state vector),
# `bytes_wave` (the wave problem's working arrays), `bytes_scratch` (the
# projection's scratch), `bytes_integrator` (RK4's scratch, five arrays
# like the state).
#
# "Uniform motion" in `CODE.md`'s "Measured results" has the numbers.

using TreeWaveOLDG
using TreeWaveOLDG: SVector, step!, statelength, statearray, field_state, level,
                    maxlevel

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

# The forest: M3's uniform box of side `L` for one level, else `levels`
# nested cubes about the centre, the finest of half-width `w0`, each with
# `2 w0 hdiv` elements across. Returns the forest and the outer side.
function box(hdiv, L, levels, w0)
    P = 3
    hₑ = 1 / hdiv
    if levels == 1
        # Blocks of `E` elements, as many as fit the box.
        E = first(e for e in (4, 2) if isinteger(L / (e * hₑ)))
        roots = round(Int, L / (E * hₑ))
        return wave_forest(Float64, Val(3), P, E; roots=roots, L=L), L
    end
    across = 2w0 * hdiv
    isinteger(across) || error("2 w0 hdiv = $across elements across a level is not whole")
    fits = [e for e in (4, 2) if round(Int, across) % (4e) == 0]
    isempty(fits) && error(
        "2 w0 hdiv = $across elements across a level is not a multiple of 4E for E = 2 " *
        "or 4, so the nested cubes' faces would not fall on block faces; choose another " *
        "w0 or hdiv")
    E = first(fits)
    Lout = 2^levels * w0
    forest = wave_forest(Float64, Val(3), P, E; roots=round(Int, across) ÷ E, L=Lout,
                         refined=nested_cubes(Val(3), w0, levels))
    return forest, Lout
end

function problem(hdiv, L, levels, w0, v, order)
    P = 3
    forest, L = box(hdiv, L, levels, w0)
    wave = WaveProblem(forest, Element(Float64, P))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    z = c + SVector(0.5, 0.0, 0.0) + SVector(1.23, 0.71, 0.93) * 1e-2
    motion = uniform_motion((z,), (SVector(0.0, v, 0.0),))
    return InspiralProblem(wave, (m,), w, motion; order=order), L
end

function emit(row...)
    println(join((Threads.nthreads(), row...), '\t'))
    flush(stdout)
    return nothing
end

function observe(p, u, t, tag)
    ue = inspiral_initial_state(p; t=t)
    e = u .- ue
    c = p.window.centre
    z = p.motion.path(t)[1].z
    F = particle_forces(p, u, t)[1]
    (zs, _), = particle_state(u, p)
    for (name, value) in ("err_R2" => ball_norm(p, e, c, p.window.R₂),
                          "err_R1" => ball_norm(p, e, c, p.window.R₁),
                          "err_z" => ball_norm(p, e, z, 0.5),
                          "errphi_R2" => ball_norm(p, e, c, p.window.R₂; vars=1),
                          "errinf_R2" => ball_norm(p, e, c, p.window.R₂; p=Inf),
                          "field_R2" => ball_norm(p, ue, c, p.window.R₂),
                          "force" => sqrt(sum(abs2, F)),
                          "Fx" => F[1], "Fy" => F[2], "Fz" => F[3],
                          "dz" => maximum(abs, zs - z))
        emit(tag..., t, name, value)
    end
    return nothing
end

function main()
    hdiv = parse(Int, option("hdiv", "8"))
    L = parse(Float64, option("L", "6"))
    levels = parse(Int, option("levels", "1"))
    w0 = parse(Float64, option("w0", string(L / 2)))
    tend = parse(Float64, option("tend", "0.5"))
    every = parse(Float64, option("every", "0.125"))
    v = parse(Float64, option("v", "0.2"))
    order = parse(Int, option("order", "3"))
    cfl = parse(Float64, option("cfl", "0.5"))
    reps = parse(Int, option("reps", "3"))
    p, Lout = problem(hdiv, L, levels, w0, v, order)
    tag = (hdiv, Lout, levels, levels == 1 ? Lout / 2 : w0, v)
    println(join(("threads", "hdiv", "L", "levels", "w0", "v", "t", "quantity", "value"),
                 '\t'))

    u0 = inspiral_initial_state(p)
    dt = oldg_timestep(p.wave; cfl=cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt=dt)
    nodes = statelength(p.wave.fs) ÷ 2
    emit(tag..., 0.0, "nodes", nodes)
    forest = p.wave.fs.forest
    if levels > 1
        for k in 0:maxlevel(forest)
            emit(tag..., 0.0, "nodes_level$k",
                 count(key -> level(key) == k, forest.leaves) * forest.N^3)
        end
    end
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    k = max(1, round(Int, every / integ.dt))
    observe(p, integ.u, integ.t, tag)
    while integ.nstep < integ.nsteps
        step!(integ)
        (integ.nstep % k == 0 || integ.nstep == integ.nsteps) &&
            observe(p, integ.u, integ.t, tag)
    end

    "--cost" in ARGS || return nothing
    u = integ.u
    du = similar(u)
    trajs = p.motion.path(integ.t)
    S = ParticleSource(p.models, p.window, trajs; order=p.order)
    supports = map(ParticleSupport, p.models, trajs)
    fu, fdu = field_state(u, p), field_state(du, p)
    work(fs) = sizeof(fs.work)
    bytes_wave = work(p.wave.fs) + work(p.wave.ufs) + sum(work, p.wave.grads)
    t = integ.t
    # `reps + 1` more steps, on the run's scratch.
    stepper = inspiral_integrator(p, copy(u), t, t + (reps + 1) * integ.dt; dt=integ.dt,
                                  reuse=integ)
    for (name, value) in (
            "rhs" => best(() -> inspiral_rhs!(du, u, p, t), reps),
            "wave" => best(() -> wave_rhs!(fdu, fu, p.wave, t), reps),
            "source" => best(() -> project_source!(
                fdu, p.wave, p.tables,
                ParticleSource(p.models, p.window, trajs; order=p.order), supports;
                ball=(p.window.centre, p.window.R₂), rnear=p.rnear,
                scratch=p.scratch), reps),
            "truncate" => best(() -> inspiral_stage!(u, integ, p, t), reps),
            "forces" => best(() -> particle_forces(p, u, t), reps),
            "step" => best(() -> step!(stepper), reps),
            "bytes_state" => sizeof(u) / nodes,
            "bytes_wave" => bytes_wave / nodes,
            "bytes_scratch" => sizeof(p.scratch) / nodes,
            "bytes_integrator" => 5 * sizeof(u) / nodes)
        emit(tag..., t, name, value)
    end
    return nothing
end

main()

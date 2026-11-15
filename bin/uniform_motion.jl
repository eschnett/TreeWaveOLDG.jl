# A particle in uniform motion, INSPIRAL-PLAN.md's M3 and M3b, at the
# thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/uniform_motion.jl [--hdiv=8] [--L=6] \
#         [--levels=1] [--w0=L/2] [--tend=0.5] [--every=0.125] [--v=0.2] \
#         [--order=3] [--cfl=0.5] [--cost] [--reps=3] \
#         [--ball] [--R1=50] [--hshell] [--backend=cpu|cuda]
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
# With `--ball` the outermost of the `levels` cubes is instead the cube of
# the seven-patch ball (`bin/ball.jl`, INSPIRAL-PLAN.md's M8c), of
# half-width `2^(levels-1) w0`, with shells to a Sommerfeld sphere at
# `R1`, of radial elements about `hshell` wide (twice the cube's root
# element width by default), and the centre is the origin: the nested
# boxes' periodic box replaced by a radiative sphere. `L` in the rows is
# then the sphere's diameter `2R1`.
#
# With `--backend=cuda` (or `metal`, which has no `Float64` and so cannot
# run the particles) the field and the particles run on that device
# (`bin/backend.jl`), from an environment that adds the device package
# (`bin/symmetry/gpu_scans.sh` builds one); the initial and exact
# remainders are filled on the device, and the norms are
# `ball_norm`'s on the device. `--cost` waits for the device after each
# call.
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
# its path (`dz`), and the mean of `Π_r` in the `R₂` ball (`pimean_R2`),
# which a net charge drives at `Q/(γV)` in a periodic box, and the mean of
# its error (`errpimean_R2`). In the ball also, at `t = 0`, `Lc`, the
# shells' radial roots `m` and element width `hshell`, and their nodes
# (`nodes_shells`).
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
                    maxlevel, patch

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "ball.jl"))

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end


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

function problem(hdiv, L, levels, w0, v, order; ball=nothing, backend=CPU())
    P = 3
    if ball === nothing
        forest, L = box(hdiv, L, levels, w0)
        wave = WaveProblem(forest, Element(Float64, P); backend=backend)
        c = SVector(L / 2, L / 2, L / 2)
    else
        wave = ball_wave(hdiv, levels, w0; ball..., backend=backend)
        L = 2ball.R1
        c = SVector(0.0, 0.0, 0.0)
    end
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    z = c + SVector(0.5, 0.0, 0.0) + SVector(1.23, 0.71, 0.93) * 1e-2
    motion = uniform_motion((z,), (SVector(0.0, v, 0.0),))
    return InspiralProblem(wave, (m,), w, motion; order=order), L
end

# Seconds per call of `f()`, the least of `reps` after a warm-up, each
# waited for on the device (`best`, `bin/backend.jl`).
timed(backend, f, reps) = best(f, backend, reps)

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
                          "dz" => maximum(abs, zs - z),
                          "pimean_R2" => ball_mean(p, u, c, p.window.R₂; vars=2),
                          "errpimean_R2" => ball_mean(p, e, c, p.window.R₂; vars=2))
        emit(tag..., t, name, value)
    end
    return nothing
end

# The particles need `Float64`, so `--type` is not taken.
function main()
    withbackend((backend, _) -> evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function evolve(backend)
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
    ball = nothing
    if "--ball" in ARGS
        hs = option("hshell", "")
        ball = (; R1=parse(Float64, option("R1", "50")),
                hshell=isempty(hs) ? nothing : parse(Float64, hs))
    end
    p, Lout = problem(hdiv, L, levels, w0, v, order; ball, backend)
    tag = (hdiv, Lout, levels, levels == 1 && ball === nothing ? Lout / 2 : w0, v)
    println(join(("threads", "hdiv", "L", "levels", "w0", "v", "t", "quantity", "value"),
                 '\t'))

    u0 = inspiral_initial_state(p)
    dt = oldg_timestep(p.wave; cfl=cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt=dt)
    nodes = statelength(p.wave.fs) ÷ 2
    emit(tag..., 0.0, "nodes", nodes)
    forest = p.wave.fs.forest
    incube(key) = ball === nothing || patch(forest, key) == 1
    if levels > 1
        for k in 0:maxlevel(forest)
            emit(tag..., 0.0, "nodes_level$k",
                 count(key -> incube(key) && level(key) == k, forest.leaves) * forest.N^3)
        end
    end
    if ball !== nothing
        (; Lc, m, hshell) = ball_layout(hdiv, levels, w0; ball...)
        emit(tag..., 0.0, "Lc", Lc)
        emit(tag..., 0.0, "m", m)
        emit(tag..., 0.0, "hshell", hshell)
        emit(tag..., 0.0, "nodes_shells",
             count(key -> !incube(key), forest.leaves) * forest.N^3)
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
    # On a full geometry (`--ball`) the gradient is `cartgrad`, and `grads`
    # is empty.
    bytes_wave = work(p.wave.fs) + work(p.wave.ufs) + sum(work, p.wave.grads; init=0) +
                 (p.wave.cartgrad === nothing ? 0 : work(p.wave.cartgrad))
    t = integ.t
    # `reps + 1` more steps, on the run's scratch.
    stepper = inspiral_integrator(p, copy(u), t, t + (reps + 1) * integ.dt; dt=integ.dt,
                                  reuse=integ)
    for (name, value) in (
            "rhs" => timed(backend, () -> inspiral_rhs!(du, u, p, t), reps),
            "wave" => timed(backend, () -> wave_rhs!(fdu, fu, p.wave, t), reps),
            "source" => timed(backend, () -> project_source!(
                fdu, p.wave, p.tables,
                ParticleSource(p.models, p.window, trajs; order=p.order), supports;
                ball=(p.window.centre, p.window.R₂), rnear=p.rnear,
                scratch=p.scratch, sourceball=p.sourceball), reps),
            "truncate" => timed(backend, () -> inspiral_stage!(u, integ, p, t), reps),
            "forces" => timed(backend, () -> particle_forces(p, u, t), reps),
            "step" => timed(backend, () -> step!(stepper), reps),
            "bytes_state" => sizeof(u) / nodes,
            "bytes_wave" => bytes_wave / nodes,
            "bytes_scratch" => sizeof(p.scratch) / nodes,
            "bytes_integrator" => 5 * sizeof(u) / nodes)
        emit(tag..., t, name, value)
    end
    return nothing
end

main()

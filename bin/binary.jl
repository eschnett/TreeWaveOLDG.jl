# The Newtonian binary of INSPIRAL.md's "Parameters for a first run",
# INSPIRAL-PLAN.md's M6b ("A short binary run"), at the thread count Julia
# was started with:
#
#     julia -t N,0 --project=. bin/binary.jl [--hdiv=4] [--levels=3] \
#         [--w0=4] [--R1=50] [--hshell=1] [--order=3] [--tend=32] \
#         [--cfl=0.5] [--v=0.2] [--T=30] [--ramp=10] [--n=4] [--onset=2π/Ω] \
#         [--n2=4] [--rx1=100] [--rx2=110] [--every=1] [--jevery=0.125] \
#         [--source-tol=0.1] [--flux=FILE] [--fradii=4,6,8,12] [--prescribed] \
#         [--ell=1e-3] [--source-addot=solved] [--window=1.25,2.5] [--mpi] \
#         [--checkpoint=DIR] [--restart-from=DIR] [--walltime=SECONDS] \
#         [--ckevery=7200] [--stop-at-step=N]
#
# Two particles of `M = 1`, `Q = ±√0.005`, `ℓ = 10⁻³`, at separation 1, each
# at speed `v`, on `binary_circular_motion`'s circle about the centre
# `o = (1.23, 0.71, 0.93)·10⁻²`, off the element faces of the orbit's
# plane (`bin/prescribed.jl`'s `OFFSET`), held there by Newton's force
# with the `G` that makes the predictor's force centripetal
# (`newtonian_binary`). The data are the retarded field, out to `rx1`,
# of a past at rest before `−T − ramp` and spun up with the smoothstep of
# order `n` to the orbit at `−T` (`CODE.md`, "Assumed pasts"), and the
# force takes over from that past over `onset`, one orbit by default, with
# `n2` (`Dynamic`). The layout is `bin/prescribed.jl`'s `--ball`
# (`bin/ball.jl`): `levels` nested cubes, the finest of half-width `w0` at
# `hₑ = 1/hdiv`, in the seven-patch ball's cube, shells of radial width
# about `hshell` to a Sommerfeld sphere at `R1`; the window `R₁ = 1.25`,
# `R₂ = 2.5`, about `o` (`--window=R₁,R₂`). `--prescribed` runs the past's
# circle instead, on the same grid and steps.
#
# Two options are experiments on the solved `ä`'s jumps where a support
# straddles an element face (`CODE.md`, "The binary" under "Measured
# results"): `--ell` sets the particles' size `ℓ`, and
# `--source-addot=predicted` writes the source with the predictor's `ä` in
# place of the solved one (`Dynamic`'s `source_addot`), which the rows
# still record.
#
# Every `every` steps it prints one row per quantity, `bin/dynamic.jl`'s
# columns:
#
# - the orbit (`orbit_diagnostics`): `d`, `ddot`, `phase`, `omega`,
#   `energy`, the centre of mass's distance from `o` (`com`) and the
#   momentum's norm (`momentum`);
# - per particle `k`, the remainder's force along the inward radius
#   (`Fr<k>`, toward the centre of mass), the velocity (`Ft<k>`) and their
#   normal (`Fn<k>`);
# - per particle, the jets as solved at the step's last stage, at `t`,
#   `a`, `ȧ`, `ä` (`a<k>x`, …,
#   `ad<k>x`, …, `add<k>x`, …), and the predictor's `a⃛` and `a⃜`
#   (`a3<k>x`, …, `a4<k>x`, …), the instabilities in the higher
#   derivatives that self-force schemes have reported showing first there;
#   and `jerkgap<k>`, `|(ä(t) − ä(t − Δt))/Δt − (a⃛(t) + a⃛(t − Δt))/2|`
#   over `|a⃛(t)|`, the solved `ä`'s rate against the predictor's `a⃛`
#   across the step;
# - what the step's last solve did: `iterations`, `projections`,
#   `residual`, and `dmh<k>`, `‖J_ää − MγI‖/M` of particle `k` with the
#   Jacobian it used, which may have been kept; every `jevery` in time the
#   step's first solve finds it afresh (`renew_jacobian!`), and that
#   step's row is `dmhfresh<k>`, with `dmhcross`, the coupling of the two
#   `ä` blocks, and `dmhgrid<k>`, `‖J_ää − mI‖/M` with the mass the
#   equation has besides the grid's, `m = σ(M + Q_b)γ + (1 − σ)M` at the
#   hand-over's `σ`, `Q_b` the other charge's potential (`other_force`):
#   the grid's self-inertia, which `dmhfresh<k>` holds beside `γQ_b`
#   (`CODE.md`, "The binary" under "Measured results");
# - the wall-clock seconds since the start.
#
# - what the step cost, in wall-clock seconds: `step_seconds` in all, and
#   of it `solve_seconds`, the jets' solves at the step's stages
#   (`solve_seconds`), split into `laplacian_seconds`, `projection_seconds`,
#   `residual_seconds` and `jacobian_seconds` (the last a part of the
#   residuals'); the rest is the field's right-hand side and the source.
#   `report_seconds` is the last report's, these rows' own cost.
#
# At `t = 0` also `G`, `K`, the leading-order `ddot0` (`dipole_ddot`),
# `nodes`, `steps`, `dt`, `ranks` and `setup_seconds`.
#
# With `--mpi` it runs over MPI under `mpiexec`, in an environment that
# has MPI.jl (SimLaunch's `mpi` kind): the ball's forest takes
# `MPI.COMM_WORLD`, and rank 0 alone prints and writes the flux
# ("Running distributed" in `CODE.md`). Every rank solves the jets, on
# the same numbers, so the run is the serial one to the bit. With `--flux=FILE` the flux
# through spheres in the cube about `o` of radii `--fradii`, every
# `--fluxevery` (`bin/waves_io.jl`), with the retarded field's beside it
# only with `--prescribed`.

include(joinpath(@__DIR__, "prescribed.jl"))
using TreeWaveOLDG: lorentz_factor, other_force, rank_reduce, resume_integrator!,
                    solve_state, solve_state!
using LinearAlgebra: I, cross, opnorm
using Serialization: serialize, deserialize

const PROCESS_START = time()
const MPI_RUN = "--mpi" in ARGS
MPI_RUN && @eval using MPI

# `MPI.COMM_WORLD`, initialized, under `--mpi`; otherwise a serial run.
function binary_comm()
    MPI_RUN || return nothing
    MPI.Init()
    QUIET[] = MPI.Comm_rank(MPI.COMM_WORLD) != 0
    return MPI.COMM_WORLD
end

function binary_main()
    withbackend((backend, _) -> binary_evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function binary_evolve(backend)
    hdiv = parse(Int, option("hdiv", "4"))
    levels = parse(Int, option("levels", "3"))
    w0 = parse(Float64, option("w0", "4"))
    R1 = parse(Float64, option("R1", "50"))
    hs = option("hshell", "1")
    order = parse(Int, option("order", "3"))
    tend = parse(Float64, option("tend", "32"))
    cfl = parse(Float64, option("cfl", "0.5"))
    v = parse(Float64, option("v", "0.2"))
    d0 = 1.0
    Ω = 2v / d0
    T = parse(Float64, option("T", "30"))
    ramp = parse(Float64, option("ramp", "10"))
    n = parse(Int, option("n", "4"))
    onset = parse(Float64, option("onset", string(2π / Ω)))
    n2 = parse(Int, option("n2", "4"))
    rx1 = parse(Float64, option("rx1", "100"))
    rx2 = parse(Float64, option("rx2", "110"))
    every = parse(Int, option("every", "1"))
    jevery = parse(Float64, option("jevery", "0.125"))
    source_tol = parse(Float64, option("source-tol", "0.1"))
    prescribed = "--prescribed" in ARGS
    ℓ = parse(Float64, option("ell", "1e-3"))
    source_addot = Symbol(option("source-addot", "solved"))
    wR₁, wR₂ = parse.(Float64, split(option("window", "1.25,2.5"), ','))
    comm = binary_comm()
    start = time()
    P = 3
    wave = ball_wave(hdiv, levels, w0; R1, hshell=parse(Float64, hs), P, backend, comm)
    o = OFFSET
    q = sqrt(0.005)
    ms = (ParticleModel(; Q=q, ℓ, M=1.0), ParticleModel(; Q=-q, ℓ, M=1.0))
    f, past = newtonian_binary(ms, d0, v; centre=o, past=T, ramp, n)
    motion = prescribed ? past :
             Dynamic((NoForce(), NoForce()), past; interaction=f, onset, n2, source_tol,
                     source_addot)
    p = InspiralProblem(wave, ms, Window(o, wR₁, wR₂; continuity=P + 1), motion; order)
    retarded = Window(o, rx1, rx2; continuity=4)
    ck = checkpoint_options(comm)
    resumed = read_checkpoint(ck, p.wave.fs)
    u0 = resumed !== nothing ? resumed.u :
         prescribed ? inspiral_initial_state(p; retarded) :
         dynamic_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt)
    if resumed !== nothing
        resume_integrator!(integ, resumed.u, resumed.nstep)
        prescribed || solve_state!(motion, resumed.solve)
    end
    tag = (prescribed ? "circle" : "binary", hdiv, levels, w0, order)
    QUIET[] || println(join(("threads", "motion", "hdiv", "levels", "w0", "order", "t",
                             "quantity", "value"), '\t'))
    emit(tag..., 0.0, "G", f.G)
    emit(tag..., 0.0, "K", binding_constant(ms, f))
    emit(tag..., 0.0, "ddot0", dipole_ddot(ms, f, d0))
    emit(tag..., 0.0, "Omega", Ω)
    emit(tag..., 0.0, "onset", onset)
    emit(tag..., 0.0, "ell", ℓ)
    emit(tag..., 0.0, "windowR1", wR₁)
    emit(tag..., 0.0, "windowR2", wR₂)
    emit(tag..., 0.0, "source_addot_predicted", source_addot === :predicted ? 1 : 0)
    emit(tag..., 0.0, "nodes", rank_reduce(+, p.wave.fs, statelength(p.wave.fs) ÷ 2))
    emit(tag..., 0.0, "ranks", comm === nothing ? 1 : MPI.Comm_size(comm))
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    emit(tag..., 0.0, "setup_seconds", time() - start)
    resumed === nothing || emit(tag..., integ.t, "resumed_step", integ.nstep)
    fluxes = flux_writer(p, o, prescribed ? retarded : nothing)
    kf = max(1, round(Int, parse(Float64, option("fluxevery", "0.0625")) / integ.dt))
    resumed === nothing &&
        write_flux(fluxes, p, integ.u, integ.t, prescribed ? retarded : nothing)
    # The last report's time and jets, and the time of the next fresh
    # Jacobian: a checkpoint carries them.
    previous = Ref{Any}(resumed === nothing ? nothing : resumed.previous)
    nextfresh = Ref(resumed === nothing ? 0.0 : resumed.nextfresh)
    renewed = Ref(false)                # the step's Jacobian was found afresh
    spent = prescribed ? nothing : solve_seconds(motion)
    step_seconds = Ref(0.0)
    report_seconds = Ref(0.0)
    function report(u, t)
        t0 = time()
        zv = particle_state(u, p)
        zs, vs = map(first, zv), map(last, zv)
        od = orbit_diagnostics(ms, f, zs, vs)
        for (name, value) in (("d", od.d), ("ddot", od.ddot), ("phase", od.phase),
                              ("omega", od.omega), ("energy", od.energy),
                              ("com", norm(od.centre - o)), ("momentum", norm(od.momentum)))
            emit(tag..., t, name, value)
        end
        Fg = particle_forces(p, u, t)
        for k in 1:2
            ρ = (od.centre - zs[k]) / norm(od.centre - zs[k])
            τ = vs[k] / norm(vs[k])
            emit(tag..., t, "Fr$k", dot(Fg[k], ρ))
            emit(tag..., t, "Ft$k", dot(Fg[k], τ))
            emit(tag..., t, "Fn$k", dot(Fg[k], cross(ρ, τ)))
        end
        jets = prescribed ? past.path(t) :
               (motion.scratch.last === nothing ? past.path(t) : motion.scratch.last.trajs)
        for k in 1:2, (name, s) in (("a", :a), ("ad", :ȧ), ("add", :ä), ("a3", :a⃛),
                                     ("a4", :a⃜))
            x = getfield(jets[k], s)
            for (i, xyz) in enumerate("xyz")
                emit(tag..., t, "$name$k$xyz", x[i])
            end
        end
        if previous[] !== nothing
            t₋, jets₋ = previous[]
            for k in 1:2
                rate = (jets[k].ä - jets₋[k].ä) / (t - t₋)
                gap = norm(rate - (jets[k].a⃛ + jets₋[k].a⃛) / 2) / norm(jets[k].a⃛)
                emit(tag..., t, "jerkgap$k", gap)
            end
        end
        previous[] = (t, jets)
        if !prescribed && motion.scratch.last !== nothing
            l = motion.scratch.last
            emit(tag..., t, "iterations", l.iterations)
            emit(tag..., t, "projections", l.projections)
            emit(tag..., t, "residual", l.residual)
            # A Jacobian found afresh at the step's first stage is kept
            # through its others, as the chord iteration converges with it.
            J = l.jacobian
            fresh = renewed[]
            for (k, blk) in ((1, 7:9), (2, 16:18))
                γ = lorentz_factor(vs[k])
                dmh = opnorm(J[blk, blk] - ms[k].M * γ * I) / ms[k].M
                emit(tag..., t, fresh ? "dmhfresh$k" : "dmh$k", dmh)
            end
            fresh && emit(tag..., t, "dmhcross", opnorm(J[7:9, 16:18]) / ms[1].M)
            # The mass the equation of motion has besides the grid's:
            # `M` and the other charge's potential `Q_b`, through the
            # hand-over's blend `σ`.
            if fresh
                σ = l.σ
                for (k, b, blk) in ((1, 2, 7:9), (2, 1, 16:18))
                    γ = lorentz_factor(vs[k])
                    Qb = other_force(ms[k], l.trajs[k], ms[b], l.trajs[b], motion.rule;
                                     order)[2]
                    mass = σ * (ms[k].M + Qb) * γ + (1 - σ) * ms[k].M
                    emit(tag..., t, "dmhgrid$k", opnorm(J[blk, blk] - mass * I) / ms[k].M)
                end
            end
        end
        emit(tag..., t, "step_seconds", step_seconds[])
        if !prescribed
            now = solve_seconds(motion)
            for k in keys(now)
                emit(tag..., t, "$(k)_seconds", now[k] - spent[k])
            end
            spent = now
        end
        emit(tag..., t, "report_seconds", report_seconds[])
        emit(tag..., t, "seconds", time() - start)
        report_seconds[] = time() - t0
        return nothing
    end
    resumed === nothing && report(integ.u, integ.t)
    loop = time()
    lastck = time()
    steps = 0
    while integ.nstep < integ.nsteps
        renewed[] = !prescribed && integ.t >= nextfresh[]
        if renewed[]
            renew_jacobian!(motion)
            nextfresh[] += jevery
        end
        t0 = time()
        step!(integ)
        step_seconds[] = time() - t0
        (integ.nstep % kf == 0 || integ.nstep == integ.nsteps) &&
            write_flux(fluxes, p, integ.u, integ.t, prescribed ? retarded : nothing)
        if integ.nstep % every == 0 || integ.nstep == integ.nsteps
            report(integ.u, integ.t)
        end
        steps += 1
        ck.dir === nothing && continue
        integ.nstep == integ.nsteps && break
        # Since the process started, with two steps' margin beyond
        # SimLaunch's for the checkpoint; the ranks agree, so that they
        # stop and write together.
        now = time()
        stop, due = rank_reduce((a, b) -> (a[1] | b[1], a[2] | b[2]), p.wave.fs,
                                (now - PROCESS_START + 2 * (now - loop) / steps >
                                 ck.walltime, now - lastck > ck.every))
        stop |= integ.nstep == ck.stopat
        if stop || due
            write_checkpoint(ck, integ, prescribed ? nothing : solve_state(motion),
                             previous[], nextfresh[])
            lastck = time()
            emit(tag..., integ.t, "checkpoint_step", integ.nstep)
        end
        if stop
            fluxes === nothing || fluxes.io === devnull || close(fluxes.io)
            exit(STOPPED)
        end
    end
    fluxes === nothing || fluxes.io === devnull || close(fluxes.io)
    return nothing
end

# The checkpoint, SimLaunch's restart contract (`SPEC.md`, "Restarts"):
# `--checkpoint=DIR` writes one into `DIR` every `--ckevery` seconds of
# wall clock (7200), and before `--walltime` seconds since the start have
# passed since the script started, after which the run exits with
# `STOPPED`, as it does after step `--stop-at-step`, for a test of the
# resumed run (`jobs/binary_restart.toml`); `--restart-from=DIR`
# resumes from the newest whole one in `DIR`, read only. A checkpoint is
# one file per rank, `step<n>-rank<r>-of-<R>.jls`, each written under a
# temporary name and renamed when whole; it holds the rank's state
# vector, the step, the solve's state (`solve_state`) and the driver's
# own, so that the resumed run is the uninterrupted one to the bit (the
# test "A binary resumed from a checkpoint …" in `test/dynamic_tests.jl`).
# A resumed run needs the same options and rank count, and writes its
# rows and its flux from the step after the checkpoint on; the initial
# data are not computed again.
const STOPPED = 3

function checkpoint_options(comm)
    rank, nranks = comm === nothing ? (0, 1) : (MPI.Comm_rank(comm), MPI.Comm_size(comm))
    dir, from = option("checkpoint", ""), option("restart-from", "")
    return (; dir=isempty(dir) ? nothing : dir, from=isempty(from) ? nothing : from,
            walltime=parse(Float64, option("walltime", "Inf")),
            every=parse(Float64, option("ckevery", "7200")),
            stopat=parse(Int, option("stop-at-step", "-1")), rank, nranks)
end

checkpoint_name(ck, nstep) = "step$(nstep)-rank$(ck.rank)-of-$(ck.nranks).jls"

function write_checkpoint(ck, integ, solve, previous, nextfresh)
    mkpath(ck.dir)
    file = joinpath(ck.dir, checkpoint_name(ck, integ.nstep))
    tmp = file * ".tmp"
    serialize(tmp, (; u=copy(integ.u), nstep=integ.nstep, t=integ.t, solve, previous,
                    nextfresh, args=checkpoint_args()))
    mv(tmp, file; force=true)
    # This rank's older checkpoints, but the one before this: the newest
    # may not be whole on another rank if the run is killed now.
    keep = previous_checkpoint(ck.dir, ck, integ.nstep)
    for (n, f) in checkpoint_steps(ck.dir, ck)
        n < keep && rm(joinpath(ck.dir, f); force=true)
    end
    return nothing
end

# This rank's checkpoint files in `dir`, by step.
function checkpoint_steps(dir, ck)
    isdir(dir) || return Tuple{Int,String}[]
    suffix = "-rank$(ck.rank)-of-$(ck.nranks).jls"
    return [(parse(Int, chopsuffix(chopprefix(f, "step"), suffix)), f)
            for f in readdir(dir) if startswith(f, "step") && endswith(f, suffix)]
end

previous_checkpoint(dir, ck, nstep) =
    maximum((n for (n, _) in checkpoint_steps(dir, ck) if n < nstep); init=-1)

# The options that must be the same in a resumed run: all but those of the
# checkpoint itself and the flux's file.
checkpoint_args() = sort([a for a in ARGS if !any(o -> startswith(a, "--$o"),
                                                 ("checkpoint=", "restart-from=",
                                                  "walltime=", "ckevery=", "flux=",
                                                  "stop-at-step="))])

# The newest checkpoint in `ck.from` that every rank has whole, or
# `nothing` without `--restart-from`.
function read_checkpoint(ck, fs)
    ck.from === nothing && return nothing
    have = Set(n for (n, _) in checkpoint_steps(ck.from, ck))
    # The ranks' newest ones may differ if a run was killed while writing:
    # the least of them, which every rank also has.
    n = rank_reduce(min, fs, isempty(have) ? -1 : maximum(have))
    rank_reduce(&, fs, n >= 0 && n in have) || error(
        "no checkpoint in $(ck.from) that every rank of $(ck.nranks) has whole")
    c = deserialize(joinpath(ck.from, checkpoint_name(ck, n)))
    c.args == checkpoint_args() || error(
        "the checkpoint was written by a run with the options $(join(c.args, ' ')), " *
        "not $(join(checkpoint_args(), ' '))")
    return c
end

binary_main()

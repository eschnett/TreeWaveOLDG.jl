# Two equal charges on one circle, half an orbit apart, in the seven-patch
# ball, with the field in the orbit's plane written at a cadence fit for a
# movie ("The movie" in `CODE.md`), at the thread count Julia was started
# with:
#
#     julia -t N,0 --project=. bin/orbit_movie.jl --plane=DIR [--hdiv=4] \
#         [--levels=3] [--w0=4] [--R1=100] [--hshell=1] [--order=3] \
#         [--v=0.2] [--rc=0.5] [--T=0] [--ramp=10] [--n=4] [--tend=90] \
#         [--rx1=200] [--rx2=210] [--cfl=0.5] [--pevery=0.25] \
#         [--grids=96:1024,8:512] [--every=1] [--waves=FILE] [--wevery=0.0625] \
#         [--wradii=10,20,30,50,70,90] [--lmax=6] [--backend=cpu|cuda]
#
# The layout is `bin/prescribed.jl`'s with `--ball` (`bin/ball.jl`): the
# ball's cube of half-width `Lc = 2^(levels-1) w0` holds `levels` nested
# cubes, the finest of half-width `w0` at `hₑ = 1/hdiv`, and shells of
# radial width about `hshell` run to a Sommerfeld sphere at `R1`. The
# charges are the first run's, `Q² = 0.005`, `ℓ = 10⁻³`, in the window
# `R₁ = 1.25`, `R₂ = 2.5`, on `binary_circular_motion`'s circle of radius
# `rc` at speed `v` about `o = (1.23, 0.71, 0.93)·10⁻²`, at rest before
# `-T - ramp` and spun up with the smoothstep of order `n` to the orbit at
# `-T`. The data are the charges' retarded field out to `rx1`, past
# included, so the orbit's waves start as a front at `|x| ≈ t + T + ramp`,
# which at the defaults reaches the sphere at `t = 90`.
#
# `--plane=DIR` writes a frame (`bin/plane_io.jl`) at `t = 0` and every
# `pevery`: the total field and its rate on each grid of `--grids`, a list
# of `halfwidth:n`, square grids about `o` in the orbit's plane. `--waves`
# writes the modes on spheres as `bin/prescribed.jl` does
# (`bin/waves_io.jl`). Every `every` it prints, as `bin/prescribed.jl`
# does, the time, the force on each charge along its acceleration
# (`Fr`) and its velocity (`Ft`) and the wall-clock seconds since the
# start. `bin/orbit_exact.jl` writes the exact field's frames for the same
# options, and `bin/visualize_orbit.jl` makes the movie.

using TreeWaveOLDG
using TreeWaveOLDG: SVector, step!, statelength, trajectories
using LinearAlgebra: dot, norm

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "ball.jl"))
include(joinpath(@__DIR__, "waves_io.jl"))
include(joinpath(@__DIR__, "plane_io.jl"))

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

num(name, default) = parse(Float64, option(name, string(default)))

function emit(row...)
    println(join((Threads.nthreads(), row...), '\t'))
    flush(stdout)
    return nothing
end

const OFFSET = SVector(1.23, 0.71, 0.93) * 1e-2

# `--grids=96:1024,8:512`, about the orbit's centre `o`, named `w96`, `w8`.
function grids(o)
    return map(split(option("grids", "96:1024,8:512"), ",")) do g
        h, n = split(strip(g), ":")
        hw = parse(Float64, h)
        (; name="w" * (isinteger(hw) ? string(Int(hw)) : string(hw)), halfwidth=hw,
         n=parse(Int, n), centre=Tuple(o))
    end
end

# The orbit's options, shared with `bin/orbit_exact.jl`.
function orbit_options()
    return (; v=num("v", 0.2), rc=num("rc", 0.5), T=num("T", 0), ramp=num("ramp", 10),
            n=parse(Int, option("n", "4")), tend=num("tend", 90),
            pevery=num("pevery", 0.25), R1=num("R1", 100), order=parse(Int, option("order", "3")))
end

orbit_motion(o, opts) = binary_circular_motion(o, opts.rc, opts.v; past=opts.T,
                                               ramp=opts.ramp, n=opts.n)

const MODEL = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)

function orbit_meta(o, opts, source; extra...)
    return Any["source" => source, "Q" => MODEL.Q, "ell" => MODEL.ℓ, "K" => 2,
               "order" => opts.order, "v" => opts.v, "rc" => opts.rc,
               "Omega" => opts.v / opts.rc, "centre" => Tuple(o), "T" => opts.T,
               "ramp" => opts.ramp, "n" => opts.n, "R1" => opts.R1,
               "window" => (1.25, 2.5), "tend" => opts.tend, "pevery" => opts.pevery,
               (String(k) => v for (k, v) in extra)...]
end

function main()
    withbackend((backend, _) -> evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function evolve(backend)
    opts = orbit_options()
    hdiv = parse(Int, option("hdiv", "4"))
    levels = parse(Int, option("levels", "3"))
    w0 = num("w0", 4)
    hs = num("hshell", 1)
    rx1, rx2 = num("rx1", 200), num("rx2", 210)
    every, cfl = num("every", 1), num("cfl", 0.5)
    dir = option("plane", "")
    P = 3
    c = SVector(0.0, 0.0, 0.0)
    o = c + OFFSET
    wave = ball_wave(hdiv, levels, w0; R1=opts.R1, hshell=hs, backend)
    layout = ball_layout(hdiv, levels, w0; R1=opts.R1, hshell=hs)
    motion = orbit_motion(o, opts)
    p = InspiralProblem(wave, (MODEL, MODEL), Window(c, 1.25, 2.5; continuity=P + 1),
                        motion; order=opts.order)
    retarded = Window(c, rx1, rx2; continuity=4)
    start = time()
    u0 = inspiral_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl)
    integ = inspiral_integrator(p, u0, 0.0, opts.tend; dt)
    println(join(("threads", "t", "quantity", "value"), '\t'))
    for (name, value) in ("nodes" => statelength(p.wave.fs) ÷ 2, "hdiv" => hdiv,
                          "Lc" => layout.Lc, "R1" => opts.R1, "m" => layout.m,
                          "hshell" => layout.hshell, "steps" => integ.nsteps,
                          "dt" => integ.dt, "Omega" => opts.v / opts.rc,
                          "setup_seconds" => time() - start)
        emit(0.0, name, value)
    end
    gs = grids(o)
    exs = [PlaneExtraction(p.wave, o, g.halfwidth, g.n) for g in gs]
    writer = isempty(dir) ? nothing :
             plane_writer(dir, orbit_meta(o, opts, "run"; hdiv, levels, w0,
                                          hshell=layout.hshell, Lc=layout.Lc,
                                          m=layout.m, E=layout.E,
                                          dt=integ.dt, rx=(rx1, rx2)), gs, 2)
    function frame(u, t)
        writer === nothing && return nothing
        write_frame(writer, t, trajectories(p, u, t), [plane_values(ex, p, u, t) for ex in exs])
        return nothing
    end
    function forces(u, t)
        Fs = particle_forces(p, u, t)
        for (k, (F, tr)) in enumerate(zip(Fs, trajectories(p, u, t)))
            ρ = -tr.a / norm(tr.a)
            emit(t, "Fr$k", dot(F, ρ))
            emit(t, "Ft$k", dot(F, tr.v / norm(tr.v)))
        end
        emit(t, "seconds", time() - start)
        return nothing
    end
    kp = max(1, round(Int, opts.pevery / integ.dt))
    kw = max(1, round(Int, num("wevery", 0.0625) / integ.dt))
    ke = max(1, round(Int, every / integ.dt))
    waves = wave_writer(p, c, retarded)
    frame(integ.u, integ.t)
    write_waves(waves, p, integ.u, integ.t, retarded)
    forces(integ.u, integ.t)
    while integ.nstep < integ.nsteps
        step!(integ)
        last = integ.nstep == integ.nsteps
        (integ.nstep % kp == 0 || last) && frame(integ.u, integ.t)
        (integ.nstep % kw == 0 || last) && write_waves(waves, p, integ.u, integ.t, retarded)
        (integ.nstep % ke == 0 || last) && forces(integ.u, integ.t)
    end
    waves === nothing || close(waves.io)
    writer === nothing || close(writer)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()

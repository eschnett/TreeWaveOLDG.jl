# A binary of opposite charges that inspirals under its forces, in the
# seven-patch ball, with the field in the orbit's plane written at a
# cadence fit for a movie ("The movie" in `CODE.md`), at the thread count
# Julia was started with:
#
#     julia -t N,0 --project=. bin/inspiral_movie.jl --plane=DIR [--ref=DIR] \
#         [--hdiv=4] [--levels=3] [--w0=4] [--R1=50] [--hshell=1] \
#         [--order=3] [--q2=0.2] [--ell=0.04] [--v=0.2] [--T=30] [--ramp=10] \
#         [--n=4] [--onset=2π/Ω] [--n2=4] [--tend=72] [--rx1=100] [--rx2=110] \
#         [--cfl=0.5] [--pevery=0.25] [--grids=48:1024,3:512] [--every=0.125] \
#         [--waves=FILE] [--wevery=0.0625] [--wradii=10,20,30,40] [--lmax=6] \
#         [--oz=1/(2hdiv)] [--source-addot=predicted] [--backend=cpu|cuda]
#
# `bin/binary.jl`'s binary with a larger charge: two particles of `M = 1`
# and `Q = ±q`, `q² = --q2`, of size `ℓ = --ell`, at separation 1, each at
# speed `v`, on `binary_circular_motion`'s circle about
# `o = (0.0123, 0.0071, oz)`, held there by Newton's force with the `G`
# that makes the predictor's force centripetal (`newtonian_binary`). The
# data are the retarded field of a past at rest before `−T − ramp`, spun
# up to the orbit at `−T` with the smoothstep of order `n`, and the force
# takes over from that past over `onset` with `n2` (`Dynamic`). From
# there the orbit shrinks at about `ḋ = −Δq²K/(6πd²)`: at the defaults
# from `d = 1` to about 0.7 by `t = 72`. `ℓ ∝ q²` keeps `δm/M = 0.26`, as
# in `bin/binary.jl`. The layout is `bin/binary.jl`'s (`bin/ball.jl`).
# The orbit's plane `z = oz` lies halfway between two element faces of
# the finest level: a support of `ℓ = 0.04` within `ℓ` of a face meets
# the elements on both sides, and each doubles the solve's projections
# near the particle, so `bin/binary.jl`'s `o_z = 0.0093`, chosen for
# `ℓ = 10⁻³`, would make every step several times dearer.
#
# The source takes the predictor's `ä`, not the solved one
# (`--source-addot=predicted`, `Dynamic`'s `source_addot`): with the
# solved `ä` the binary breathes at a period of about 0.8, growing from
# the start at every `q²` above about 0.02, and the run at `q² = 0.2`
# broke down at `t = 17` ("The breathing instability" in `CODE.md`).
# `--source-addot=solved` brings that back.
#
# `--plane=DIR` writes a frame (`bin/plane_io.jl`) at `t = 0` and every
# `pevery`: the total field and its rate on each grid of `--grids`, a
# list of `halfwidth:n`, square grids about `o` in the orbit's plane. The
# run records its path in `DIR/jets.tsv`, every step: `t` and each
# particle's `z`, `v` and solved `a`. There is no exact field to compare
# with, so the reference is the retarded field of that recorded path: a
# quintic Hermite interpolation of `z`, `v` and `a` between steps, the past
# before `t = 0`. `--ref=DIR` writes its frames, on the same grids at the
# same times, and `--waves=FILE` its modes beside the run's (with the
# `ref_` columns of `bin/waves_io.jl`); both differ from the run by the
# grid's error and the particles' size alone. `bin/visualize_orbit.jl`
# makes the movie.
#
# Every `every` in time it prints the orbit (`orbit_diagnostics`): `d`,
# `ddot`, the leading order's `ddot0` at that `d` (`dipole_ddot`),
# `phase`, `omega`, `energy`, `com` (the centre of mass's distance from
# `o`) and `momentum`; per particle the remainder's force along the
# inward radius and the velocity (`Fr<k>`, `Ft<k>`); Newton's
# `iterations` and `residual`; and the wall-clock `seconds`.

include(joinpath(@__DIR__, "orbit_movie.jl"))
using TreeWaveOLDG: trajectory_at
using LinearAlgebra: cross

# The path the run takes, as `retarded_field` reads it: the past before
# the first record, a quintic Hermite interpolation of the recorded `z`,
# `v` and `a` between records, and the last record's Taylor polynomial
# after it.
struct RecordedPath{P,K}
    past::P
    t::Vector{Float64}
    z::Vector{NTuple{K,SVector{3,Float64}}}
    v::Vector{NTuple{K,SVector{3,Float64}}}
    a::Vector{NTuple{K,SVector{3,Float64}}}
end

RecordedPath(past, K) = RecordedPath{typeof(past),K}(past, Float64[],
                                                     NTuple{K,SVector{3,Float64}}[],
                                                     NTuple{K,SVector{3,Float64}}[],
                                                     NTuple{K,SVector{3,Float64}}[])

function record!(r::RecordedPath, t, zs, vs, as)
    push!(r.t, t)
    push!(r.z, Tuple(zs))
    push!(r.v, Tuple(vs))
    push!(r.a, Tuple(as))
    return r
end

function (r::RecordedPath{P,K})(s) where {P,K}
    (isempty(r.t) || s <= r.t[1]) && return r.past(s)
    n = length(r.t)
    if n == 1 || s >= r.t[n]
        σ = s - r.t[n]
        return ntuple(k -> (; z=r.z[n][k] + σ * (r.v[n][k] + σ * r.a[n][k] / 2),
                            v=r.v[n][k] + σ * r.a[n][k]), Val(K))
    end
    i = searchsortedlast(r.t, s)
    h = r.t[i + 1] - r.t[i]
    τ = (s - r.t[i]) / h
    τ2, τ3 = τ^2, τ^3
    τ4, τ5 = τ2^2, τ2 * τ3
    # The basis of values, first and second derivatives at each end.
    H0, H5 = 1 - 10τ3 + 15τ4 - 6τ5, 10τ3 - 15τ4 + 6τ5
    H1, H4 = τ - 6τ3 + 8τ4 - 3τ5, -4τ3 + 7τ4 - 3τ5
    H2, H3 = (τ2 - 3τ3 + 3τ4 - τ5) / 2, (τ3 - 2τ4 + τ5) / 2
    D0 = -30τ2 + 60τ3 - 30τ4
    D1, D4 = 1 - 18τ2 + 32τ3 - 15τ4, -12τ2 + 28τ3 - 15τ4
    D2, D3 = (2τ - 9τ2 + 12τ3 - 5τ4) / 2, (3τ2 - 8τ3 + 5τ4) / 2
    return ntuple(Val(K)) do k
        z0, z1 = r.z[i][k], r.z[i + 1][k]
        v0, v1 = r.v[i][k], r.v[i + 1][k]
        a0, a1 = r.a[i][k], r.a[i + 1][k]
        z = H0 * z0 + H5 * z1 + h * (H1 * v0 + H4 * v1) + h^2 * (H2 * a0 + H3 * a1)
        v = D0 * (z0 - z1) / h + D1 * v0 + D4 * v1 + h * (D2 * a0 + D3 * a1)
        (; z, v)
    end
end

# The retarded field of `path`'s charges `Qs` and its rate on the points
# `xs` of an `n × n` grid at `t`, `NaN` beyond the sphere at `R1`.
function retarded_plane(path, Qs, xs, t, R1)
    Φ = fill(NaN32, length(xs))
    Π = fill(NaN32, length(xs))
    Threads.@threads for j in eachindex(xs)
        x = xs[j]
        norm(x) < R1 || continue
        s, r = 0.0, 0.0
        for k in eachindex(Qs)
            f, ft = retarded_field(path, k, Qs[k], x, t)
            s += f
            r += ft
        end
        Φ[j], Π[j] = s, r
    end
    n = isqrt(length(xs))
    return reshape(Φ, n, n), reshape(Π, n, n)
end

function inspiral_main()
    withbackend((backend, _) -> inspiral_evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function inspiral_evolve(backend)
    hdiv = parse(Int, option("hdiv", "4"))
    levels = parse(Int, option("levels", "3"))
    w0, R1, hs = num("w0", 4), num("R1", 50), num("hshell", 1)
    order = parse(Int, option("order", "3"))
    q2, ℓ = num("q2", 0.2), num("ell", 0.04)
    v, d0 = num("v", 0.2), 1.0
    Ω = 2v / d0
    T, ramp = num("T", 30), num("ramp", 10)
    n, n2 = parse(Int, option("n", "4")), parse(Int, option("n2", "4"))
    onset = num("onset", 2π / Ω)
    tend, cfl = num("tend", 72), num("cfl", 0.5)
    rx1, rx2 = num("rx1", 100), num("rx2", 110)
    pevery, every = num("pevery", 0.25), num("every", 0.125)
    dir, refdir = option("plane", ""), option("ref", "")
    source_addot = Symbol(option("source-addot", "predicted"))
    isempty(dir) && error("--plane=DIR is needed: the path is recorded there")
    P = 3
    start = time()
    o = SVector(OFFSET[1], OFFSET[2], num("oz", 1 / (2hdiv)))
    wave = ball_wave(hdiv, levels, w0; R1, hshell=hs, P, backend)
    layout = ball_layout(hdiv, levels, w0; R1, hshell=hs)
    q = sqrt(q2)
    ms = (ParticleModel(; Q=q, ℓ, M=1.0), ParticleModel(; Q=-q, ℓ, M=1.0))
    f, past = newtonian_binary(ms, d0, v; centre=o, past=T, ramp, n)
    motion = Dynamic((NoForce(), NoForce()), past; interaction=f, onset, n2,
                     source_addot)
    p = InspiralProblem(wave, ms, Window(o, 1.25, 2.5; continuity=P + 1), motion; order)
    retarded = Window(o, rx1, rx2; continuity=4)
    u0 = dynamic_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt)
    K = binding_constant(ms, f)
    println(join(("threads", "t", "quantity", "value"), '\t'))
    for (name, value) in ("q2" => q2, "ell" => ℓ, "G" => f.G, "K" => K,
                          "ddot0" => dipole_ddot(ms, f, d0), "Omega" => Ω, "onset" => onset,
                          "source_addot_predicted" => Int(source_addot === :predicted),
                          "nodes" => statelength(p.wave.fs) ÷ 2, "hdiv" => hdiv,
                          "Lc" => layout.Lc, "R1" => R1, "m" => layout.m,
                          "hshell" => layout.hshell, "steps" => integ.nsteps,
                          "dt" => integ.dt, "setup_seconds" => time() - start)
        emit(0.0, name, value)
    end
    meta(source) = Any["source" => source, "dynamic" => 1, "Q" => (q, -q), "ell" => ℓ,
                       "M" => 1.0, "K" => 2, "order" => order, "v" => v, "d0" => d0,
                       "rc" => d0 / 2, "Omega" => Ω, "centre" => Tuple(o), "T" => T,
                       "ramp" => ramp, "n" => n, "onset" => onset, "n2" => n2,
                       "source_addot" => String(source_addot),
                       "G" => f.G, "binding" => K, "R1" => R1, "window" => (1.25, 2.5),
                       "tend" => tend, "pevery" => pevery, "hdiv" => hdiv,
                       "levels" => levels, "w0" => w0, "hshell" => layout.hshell,
                       "Lc" => layout.Lc, "m" => layout.m, "E" => layout.E,
                       "dt" => integ.dt, "rx" => (rx1, rx2)]
    gs = grids(o, "48:1024,3:512")
    exs = [PlaneExtraction(p.wave, o, g.halfwidth, g.n) for g in gs]
    points = [TreeWaveOLDG.plane_grid(SVector(g.centre), g.halfwidth, g.n) for g in gs]
    writer = plane_writer(dir, meta("run"), gs, 2)
    refwriter = isempty(refdir) ? nothing : plane_writer(refdir, meta("retarded"), gs, 2)
    path = RecordedPath(past.path, 2)
    jets = open(joinpath(dir, "jets.tsv"), "w")
    println(jets, join(("t", ("$w$k$c" for k in 1:2 for w in ("z", "v", "a")
                              for c in ("x", "y", "z"))...), '\t'))
    function record(u, t)
        zv = particle_state(u, p)
        sl = motion.scratch.last        # the step's last solve
        as = sl === nothing ? map(tr -> tr.a, past.path(t)) :
             map(tr -> trajectory_at(tr, t - sl.t).a, sl.trajs)
        record!(path, t, map(first, zv), map(x -> x[2], zv), as)
        println(jets, join((t, (x for k in 1:2 for x in (zv[k][1]..., zv[k][2]..., as[k]...))...),
                           '\t'))
        return nothing
    end
    function frame(u, t)
        trajs = trajectories(p, u, t)
        write_frame(writer, t, trajs,
                    [plane_values(ex, p, u, t; trajs) for ex in exs])
        if refwriter !== nothing
            write_frame(refwriter, t, path(t),
                        [retarded_plane(path, (q, -q), xs, t, R1) for xs in points])
        end
        flush(jets)
        return nothing
    end
    function report(u, t)
        zv = particle_state(u, p)
        zs, vs = map(first, zv), map(last, zv)
        od = orbit_diagnostics(ms, f, zs, vs)
        for (name, value) in (("d", od.d), ("ddot", od.ddot),
                              ("ddot0", dipole_ddot(ms, f, od.d)), ("phase", od.phase),
                              ("omega", od.omega), ("energy", od.energy),
                              ("com", norm(od.centre - o)), ("momentum", norm(od.momentum)))
            emit(t, name, value)
        end
        Fg = particle_forces(p, u, t)
        for k in 1:2
            ρ = (od.centre - zs[k]) / norm(od.centre - zs[k])
            emit(t, "Fr$k", dot(Fg[k], ρ))
            emit(t, "Ft$k", dot(Fg[k], vs[k] / norm(vs[k])))
        end
        l = motion.scratch.last
        if l !== nothing
            emit(t, "iterations", l.iterations)
            emit(t, "residual", l.residual)
        end
        emit(t, "seconds", time() - start)
        return nothing
    end
    kp = max(1, round(Int, pevery / integ.dt))
    kw = max(1, round(Int, num("wevery", 0.0625) / integ.dt))
    ke = max(1, round(Int, every / integ.dt))
    waves = wave_writer(p, o, retarded)
    record(integ.u, integ.t)
    frame(integ.u, integ.t)
    write_waves(waves, p, integ.u, integ.t, retarded; path)
    report(integ.u, integ.t)
    while integ.nstep < integ.nsteps
        step!(integ)
        record(integ.u, integ.t)
        final = integ.nstep == integ.nsteps
        (integ.nstep % kp == 0 || final) && frame(integ.u, integ.t)
        (integ.nstep % kw == 0 || final) && write_waves(waves, p, integ.u, integ.t, retarded;
                                                       path)
        (integ.nstep % ke == 0 || final) && report(integ.u, integ.t)
    end
    close(jets)
    waves === nothing || close(waves.io)
    close(writer)
    refwriter === nothing || close(refwriter)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && inspiral_main()

# A particle on a prescribed path with known acceleration, INSPIRAL-PLAN.md's
# M4 (INSPIRAL.md, "Validation", step 4), at the thread count Julia was
# started with:
#
#     julia -t N,0 --project=. bin/prescribed.jl [--path=circle] [--hdiv=8] \
#         [--levels=5] [--w0=4] [--order=3] [--tend] [--every=0.0625] \
#         [--fevery=1] [--cfl=0.5] [--v=0.05] [--rc=0.5] [--g=0.1] \
#         [--data=instantaneous] [--rx1] [--rx2] [--plain] [--both] \
#         [--cost] [--reps=3]
#
# Like `bin/uniform_motion.jl`, this runs against the package environment;
# `bin/symmetry/prescribed.sh` runs M4's series on a Symmetry node.
#
# The forest is M3b's nested boxes (`nested_cubes`), by default the
# layout chosen for M4 to M7: five cubes about the centre `c`, the finest
# of half-width `w0 = 4` at `hₑ = 1/hdiv`, each coarser one twice as wide
# at twice the element width, inside a periodic box of side `2^levels w0`;
# `2 w0 hdiv` elements across every level, a multiple of `4E` (`E` is 4
# where that fits, else 2). `--levels=1` is a uniform periodic box of
# side `2 w0`. One particle of the first run of INSPIRAL.md ("Parameters
# for a first run"), `Q² = 0.005`, `ℓ = 10⁻³`, `M = 1`; the window
# `R₁ = 1.25`, `R₂ = 2.5` about `c`, of continuity `P + 1`; `P = 3`, RK4 at
# `cfl`, the near field carried to `order`. `o = (1.23, 0.71, 0.93)·10⁻²`
# keeps the particle off the element faces' planes, as in M3.
#
# `--path=circle`: the slow circle, about `c + o` in the `xy` plane, of
# radius `rc` at speed `v`, starting at `c + o + (rc, 0, 0)` and turning
# counterclockwise. With `--data=instantaneous`, the default for the
# circle, the remainder starts as `(1 - W)Ψ̂`, the particle field of the
# instantaneous state ("Initial data", step 2), which lacks the
# radiation-reaction field; that start-up junk leaves, and the
# remainder's force should then be `(Q²/12π)ȧ` along the motion and zero
# across it. With `--data=retarded` the circle starts as the hyperbolic
# run below does, from the retarded field of the point charge on the
# circle forever, here inside `rx1 = 24`, blended into the particle field
# by `rx2 = 30`, and has the same reference. `--tend` defaults to 8.
#
# `--path=hyperbolic`: hyperbolic motion of proper acceleration `g` along
# `y`, at rest at `c + o + (0.5, 0, 0)` at `t = 0`, always with
# `--data=retarded`. Inside the ball of radius `rx1 = 3` about `c` the
# initial total field is the point charge's
# retarded field (`retarded_field`, by root-finding for the retarded
# time), blended into the particle field by `rx2 = 3.75`
# (`inspiral_initial_state(p; retarded = Window(c, rx1, rx2))`), so the
# remainder starts as the exact one there, which at `t = 0` is the
# time-symmetric field beyond the orders carried. The same function at
# `t` is the reference: in the ball of radius `R₁` about `c` it is exact
# until a signal from `rx1` arrives. The retarded field reaches only
# `1/g` behind the particle, which `rx2 + 0.6` must not exceed.
# `--tend` defaults to the time at which the speed reaches 0.05.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads path hdiv levels w0 order t quantity value
#
# with `path` `circle-ret` for the circle from the retarded field.
# At `t = 0`: `nodes`, `nodes_level<k>` per level `k` (0 the root),
# `steps`, `dt`, and for the circle `rr`, the radiation-reaction force
# `(Q²/12π)ȧ·v̂` (negative), and `Omega`. Every `fevery` steps, the
# remainder's force on the particle: for the circle its radial
# (outward, `Fr`), tangential (along `v`, `Ft`) and normal (`Fn`)
# components; for hyperbolic motion its component along the acceleration
# (`Fa`) and the size of the rest (`Fp`). Whenever the particle's centre
# crosses a plane of element faces of the finest level, `cross_x`,
# `cross_y` or `cross_z` with the time, interpolated between steps.
# Every `every` in time: the remainder's L2 norm in the ball of radius
# `R₂` about `c` (`field_R2`), the L2 norm of `Π_r` alone in the shell
# between `R₂` and `w0` (`junk`), where the start-up junk passes on its
# way out, and how far the particle's state is from its path (`dz`); with
# `--data=retarded` also the error against the reference, `(Φ_r, Π_r)`
# together, in quadrature-weighted L2, in the ball `R₁` about `c` without
# the ball of radius 1/4 about the particle (`err_R1`), in the whole ball
# (`err_R1_all`), of `Φ_r` alone (`errphi_R1`), its L∞ there
# (`errinf_R1`), in the ball of radius 1/2 about the particle (`err_z`),
# and the reference's L2 in the ball `R₁` without the hole (`field_R1`).
#
# With `--reading` and `--data=retarded`, nothing is evolved: the force of
# the exact remainder as the pairing reads it, at `t = 0` for the circle
# (`Fr0`, `Ft0`, `Fn0`) and at `tend` for hyperbolic motion (`Fa0`,
# `Fp0`), and the same with the `t′²` term of the lab-slice re-expansion
# of `Φ_C`, `-½γ²(v·r)²(α′·∇Φ₀)`, put back into the remainder (`Fr0_not2`,
# …), which is what the remainder held before the particle field carried
# that term (M4's first series).
#
# The source is written through the averaging over the nodes elements
# share and the force read through it (`InspiralProblem`'s `continuous`);
# `--plain` runs without it, as M4's first two series did. With `--both`,
# every force row is followed by the same read the other way: from the
# averaged field after `--plain` (`Frc`, `Ftc`, `Fnc`, `Fac`, `Fpc`, and
# `Fr0c`, … with `--reading`), which is what M4's `--continuous` runs
# recorded, and from the field as it is otherwise (`Frp`, …).
#
# With `--cost`, at the end, as `bin/uniform_motion.jl`: seconds per
# call, the least of `reps` after a warm-up, of `inspiral_rhs!` (`rhs`),
# `wave_rhs!` (`wave`), `project_source!` (`source`), `particle_forces`
# (`forces`) and an RK4 step (`step`).
#
# "Prescribed paths" in `CODE.md`'s "Measured results" has the numbers.

using TreeWaveOLDG
using TreeWaveOLDG: SVector, step!, statelength, level, maxlevel, field_state, statearray,
                    gather!, truncate_state!, trajectory_at, boost, ForwardDiff, TreeAMR,
                    get_backend
using LinearAlgebra: dot, norm

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function emit(row...)
    println(join((Threads.nthreads(), row...), '\t'))
    flush(stdout)
    return nothing
end

# The forest, as in `bin/uniform_motion.jl`: `levels` nested cubes about the
# centre, the finest of half-width `w0`, `2 w0 hdiv` elements across each,
# or for one level a uniform periodic box of side `2 w0`. Returns the
# forest and the outer side.
function box(hdiv, levels, w0)
    P = 3
    across = 2w0 * hdiv
    isinteger(across) || error("2 w0 hdiv = $across elements across a level is not whole")
    across = round(Int, across)
    if levels == 1
        E = first(e for e in (4, 2) if across % e == 0)
        return wave_forest(Float64, Val(3), P, E; roots=across ÷ E, L=2w0), 2w0
    end
    fits = [e for e in (4, 2) if across % (4e) == 0]
    isempty(fits) && error(
        "2 w0 hdiv = $across elements across a level is not a multiple of 4E for E = 2 " *
        "or 4, so the nested cubes' faces would not fall on block faces; choose another " *
        "w0 or hdiv")
    E = first(fits)
    Lout = 2^levels * w0
    forest = wave_forest(Float64, Val(3), P, E; roots=across ÷ E, L=Lout,
                         refined=nested_cubes(Val(3), w0, levels))
    return forest, Lout
end

const OFFSET = SVector(1.23, 0.71, 0.93) * 1e-2

# The `t′²` term of the lab-slice re-expansion of `Φ_C` (INSPIRAL.md,
# "Moving particles and the lab slice"; `CODE.md`, "Prescribed paths"),
# `-½γ²(v·r)²(α′·∇Φ₀(y))`, in its point form `∇Φ₀ = Qy/4π|y|³`, which the
# particle field carries from order 1 on (`lab_slice`). For `--reading`
# only: put back into the remainder, it shows what the remainder held
# before the particle field carried it.
function tprime2(Q, x, tr)
    v, r = tr.v, x - tr.z
    γ² = 1 / (1 - dot(v, v))
    y = boost(v, r)
    α′ = γ² * boost(v, tr.a)
    return -γ² / 2 * dot(v, r)^2 * dot(α′, Q * y / (4π * norm(y)^3))
end

struct TPrime2{PA,W}
    path::PA
    Q::Float64
    window::W
    t::Float64
end

function (f::TPrime2)(x, var)
    X = SVector{3}(x)
    w = f.window(X)
    tr = f.path(f.t)[1]
    (iszero(w) || norm(X - tr.z) < 1e-6) && return 0.0
    var == 1 && return w * tprime2(f.Q, X, tr)
    return w * ForwardDiff.derivative(s -> tprime2(f.Q, X, trajectory_at(tr, s)), 0.0)
end

# `--reading`: the force of the exact remainder at `t`, read by the
# pairing without any evolution, as it is and with the `t′²` term of `Φ_C`
# put back into it, as it was before the particle field carried it.
function reading(path, p, retarded, t, tag)
    u0 = inspiral_initial_state(p; t, retarded)
    fill_at_nodes!(TPrime2(p.motion.path, p.models[1].Q, retarded, t), p.wave.fs,
                   p.wave.element)
    t2 = zero(u0)
    f2 = field_state(t2, p)
    TreeAMR.zerofill!(statearray(f2, p.wave.fs), get_backend(p.wave.fs.work))
    gather!(f2, p.wave.fs)
    truncate_state!(f2, p.wave)
    for (suffix, u) in (("0", u0), ("0_not2", u0 .+ t2))
        for (name, value) in force_rows(path, p, u, t)
            emit(tag..., t, name * suffix, value)
        end
    end
    return nothing
end

function problem(path, hdiv, levels, w0, order; v, rc, g)
    P = 3
    forest, L = box(hdiv, levels, w0)
    wave = WaveProblem(forest, Element(Float64, P))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    motion = path == "circle" ? circular_motion(c + OFFSET, rc, v) :
             path == "hyperbolic" ?
             hyperbolic_motion(c + OFFSET + SVector(0.5, 0.0, 0.0), g; direction=(0, 1, 0)) :
             error("--path is circle or hyperbolic, got $path")
    return InspiralProblem(wave, (m,), w, motion; order=order,
                           continuous=!("--plain" in ARGS)), c
end

# The force's components: radial, tangential and normal on the circle;
# along the acceleration and the size of the rest in hyperbolic motion.
# With `--both` the same again, read the other way, with the names ending
# in `c` for the averaged reading or `p` for the plain one.
function force_rows(path, p, u, t)
    rows = force_components(path, p, u, t, p.continuous)
    "--both" in ARGS || return rows
    suffix = p.continuous ? "p" : "c"
    return (rows..., map(r -> r.first * suffix => r.second,
                         force_components(path, p, u, t, !p.continuous))...)
end

function force_components(path, p, u, t, continuous)
    F = particle_forces(p, u, t; continuous)[1]
    tr = p.motion.path(t)[1]
    if path == "circle"
        ρ = -tr.a / norm(tr.a)
        τ = tr.v / norm(tr.v)
        n = SVector(ρ[2] * τ[3] - ρ[3] * τ[2], ρ[3] * τ[1] - ρ[1] * τ[3],
                    ρ[1] * τ[2] - ρ[2] * τ[1])
        return ("Fr" => dot(F, ρ), "Ft" => dot(F, τ), "Fn" => dot(F, n))
    end
    â = tr.a / norm(tr.a)
    Fa = dot(F, â)
    return ("Fa" => Fa, "Fp" => norm(F - Fa * â))
end

function observe(path, p, c, w0, u, t, tag; retarded)
    tr = p.motion.path(t)[1]
    (zs, _), = particle_state(u, p)
    rows = Pair{String,Float64}[
        "field_R2" => ball_norm(p, u, c, p.window.R₂),
        "junk" => ball_norm(p, u, c, w0; vars=2, hole=(c, p.window.R₂)),
        "dz" => maximum(abs, zs - tr.z)]
    if retarded !== nothing
        ue = inspiral_initial_state(p; t=t, retarded=retarded)
        e = u .- ue
        R₁ = p.window.R₁
        hole = (tr.z, 0.25)
        append!(rows, ["err_R1" => ball_norm(p, e, c, R₁; hole),
                       "err_R1_all" => ball_norm(p, e, c, R₁),
                       "errphi_R1" => ball_norm(p, e, c, R₁; vars=1, hole),
                       "errinf_R1" => ball_norm(p, e, c, R₁; p=Inf, hole),
                       "err_z" => ball_norm(p, e, tr.z, 0.5),
                       "field_R1" => ball_norm(p, ue, c, R₁; hole)])
    end
    for (name, value) in rows
        emit(tag..., t, name, value)
    end
    return nothing
end

function main()
    path = option("path", "circle")
    hdiv = parse(Int, option("hdiv", "8"))
    levels = parse(Int, option("levels", "5"))
    w0 = parse(Float64, option("w0", "4"))
    order = parse(Int, option("order", "3"))
    every = parse(Float64, option("every", "0.0625"))
    fevery = parse(Int, option("fevery", "1"))
    cfl = parse(Float64, option("cfl", "0.5"))
    v = parse(Float64, option("v", "0.05"))
    rc = parse(Float64, option("rc", "0.5"))
    g = parse(Float64, option("g", "0.1"))
    data = option("data", path == "circle" ? "instantaneous" : "retarded")
    data in ("instantaneous", "retarded") ||
        error("--data is instantaneous or retarded, got $data")
    path == "hyperbolic" && data != "retarded" &&
        error("hyperbolic motion starts from the retarded field")
    rx1 = parse(Float64, option("rx1", path == "circle" ? "24" : "3"))
    rx2 = parse(Float64, option("rx2", path == "circle" ? "30" : "3.75"))
    reps = parse(Int, option("reps", "3"))
    # The hyperbolic run ends when the speed reaches 0.05, gt/√(1 + g²t²).
    tdefault = path == "circle" ? 8.0 : 0.05 / (g * sqrt(1 - 0.05^2))
    tend = parse(Float64, option("tend", string(tdefault)))
    p, c = problem(path, hdiv, levels, w0, order; v, rc, g)
    path == "hyperbolic" && (rx2 + 0.6 < 1 / g || error(
        "the retarded field reaches 1/g = $(1 / g) behind the particle, and the ball " *
        "of radius rx2 = $rx2 about the centre reaches about $(rx2 + 0.6) behind it; " *
        "choose a smaller g or rx2"))
    retarded = data == "retarded" ? Window(c, rx1, rx2; continuity=4) : nothing
    tag = (path * (path == "circle" && data == "retarded" ? "-ret" : ""), hdiv, levels,
           w0, order)
    println(join(("threads", "path", "hdiv", "levels", "w0", "order", "t", "quantity",
                  "value"), '\t'))

    if "--reading" in ARGS
        retarded === nothing && error("--reading needs --data=retarded")
        if path == "circle"
            m = p.models[1]
            tr = p.motion.path(0.0)[1]
            emit(tag..., 0.0, "rr", m.Q^2 / 12π * dot(tr.ȧ, tr.v / norm(tr.v)))
        end
        return reading(path, p, retarded, path == "circle" ? 0.0 : tend, tag)
    end
    u0 = inspiral_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl=cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt=dt)
    nodes = statelength(p.wave.fs) ÷ 2
    emit(tag..., 0.0, "nodes", nodes)
    forest = p.wave.fs.forest
    for k in 0:maxlevel(forest)
        emit(tag..., 0.0, "nodes_level$k",
             count(key -> level(key) == k, forest.leaves) * forest.N^3)
    end
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    if path == "circle"
        m = p.models[1]
        tr = p.motion.path(0.0)[1]
        emit(tag..., 0.0, "rr", m.Q^2 / 12π * dot(tr.ȧ, tr.v / norm(tr.v)))
        emit(tag..., 0.0, "Omega", v / rc)
    end
    hₑ = 1 / hdiv
    cell(t) = floor.(Int, (p.motion.path(t)[1].z - c) / hₑ)
    k = max(1, round(Int, every / integ.dt))
    observe(path, p, c, w0, integ.u, integ.t, tag; retarded)
    for (name, value) in force_rows(path, p, integ.u, integ.t)
        emit(tag..., integ.t, name, value)
    end
    t₋, cell₋ = integ.t, cell(integ.t)
    while integ.nstep < integ.nsteps
        step!(integ)
        t = integ.t
        cellt = cell(t)
        for d in 1:3
            cellt[d] == cell₋[d] && continue
            # The crossing, linear between the steps.
            z₋, z = p.motion.path(t₋)[1].z[d], p.motion.path(t)[1].z[d]
            face = c[d] + hₑ * max(cellt[d], cell₋[d])
            emit(tag..., t₋ + (t - t₋) * (face - z₋) / (z - z₋), "cross_" * "xyz"[d], face)
        end
        t₋, cell₋ = t, cellt
        if integ.nstep % fevery == 0 || integ.nstep == integ.nsteps
            for (name, value) in force_rows(path, p, integ.u, t)
                emit(tag..., t, name, value)
            end
        end
        (integ.nstep % k == 0 || integ.nstep == integ.nsteps) &&
            observe(path, p, c, w0, integ.u, t, tag; retarded)
    end

    "--cost" in ARGS || return nothing
    u = integ.u
    du = similar(u)
    t = integ.t
    trajs = p.motion.path(t)
    supports = map(ParticleSupport, p.models, trajs)
    fu, fdu = field_state(u, p), field_state(du, p)
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
            "forces" => best(() -> particle_forces(p, u, t), reps),
            "step" => best(() -> step!(stepper), reps))
        emit(tag..., t, name, value)
    end
    return nothing
end

main()

# Outgoing waves through nested boxes, INSPIRAL-PLAN.md's M3b, the field
# alone, at the thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/nested_waves.jl [--levels=4] [--w0=4] \
#         [--hdiv=8] [--lambda=16] [--sigma=lambda/2] [--tend=2^levels w0] \
#         [--every=0.5] [--cfl=0.5]
#
# Like `bin/uniform_motion.jl`, this runs against the package environment;
# `bin/symmetry/nested_waves.sh` runs it on a Symmetry node.
#
# The forest is M3b's nested boxes (`nested_cubes`): `levels` cubes about
# the centre `c` of a periodic box of side `2^levels w0`, the finest of
# half-width `w0` at `hₑ = 1/hdiv`, each coarser one twice as wide at
# twice the element width, `2 w0 hdiv` elements across every level (a
# multiple of `4E`; `E` is 4 where that fits, else 2); `P = 3`, RK4 at
# `cfl`, truncation on. The data are the exact outgoing pulse of the wave
# equation in three dimensions,
#
#     u(r, t) = [g(r - t) - g(r + t)] / r,    r = |x - c|,
#     g(s) = exp(-(s/σ)²) cos(2πs/λ),
#
# which is even in `s`, so `u` is regular at the centre; at `t = 0` it is
# `u = 0`, `v = -2g′(r)/r`. Its wavelength is `λ`: 16 is the first run's
# radiation, 4 stands in for the start-up junk. Near the centre the
# quotient is replaced by its Taylor series in `r`. The pulse leaves the
# finest cube, and what comes back into it afterwards is what the
# coarse-fine faces reflect, the first at `w0`, each further one at
# `2^k w0`, and the periodic box's faces at `2^(levels-1) w0`; a reflection
# from the face at `w` returns to the centre no sooner than about `2w`.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads levels w0 hdiv lambda sigma t quantity value
#
# At `t = 0`: `nodes`, `nodes_level<k>` per level `k` (0 the root),
# `steps`, `dt`. Every `every` in time, for each level's cube, the ball of
# radius `w_k = 2^k w0` about the centre (`k = 0` the finest, inside the
# finest level): the error against the exact solution, `u` and `v`
# together, in quadrature-weighted L2 (`err_w<w>`) and L∞ (`errinf_w<w>`),
# the exact solution's L∞ there, `u` and `v` together (`exactinf_w<w>`),
# and the discrete
# energy (`energy`), which the nested boxes conserve to RK4's loss until
# the pulse reaches the periodic box's faces.

using TreeWaveOLDG
using TreeWaveOLDG: ForwardDiff, step!, statelength, level, maxlevel

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

function emit(row...)
    println(join((Threads.nthreads(), row...), '\t'))
    flush(stdout)
    return nothing
end

# The exact outgoing pulse about `c`, as `(x, var) -> value` for
# `fill_at_nodes!`: `u` for variable 1, `v = ∂ₜu` for variable 2. `isbits`.
struct OutgoingPulse
    c::NTuple{3,Float64}
    σ::Float64
    λ::Float64
    t::Float64
end

pulse_g(p::OutgoingPulse, s) = exp(-(s / p.σ)^2) * cos(2π * s / p.λ)
pulse_dg(p::OutgoingPulse, s) = ForwardDiff.derivative(s -> pulse_g(p, s), s)

# `u` at `(r, t)`. Below `r = 10⁻³σ` the quotient loses digits, and it is
# its Taylor series `-2[g′(t) + g‴(t) r²/6]` instead, whose next term is
# `O(r⁴/σ⁴)` relative.
function pulse_u(p::OutgoingPulse, r, t)
    r > 1e-3 * p.σ && return (pulse_g(p, r - t) - pulse_g(p, r + t)) / r
    d1 = pulse_dg(p, t)
    d3 = ForwardDiff.derivative(s -> ForwardDiff.derivative(
                                    s′ -> pulse_dg(p, s′), s), t)
    return -2 * (d1 + d3 * r^2 / 6)
end

function (p::OutgoingPulse)(x, var)
    r = sqrt(sum(d -> (x[d] - p.c[d])^2, 1:3))
    var == 1 && return pulse_u(p, r, p.t)
    return ForwardDiff.derivative(t -> pulse_u(p, r, t), p.t)
end

function forest_of(levels, w0, hdiv)
    P = 3
    across = 2w0 * hdiv
    isinteger(across) || error("2 w0 hdiv = $across elements across a level is not whole")
    fits = [e for e in (4, 2) if round(Int, across) % (4e) == 0]
    isempty(fits) && error(
        "2 w0 hdiv = $across elements across a level is not a multiple of 4E for E = 2 " *
        "or 4, so the nested cubes' faces would not fall on block faces; choose another " *
        "w0 or hdiv")
    E = first(fits)
    return wave_forest(Float64, Val(3), P, E; roots=round(Int, across) ÷ E,
                       L=2^levels * w0, refined=nested_cubes(Val(3), w0, levels))
end

function observe(p, u, t, c, σ, λ, levels, w0, tag)
    ue = initial_state(p, OutgoingPulse(Tuple(c), σ, λ, t))
    e = u .- ue
    for k in 0:(levels - 1)
        w = 2^k * w0
        name = isinteger(w) ? string(Int(w)) : string(w)
        emit(tag..., t, "err_w$name", ball_norm(p, e, c, w))
        emit(tag..., t, "errinf_w$name", ball_norm(p, e, c, w; p=Inf))
        emit(tag..., t, "exactinf_w$name", ball_norm(p, ue, c, w; p=Inf))
    end
    emit(tag..., t, "energy", wave_energy(p, u))
    return nothing
end

function main()
    levels = parse(Int, option("levels", "4"))
    w0 = parse(Float64, option("w0", "4"))
    hdiv = parse(Int, option("hdiv", "8"))
    λ = parse(Float64, option("lambda", "16"))
    σ = parse(Float64, option("sigma", string(λ / 2)))
    tend = parse(Float64, option("tend", string(2^levels * w0)))
    every = parse(Float64, option("every", "0.5"))
    cfl = parse(Float64, option("cfl", "0.5"))
    tag = (levels, w0, hdiv, λ, σ)
    println(join(("threads", "levels", "w0", "hdiv", "lambda", "sigma", "t", "quantity",
                  "value"), '\t'))

    forest = forest_of(levels, w0, hdiv)
    p = WaveProblem(forest, Element(Float64, 3))
    L = 2^levels * w0
    c = (L / 2, L / 2, L / 2)
    u0 = initial_state(p, OutgoingPulse(c, σ, λ, 0.0))
    integ = wave_integrator(p, u0, 0.0, tend; dt=oldg_timestep(p; cfl=cfl))
    emit(tag..., 0.0, "nodes", statelength(p.fs) ÷ 2)
    for k in 0:maxlevel(forest)
        emit(tag..., 0.0, "nodes_level$k",
             count(key -> level(key) == k, forest.leaves) * forest.N^3)
    end
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    n = max(1, round(Int, every / integ.dt))
    observe(p, integ.u, integ.t, c, σ, λ, levels, w0, tag)
    while integ.nstep < integ.nsteps
        step!(integ)
        (integ.nstep % n == 0 || integ.nstep == integ.nsteps) &&
            observe(p, integ.u, integ.t, c, σ, λ, levels, w0, tag)
    end
    return nothing
end

main()

# Outgoing waves through nested cubes inside the seven-patch ball,
# INSPIRAL-PLAN.md's M8b, the field alone, at the thread count Julia was
# started with:
#
#     julia -t N,0 --project=. bin/nested_ball.jl [--levels=4] [--w0=4] \
#         [--hdiv=8] [--lambda=16] [--sigma=lambda/2] [--R1=3Lc] \
#         [--hshell=hroot] [--tend=2R1] [--every=0.5] [--cfl=0.5] [--cost=no]
#
# Like `bin/nested_waves.jl`, which it mirrors, this runs against the
# package environment; `bin/symmetry/nested_ball.sh` runs it on a Symmetry
# node.
#
# The forest is the seven-patch ball (`sevenpatch_forest`) whose cube holds
# M3b's nested cubes: `levels` levels about the origin, the finest of
# half-width `w0` at `hₑ = 1/hdiv`, each coarser one twice as wide at twice
# the element width, `2 w0 hdiv` elements across every level (a multiple of
# `4E`; `E` is 4 where that fits, else 2), so that the cube, of half-width
# `Lc = 2^(levels-1) w0`, is `nested_waves.jl`'s periodic box with the same
# options. Six shells continue it to the sphere `R1` (default `3Lc`), with
# radial elements about `hshell` wide at the cube's face centres (default
# the cube's root level's element width `hroot`), and a
# `RadiativeBoundary()` there (Sommerfeld's condition with its `u/R` term,
# exact for this monopole). Beside the cube's corners a shell's radial
# line runs from `√3 Lc` to `R1` only, and its elements there are thinner
# and sheared: at `R1 = 3Lc` the shells' own step is about 0.32 of the
# cube's root level's, so with fewer than three levels they, not the
# finest cube, set the time step (CODE.md, "The seven-patch ball").
# `P = 3`, RK4 at `cfl` times `oldg_timestep`, truncation on. The data are
# `nested_waves.jl`'s exact outgoing pulse about the origin,
#
#     u(r, t) = [g(r - t) - g(r + t)] / r,    g(s) = exp(-(s/σ)²) cos(2πs/λ),
#
# filled at the nodes' physical positions. What comes back into the finest
# cube after the pulse has left it is what the coarse-fine faces reflect,
# as on nested boxes, then what the cube's faces, where the shells take
# over, and the sphere send back; the sphere's return reaches the centre
# no sooner than about `2R1`.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads levels w0 hdiv lambda sigma R1 t quantity value
#
# At `t = 0`: `nodes`, `nodes_level<k>` per level `k` (0 the root),
# `nodes_shells`, `shell_radial_elements`, `steps`, `dt`. Every `every` in time, for each level's
# cube, the ball of radius `w_k = 2^k w0` about the origin (`k = 0` the
# finest), and for the whole ball (`w = R1`): the error against the exact
# solution, `u` and `v` together, `√g`-weighted L2 (`err_w<w>`) and L∞
# (`errinf_w<w>`), the exact solution's L∞ there (`exactinf_w<w>`), and the
# discrete energy (`energy`), the radiative boundary's own included. The
# balls are taken in physical coordinates on every patch, since a logical
# position names a point only with its patch. Its wall-clock times, in
# seconds, go with them: `build_s` at `t = 0`, the problem's build with
# the shells' truncation tables; then at each observation `step_s`, the
# mean step since the last one, `observe_s`, this observation's own, and
# `wall_s`, since the start.
#
# With `--cost=yes` it first times the right-hand side and the truncation
# per node, on the ball (`rhs_ns_per_node`, `truncate_ns_per_node`) and on
# `nested_waves.jl`'s nested boxes with the same cubes (`…_brick`), and
# reports each problem's memory (`problem_bytes`, `problem_bytes_brick`)
# and the brick's node count (`nodes_brick`).

using TreeWaveOLDG
using TreeWaveOLDG: ForwardDiff, step!, statelength, level, maxlevel, node_weight
# TreeAMR through TreeWaveOLDG: the cluster's environment holds only the
# package (`bin/symmetry/nested_ball.sh`).
using TreeWaveOLDG.TreeAMR: TreeAMR, WithPatch, mesh_fold, patch, statevector, Sum, Max

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

# `nested_waves.jl`'s exact outgoing pulse about the origin at time `t`,
# as `(x, var) -> value` of the physical position. `isbits`.
struct ModulatedPulse
    σ::Float64
    λ::Float64
    t::Float64
end

pulse_g(p::ModulatedPulse, s) = exp(-(s / p.σ)^2) * cos(2π * s / p.λ)
pulse_dg(p::ModulatedPulse, s) = ForwardDiff.derivative(s -> pulse_g(p, s), s)

# `u` at `(r, t)`. Below `r = 10⁻³σ` the quotient loses digits, and it is
# its Taylor series `-2[g′(t) + g‴(t) r²/6]` instead, whose next term is
# `O(r⁴/σ⁴)` relative.
function pulse_u(p::ModulatedPulse, r, t)
    r > 1e-3 * p.σ && return (pulse_g(p, r - t) - pulse_g(p, r + t)) / r
    d1 = pulse_dg(p, t)
    d3 = ForwardDiff.derivative(s -> ForwardDiff.derivative(
                                    s′ -> pulse_dg(p, s′), s), t)
    return -2 * (d1 + d3 * r^2 / 6)
end

function (p::ModulatedPulse)(x, var)
    r = sqrt(sum(abs2, x))
    var == 1 && return pulse_u(p, r, p.t)
    return ForwardDiff.derivative(t -> pulse_u(p, r, t), p.t)
end

# The pulse at logical `x` of patch `q`, for `fill_at_nodes!` through
# `WithPatch`.
struct OnBall{G}
    geometry::G
    pulse::ModulatedPulse
end

(s::OnBall)(x, q, var) = s.pulse(physical_coordinates(s.geometry, x, q), var)

# The `√g`-weighted `p`-th power sum and volume of the nodes within the
# physical ball of radius `radius` about the origin, the patch from the
# fold's block tag and `√g` from the problem's coefficient set, at the
# point's own block and index.
struct PhysicalBall{EL,G,C}
    el::EL
    geometry::G
    coef::C
    radius::Float64
    p::Int
end

function (bf::PhysicalBall)(acc, q)
    x = physical_coordinates(bf.geometry, q.x, q.tag)
    sum(abs2, x) < bf.radius^2 || return acc
    w = node_weight(q.I, q.h, bf.el, Val(3)) * bf.coef[q.I..., 1, q.b]
    s = 0.0
    for v in q.u
        s += abs(v)^bf.p
    end
    return TreeAMR.combine(acc, (Sum(w * s), Sum(w * length(q.u))))
end

function ball_errors(p, geo, u, radius; norm=2)
    forest = p.fs.forest
    tag = key -> Int(patch(forest, key))
    if isinf(norm)
        fmax = (acc, q) -> begin
            x = physical_coordinates(geo, q.x, q.tag)
            sum(abs2, x) < radius^2 || return acc
            TreeAMR.combine(acc, Max(maximum(abs, q.u)))
        end
        return mesh_fold(fmax, TreeAMR.combine, Max(0.0), p.fs, u; tag=tag).value
    end
    rf = PhysicalBall(p.element, geo, p.metric.coef, Float64(radius), Int(norm))
    total, volume = mesh_fold(rf, TreeAMR.combine, (Sum(0.0), Sum(0.0)), p.fs, u;
                              tag=tag)
    return (total.value / volume.value)^(1 / norm)
end

# The ball, its cubes and its shells: `n` roots across the cube, `m`
# radially in each shell, `E` elements per block along each dimension.
function layout(levels, w0, hdiv, R1, hshell)
    across = 2w0 * hdiv
    isinteger(across) || error("2 w0 hdiv = $across elements across a level is not whole")
    fits = [e for e in (4, 2) if round(Int, across) % (4e) == 0]
    isempty(fits) && error(
        "2 w0 hdiv = $across elements across a level is not a multiple of 4E for E = 2 " *
        "or 4, so the nested cubes' faces would not fall on block faces; choose another " *
        "w0 or hdiv")
    E = first(fits)
    n = round(Int, across) ÷ E
    Lc = 2^(levels - 1) * w0
    hroot = 2Lc / across                       # the cube's root element width
    h = hshell === nothing ? hroot : hshell
    m = max(1, ceil(Int, (R1 - Lc) / h / E))
    return (; E, n, m, Lc)
end

function ball_problem(levels, w0, hdiv, R1, hshell)
    (; E, n, m, Lc) = layout(levels, w0, hdiv, R1, hshell)
    forest, geo = sevenpatch_forest(Float64, 3, E; roots=(n, m), Lc=Lc, R1=R1,
                                    refined=nested_cubes(Val(3), n / 2^levels, levels))
    p = WaveProblem(forest, Element(Float64, 3); geometry=geo,
                    boundary=RadiativeBoundary())
    return p, geo
end

function brick_problem(levels, w0, hdiv)
    (; E, n) = layout(levels, w0, hdiv, 2^levels * w0, nothing)
    forest = wave_forest(Float64, Val(3), 3, E; roots=n, L=2^levels * w0,
                         refined=nested_cubes(Val(3), w0, levels))
    return WaveProblem(forest, Element(Float64, 3))
end

# Nanoseconds per node of `f!()`, the fastest of `reps` calls after two to
# compile and warm up.
function per_node(f!, nodes; reps=10)
    f!()
    f!()
    return minimum(@elapsed(f!()) for _ in 1:reps) / nodes * 1e9
end

function cost(p, tag, suffix)
    u = statevector(p.fs)
    u .= sin.(1:length(u))
    du = similar(u)
    nodes = statelength(p.fs) ÷ 2
    emit(tag..., 0.0, "rhs_ns_per_node$suffix", per_node(() -> wave_rhs!(du, u, p, 0.0),
                                                         nodes))
    emit(tag..., 0.0, "truncate_ns_per_node$suffix", per_node(() -> truncate_state!(u, p),
                                                              nodes))
    emit(tag..., 0.0, "problem_bytes$suffix", Base.summarysize(p))
    return nothing
end

function observe(p, geo, u, t, σ, λ, levels, w0, R1, tag)
    ue = initial_state(p, WithPatch(OnBall(geo, ModulatedPulse(σ, λ, t))))
    e = u .- ue
    for w in [2.0^k * w0 for k in 0:(levels - 1)]
        name = isinteger(w) ? string(Int(w)) : string(w)
        emit(tag..., t, "err_w$name", ball_errors(p, geo, e, w))
        emit(tag..., t, "errinf_w$name", ball_errors(p, geo, e, w; norm=Inf))
        emit(tag..., t, "exactinf_w$name", ball_errors(p, geo, ue, w; norm=Inf))
    end
    name = isinteger(R1) ? string(Int(R1)) : string(R1)
    emit(tag..., t, "err_w$name", ball_errors(p, geo, e, R1 + 1))
    emit(tag..., t, "errinf_w$name", ball_errors(p, geo, e, R1 + 1; norm=Inf))
    emit(tag..., t, "exactinf_w$name", ball_errors(p, geo, ue, R1 + 1; norm=Inf))
    emit(tag..., t, "energy", wave_energy(p, u))
    return nothing
end

function main()
    start = time()
    levels = parse(Int, option("levels", "4"))
    w0 = parse(Float64, option("w0", "4"))
    hdiv = parse(Int, option("hdiv", "8"))
    λ = parse(Float64, option("lambda", "16"))
    σ = parse(Float64, option("sigma", string(λ / 2)))
    Lc = 2^(levels - 1) * w0
    R1 = parse(Float64, option("R1", string(3Lc)))
    hs = option("hshell", "")
    hshell = isempty(hs) ? nothing : parse(Float64, hs)
    tend = parse(Float64, option("tend", string(2R1)))
    every = parse(Float64, option("every", "0.5"))
    cfl = parse(Float64, option("cfl", "0.5"))
    withcost = option("cost", "no") == "yes"
    tag = (levels, w0, hdiv, λ, σ, R1)
    println(join(("threads", "levels", "w0", "hdiv", "lambda", "sigma", "R1", "t",
                  "quantity", "value"), '\t'))

    build = @elapsed (p, geo) = ball_problem(levels, w0, hdiv, R1, hshell)
    forest = p.fs.forest
    emit(tag..., 0.0, "build_s", build)
    if withcost
        cost(p, tag, "")
        brick = brick_problem(levels, w0, hdiv)
        emit(tag..., 0.0, "nodes_brick", statelength(brick.fs) ÷ 2)
        cost(brick, tag, "_brick")
        brick = nothing
        GC.gc()
    end
    u0 = initial_state(p, WithPatch(OnBall(geo, ModulatedPulse(σ, λ, 0.0))))
    integ = wave_integrator(p, u0, 0.0, tend; dt=oldg_timestep(p; cfl=cfl))
    emit(tag..., 0.0, "nodes", statelength(p.fs) ÷ 2)
    for k in 0:maxlevel(forest)
        emit(tag..., 0.0, "nodes_level$k",
             count(key -> patch(forest, key) == 1 && level(key) == k, forest.leaves) *
             forest.N^3)
    end
    emit(tag..., 0.0, "nodes_shells", count(key -> patch(forest, key) > 1, forest.leaves) *
                                      forest.N^3)
    emit(tag..., 0.0, "shell_radial_elements",
         layout(levels, w0, hdiv, R1, hshell).m * layout(levels, w0, hdiv, R1, hshell).E)
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    n = max(1, round(Int, every / integ.dt))
    timed_observe() = begin
        s = @elapsed observe(p, geo, integ.u, integ.t, σ, λ, levels, w0, R1, tag)
        emit(tag..., integ.t, "observe_s", s)
        emit(tag..., integ.t, "wall_s", time() - start)
    end
    timed_observe()
    steps = 0
    stepping = 0.0
    while integ.nstep < integ.nsteps
        stepping += @elapsed step!(integ)
        steps += 1
        if integ.nstep % n == 0 || integ.nstep == integ.nsteps
            emit(tag..., integ.t, "step_s", stepping / steps)
            steps = 0
            stepping = 0.0
            timed_observe()
        end
    end
    return nothing
end

main()

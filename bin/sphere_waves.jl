# Outgoing waves through the seven-patch ball's radiative sphere,
# INSPIRAL-PLAN.md's M8a, the field alone, at the thread count Julia was
# started with:
#
#     julia -t N,0 --project=. bin/sphere_waves.jl [--R1=50] [--Lc=16] \
#         [--n=16] [--m=5] [--E=4] [--lambda=16] [--sigma=lambda/2] \
#         [--ell=0] [--tend=2 R1 + 3 sigma] [--every=1] [--cfl=0.5] \
#         [--radii=4,8,16] [--Rm=32 --m2=4]
#
# Like `bin/nested_waves.jl`, this runs against the package environment;
# `bin/symmetry/sphere_waves.sh` runs it on a Symmetry node.
#
# The forest is `sevenpatch_forest`'s uniform ball: the cube of half-width
# `Lc` with `n` roots along each side, six shells of `n` roots tangentially
# and `m` radially out to the sphere `|x| = R1`, each block `E³` elements
# of `P = 3`; RK4 at `cfl` times `oldg_timestep`, truncation on, and a
# `RadiativeBoundary` with `κ = 1` and the `u/R` term (Sommerfeld's
# condition) on the sphere. The cube's element width is `2Lc/(nE)`, and a
# shell's tangential one at the sphere, largest at its face's centre,
# `2R1/(nE)`. With `--Rm` and `--m2` the forest is instead
# `thirteenpatch_forest`'s ball: the same cube and shells of `m` radial
# roots out to the middle sphere `|x| = Rm`, and six exactly radial shells
# of `m2` radial roots from there to `R1`; the rows at `t = 0` then add
# `Rm`, `m2` and `nodes_outer`, the outer shells' share of `nodes_shells`.
#
# The data are `nested_waves.jl`'s exact outgoing pulse about the origin,
#
#     U(r, t) = [g(r - t) - g(r + t)] / r,    g(s) = exp(-(s/σ)²) cos(2πs/λ),
#
# for `--ell=0`, and its `z`-derivative `∂_z U` for `--ell=1`, a dipole,
# which is also a solution and is the binary's angular order. Sommerfeld's
# condition is exact for the first up to the incoming part, which is below
# `exp(-(R1/σ)²)` at the sphere; for the second it reflects at
# `O(ℓ(ℓ+1)/(kR1)²)`. The pulse leaves the ball of radius `r` at about
# `r + 2σ`, reaches the sphere at `R1 - 2σ`, and what the sphere sends
# back reaches the centre no sooner than about `2R1 - 2σ`.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads R1 Lc n m E lambda sigma ell t quantity value
#
# At `t = 0`: `nodes`, `nodes_cube`, `nodes_shells`, `he_cube`,
# `he_sphere`, `steps`, `dt`. Every `every` in time, for each radius `r` of
# `radii`, in the physical ball `|x| < r`: the error against the exact
# solution, `u` and `v` together, `√g`-weighted L2 (`err_r<r>`) and L∞
# (`errinf_r<r>`), and the exact solution's L∞ there (`exactinf_r<r>`);
# and the discrete energy with the boundary's term (`energy`), which falls
# as the pulse leaves.

using TreeWaveOLDG
using TreeWaveOLDG: ForwardDiff, step!, statelength, TreeAMR, metric_factor, node_weight
using TreeWaveOLDG.TreeAMR: mesh_fold, combine, Sum, Max, patch, nleaves, WithPatch

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

# The exact pulse, as `(x, p, var) -> value` through `WithPatch`: `u` for
# variable 1, `v = ∂ₜu` for variable 2, at the physical position of
# logical `x` in patch `p`. `isbits`.
struct Pulse{G}
    geometry::G
    σ::Float64
    λ::Float64
    ell::Int
    t::Float64
end

pulse_g(p::Pulse, s) = exp(-(s / p.σ)^2) * cos(2π * s / p.λ)
pulse_dg(p::Pulse, s) = ForwardDiff.derivative(s -> pulse_g(p, s), s)

# `U` at `r² = q` and `t`. Below `r = 10⁻³σ` the quotient loses digits, and
# it is its Taylor series `-2[g′(t) + g‴(t) r²/6]` instead, whose next term
# is `O(r⁴/σ⁴)` relative. Written in `q`, so that a derivative in `x`
# through `q = |x|²` is finite at the origin.
function pulse_U(p::Pulse, q, t)
    if q > (1e-3 * p.σ)^2
        r = sqrt(q)
        return (pulse_g(p, r - t) - pulse_g(p, r + t)) / r
    end
    d1 = pulse_dg(p, t)
    d3 = ForwardDiff.derivative(s -> ForwardDiff.derivative(
                                    s′ -> pulse_dg(p, s′), s), t)
    return -2 * (d1 + d3 * q / 6)
end

# `U` for `ell = 0`, `∂_z U` for `ell = 1`, at the physical `x` and `t`.
function pulse_u(p::Pulse, x, t)
    p.ell == 0 && return pulse_U(p, x[1]^2 + x[2]^2 + x[3]^2, t)
    return ForwardDiff.derivative(z -> pulse_U(p, x[1]^2 + x[2]^2 + z^2, t), x[3])
end

function (p::Pulse)(ξ, patchid, var)
    x = physical_coordinates(p.geometry, ξ, patchid)
    var == 1 && return pulse_u(p, x, p.t)
    return ForwardDiff.derivative(t -> pulse_u(p, x, t), p.t)
end

# The `√g`-weighted `p`-th power sum and volume, or the largest value, over
# the nodes in the physical ball `|x| < r`. The patch reaches the fold as
# each block's tag; `√g` is evaluated only inside the ball.
struct PhysBall{EL,G}
    el::EL
    geometry::G
    r::Float64
    p::Int                              # 0: the largest value
end

function (b::PhysBall)(acc, q)
    x = physical_coordinates(b.geometry, q.x, q.tag)
    x[1]^2 + x[2]^2 + x[3]^2 < b.r^2 || return acc
    if b.p == 0
        return combine(acc, Max(maximum(abs, q.u)))
    end
    w = node_weight(q.I, q.h, b.el, Val(3)) * metric_factor(b.geometry, q.x, q.tag, 0)
    s = zero(w)
    for v in q.u
        s += abs(v)^b.p
    end
    return combine(acc, (Sum(w * s), Sum(w * length(q.u))))
end

function phys_ball_norm(prob, geo, u, r; p=2)
    fs = prob.fs
    forest = fs.forest
    tag = key -> Int(patch(forest, key))
    if isinf(p)
        b = PhysBall(prob.element, geo, Float64(r), 0)
        return mesh_fold(b, combine, Max(0.0), fs, u; vars=1:2, tag=tag).value
    end
    b = PhysBall(prob.element, geo, Float64(r), Int(p))
    total, volume = mesh_fold(b, combine, (Sum(0.0), Sum(0.0)), fs, u; vars=1:2, tag=tag)
    volume.value == 0 && return 0.0
    return (total.value / volume.value)^(1 / p)
end

function observe(prob, geo, u, t, σ, λ, ell, radii, tag)
    ue = initial_state(prob, WithPatch(Pulse(geo, σ, λ, ell, t)))
    e = u .- ue
    for r in radii
        name = isinteger(r) ? string(Int(r)) : string(r)
        emit(tag..., t, "err_r$name", phys_ball_norm(prob, geo, e, r))
        emit(tag..., t, "errinf_r$name", phys_ball_norm(prob, geo, e, r; p=Inf))
        emit(tag..., t, "exactinf_r$name", phys_ball_norm(prob, geo, ue, r; p=Inf))
    end
    emit(tag..., t, "energy", wave_energy(prob, u))
    return nothing
end

function main()
    R1 = parse(Float64, option("R1", "50"))
    Lc = parse(Float64, option("Lc", "16"))
    n = parse(Int, option("n", "16"))
    m = parse(Int, option("m", "5"))
    E = parse(Int, option("E", "4"))
    λ = parse(Float64, option("lambda", "16"))
    σ = parse(Float64, option("sigma", string(λ / 2)))
    ell = parse(Int, option("ell", "0"))
    ell in (0, 1) || error("--ell is 0 (the monopole) or 1 (the dipole), got $ell")
    tend = parse(Float64, option("tend", string(2R1 + 3σ)))
    every = parse(Float64, option("every", "1"))
    cfl = parse(Float64, option("cfl", "0.5"))
    radii = [parse(Float64, strip(r)) for r in split(option("radii", "4,8,16"), ",")]
    Rm = option("Rm", nothing)
    Rm = Rm === nothing ? nothing : parse(Float64, Rm)
    m2 = parse(Int, option("m2", "0"))
    Rm === nothing || m2 > 0 || error("--Rm needs --m2, the outer shells' radial roots")
    P = 3
    tag = (R1, Lc, n, m, E, λ, σ, ell)
    println(join(("threads", "R1", "Lc", "n", "m", "E", "lambda", "sigma", "ell", "t",
                  "quantity", "value"), '\t'))
    flush(stdout)

    forest, geo = Rm === nothing ?
                  sevenpatch_forest(Float64, P, E; roots=(n, m), Lc=Lc, R1=R1) :
                  thirteenpatch_forest(Float64, P, E; roots=(n, m, m2), Lc=Lc, Rm=Rm, R1=R1)
    prob = WaveProblem(forest, Element(Float64, P); geometry=geo,
                       boundary=RadiativeBoundary(; κ=1.0, inverse_r=true))
    u0 = initial_state(prob, WithPatch(Pulse(geo, σ, λ, ell, 0.0)))
    integ = wave_integrator(prob, u0, 0.0, tend; dt=oldg_timestep(prob; cfl=cfl))
    perblock = forest.N^3
    ncube = count(key -> patch(forest, key) == 1, forest.leaves) * perblock
    emit(tag..., 0.0, "nodes", statelength(prob.fs) ÷ 2)
    emit(tag..., 0.0, "nodes_cube", ncube)
    emit(tag..., 0.0, "nodes_shells", nleaves(forest) * perblock - ncube)
    if Rm !== nothing
        emit(tag..., 0.0, "Rm", Rm)
        emit(tag..., 0.0, "m2", m2)
        emit(tag..., 0.0, "nodes_outer",
             count(key -> patch(forest, key) >= 8, forest.leaves) * perblock)
    end
    emit(tag..., 0.0, "he_cube", 2Lc / (n * E))
    emit(tag..., 0.0, "he_sphere", 2R1 / (n * E))
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    k = max(1, round(Int, every / integ.dt))
    observe(prob, geo, integ.u, integ.t, σ, λ, ell, radii, tag)
    t0 = time()
    while integ.nstep < integ.nsteps
        step!(integ)
        if integ.nstep % k == 0 || integ.nstep == integ.nsteps
            observe(prob, geo, integ.u, integ.t, σ, λ, ell, radii, tag)
            emit(tag..., integ.t, "seconds", time() - t0)
        end
    end
    return nothing
end

main()

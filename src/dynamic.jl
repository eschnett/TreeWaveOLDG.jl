# Dynamic particles, INSPIRAL-PLAN.md's M6a: the parts of the equation
# of motion (INSPIRAL.md, "Equations of motion") that are not the grid's,
# the averages over a particle of the analytic fields near it, its own
# near field and the other particles' fields. See "The equations of
# motion" in `CODE.md`.

"""
    SupportRule(T; n_r = 16, n_θ = 8)

The quadrature over a particle's support of
[`support_integrals`](@ref): [`ball_rule`](@ref)`(T, n_r, n_θ)` on the
rest frame's unit ball, mapped to the support's ellipsoid in the lab.
The density is a polynomial of `|y|²` inside the support, and so are the
near-field terms and the lab slice, at any velocity, so the rule is
exact for the particle's own near field when `n_r` and `n_θ` cover their
degrees; the defaults hold the averages to roundoff (tested). The
support's edge is the rule's, so the profile's `C³` kink costs nothing.
"""
struct SupportRule{T}
    y::Vector{SVector{3,T}}
    w::Vector{T}
end

SupportRule(::Type{T}; n_r::Integer=16, n_θ::Integer=8) where {T} =
    SupportRule{T}(ball_rule(T, n_r, n_θ)...)

"""
    support_integrals(m, traj, rule, h)

`(∫ρ g, ∫ρ ġ, ∫(∇ρ) g)` over the support of a particle of model `m` on
`traj`, for `h(x) = (g, ġ)` returning two numbers, by `rule`
([`SupportRule`](@ref)): the lab point `x = z + Λ(v)⁻¹ ℓy` of each
rest-frame point `y` of the unit ball, with `d³x = ℓ³ d³y/γ`. Generic in
the number type, so that it works on dual numbers along the motion.
"""
function support_integrals(m::ParticleModel, traj::Trajectory, rule::SupportRule, h)
    (; z, v) = traj
    γ = lorentz_factor(v)
    scale = m.ℓ^3 / γ
    S = typeof(scale * first(h(z)))
    # The rule's points in chunks of a fixed size, threaded, and the
    # chunks' sums added in their order: the result is the same at every
    # thread count. A dynamic step evaluates these sums along nested dual
    # numbers in `τ` for its own field and the other particles' at every
    # residual, which serially was most of the solve's cost.
    n = length(rule.w)
    nchunks = cld(n, SUPPORT_CHUNK)
    parts = Vector{Tuple{S,S,SVector{3,S}}}(undef, nchunks)
    Threads.@threads :dynamic for k in 1:nchunks
        lo = (k - 1) * SUPPORT_CHUNK + 1
        parts[k] = support_sum(m, traj, rule, h, lo:min(n, lo + SUPPORT_CHUNK - 1), S)
    end
    Iρg, Iρġ, I∇ρg = parts[1]
    for k in 2:nchunks
        Iρg += parts[k][1]
        Iρġ += parts[k][2]
        I∇ρg += parts[k][3]
    end
    return scale * Iρg, scale * Iρġ, scale * I∇ρg
end

# The points of a chunk: 2048 of `SupportRule`'s defaults in 64 chunks,
# one per thread of a Symmetry node.
const SUPPORT_CHUNK = 32

# `support_integrals`' sums over the rule's points `js`, without the
# scale `ℓ³/γ`.
function support_sum(m::ParticleModel, traj::Trajectory, rule::SupportRule, h, js,
                     ::Type{S}) where {S}
    (; z, v) = traj
    ℓ = m.ℓ
    γ = lorentz_factor(v)
    c = γ / (γ + 1)
    Iρg, Iρġ, I∇ρg = zero(S), zero(S), zero(SVector{3,S})
    for j in js
        y, w = rule.y[j], rule.w[j]
        ly = ℓ * y
        x = z + ly - (c * dot(v, ly)) * v
        q = dot(ly, ly)
        ρ = m.Q * _rho0(q, ℓ)
        # `∇ₓρ = Λ ∇_y ρ₀(|y|²)`, and `Λ` is symmetric.
        ∇ρ = boost(v, (2 * m.Q * ForwardDiff.derivative(s -> _rho0(s, ℓ), q)) * ly)
        g, ġ = h(x)
        Iρg += w * ρ * g
        Iρġ += w * ρ * ġ
        I∇ρg += (w * g) * ∇ρ
    end
    return Iρg, Iρġ, I∇ρg
end

# The particle's own near field, `Ψ - Φ_C`: the near-field terms and the
# lab slices of orders 1 to `order`.
@inline function own_field(m::ParticleModel, x, traj::Trajectory, order::Int)
    Ψ = zero(eltype(x)) * zero(eltype(traj.a))
    order >= 1 && (Ψ += near_field(m, Val(1), x, traj) + lab_slice(m, Val(1), x, traj))
    order >= 2 && (Ψ += near_field(m, Val(2), x, traj) + lab_slice(m, Val(2), x, traj))
    order >= 3 && (Ψ += near_field(m, Val(3), x, traj) + lab_slice(m, Val(3), x, traj))
    return Ψ
end

"""
    field_force(m, traj, rule, h)

The force of the field `g` with rate `ġ`, `h(x) = (g, ġ)`, on a
particle of model `m` on `traj`, and its potential: `(F, QΦ̄)` with

    F = -(Q/γ)(∇̄g + v ġ̄) = ∫(∇ρ) g - v ∫ρ ġ,   QΦ̄ = γ ∫ρ g,

the averages of INSPIRAL.md's "Equations of motion", by
[`support_integrals`](@ref).
"""
function field_force(m::ParticleModel, traj::Trajectory, rule::SupportRule, h)
    Iρg, Iρġ, I∇ρg = support_integrals(m, traj, rule, h)
    return I∇ρg - traj.v * Iρġ, lorentz_factor(traj.v) * Iρg
end

"""
    own_force(m, traj, rule; order = 3)

The force of the particle's own near field on it, and that field's
potential, `(F, QΦ̄)`, as [`field_force`](@ref) gives them: the
near-field terms and the lab slices of orders 1 to `order`, `Ψ - Φ_C`,
and their rate along the trajectory. `Φ_C` exerts no force and its
potential is absorbed into the bare mass (INSPIRAL.md, "Equations of
motion"). At rest it is M1's `-δm a`, `-(2541/96577π)Q²ℓ ä -
(15246/482885π)Q²ℓ a²a` and `(2541/193154π)Qℓ a²` (`CODE.md`, "The near
field"), which the tests hold it to.
"""
function own_force(m::ParticleModel, traj::Trajectory, rule::SupportRule; order::Int=3)
    check_order(order)
    t0 = zero(eltype(traj.z))
    h(x) = (own_field(m, x, traj, order),
            ForwardDiff.derivative(t -> own_field(m, x, trajectory_at(traj, t), order), t0))
    return field_force(m, traj, rule, h)
end

"""
    other_force(m, traj, mb, trajb, rule; order = 3)

The force of the particle field `Ψ_b` of another particle, of model `mb`
on `trajb`, on a particle of model `m` on `traj`, and its potential,
`(F, QΦ̄)`, as [`field_force`](@ref) gives them, with `Ψ_b`'s rate
along `trajb`. The window is taken to be 1 on the support.
"""
function other_force(m::ParticleModel, traj::Trajectory, mb::ParticleModel,
                     trajb::Trajectory, rule::SupportRule; order::Int=3)
    check_order(order)
    t0 = zero(eltype(trajb.z))
    h(x) = (field_of_order(mb, x, trajb, order),
            ForwardDiff.derivative(t -> field_of_order(mb, x, trajectory_at(trajb, t), order),
                                   t0))
    return field_force(m, traj, rule, h)
end

# The kernels of the grid's part of the force, `∇ρ` and `ρ` of a particle
# on `traj`, as one static vector `(∇ρ..., ρ)`, so that one pairing reads
# both; generic, so that `traj` may be a trajectory of dual numbers in
# time, whose kernels stay compact on the support at `τ = 0` and
# polynomial inside it.
@inline function density_kernel(m::ParticleModel, x, traj::Trajectory)
    y = boost(traj.v, x - traj.z)
    q = dot(y, y)
    ρ = m.Q * _rho0(q, m.ℓ)
    ∇ρ = boost(traj.v, (2 * m.Q * ForwardDiff.derivative(s -> _rho0(s, m.ℓ), q)) * y)
    return SVector(∇ρ[1], ∇ρ[2], ∇ρ[3], ρ)
end

mutable struct DynamicScratch
    du::Any                             # a field's rate, outside the right-hand side
    w::Any                              # (Π_r, 0), whose rate is (0, LΠ_r)
    dw::Any
    last::Any                           # what the last solve did
    renew::Bool                         # find the next solve's Jacobian afresh
    bias::Any                           # the running mean of the predictor's `ä`'s bias
    bias_t::Any                         # when it was last brought forward
    seconds::Dict{Symbol,Float64}       # wall-clock seconds by part, summed over solves
end

DynamicScratch() = DynamicScratch(nothing, nothing, nothing, nothing, false, nothing,
                                  nothing, Dict(k => 0.0 for k in SOLVE_PARTS))

# The parts of a solve whose seconds `solve_seconds` reports: the whole
# solve, `LΠ`'s `wave_rhs!`, the source's projections, the residuals
# (those of the Jacobian included) and the Jacobians.
const SOLVE_PARTS = (:solve, :laplacian, :projection, :residual, :jacobian)

# `f()`, its wall-clock seconds added to part `key` of `sc`.
function tally(f, sc::DynamicScratch, key::Symbol)
    t0 = time_ns()
    result = f()
    sc.seconds[key] += (time_ns() - t0) / 1e9
    return result
end

"""
    solve_seconds(m::Dynamic)

The wall-clock seconds `m`'s solves have taken so far, by part, as a
named tuple: `solve` in all, and of it `laplacian` (`LΠ`'s `wave_rhs!`),
`projection` (the source's projections near the particles), `residual`
(every residual, the Jacobians' included) and `jacobian` (the Jacobians
by finite differences). For the drivers' cost rows.
"""
solve_seconds(m) = NamedTuple{SOLVE_PARTS}(map(k -> m.scratch.seconds[k], SOLVE_PARTS))

"""
    renew_jacobian!(m::Dynamic)

Have the next solve of `m`'s jets find its Jacobian afresh rather than
keep the last one, so that `m.scratch.last.jacobian` is then the one at
that solve's state: its `ä` block's distance from `MγI` is the grid's
self-inertia at the current `γ`, which a kept Jacobian misreads by the
change in `γ` since it was found (`CODE.md`, "The equations of motion"
under "Measured results"). For diagnostics; it costs a Jacobian.
"""
renew_jacobian!(m) = (m.scratch.renew = true; m)

"""
    solve_state(m::Dynamic)
    solve_state!(m::Dynamic, s)

What `m`'s next solve starts from beyond the state vector, as a named
tuple, and setting it: the last solve's record, whose jets give the next
guess and whose Jacobian the next solve keeps, whether the next one
finds its Jacobian afresh, and the running mean of the predictor's
`ä`'s bias with the time it was brought to, which `source_addot =
:corrected` writes into the source. A run resumed from a checkpoint of the state
vector and this, in a fresh `Dynamic` of the same arguments
([`resume_integrator!`](@ref)), takes the uninterrupted run's steps to
the bit (`bin/binary.jl`'s `--checkpoint`).
"""
solve_state(m) = (; last=m.scratch.last, renew=m.scratch.renew, bias=m.scratch.bias,
                   bias_t=m.scratch.bias_t)

function solve_state!(m, s)
    m.scratch.last = s.last
    m.scratch.renew = s.renew
    m.scratch.bias = s.bias
    m.scratch.bias_t = s.bias_t
    return m
end

"""
    Dynamic(external, past; interaction = NoInteraction(), onset = 0, n2 = 4,
            rule = SupportRule(T), tol = 1e-10, source_tol = 0.1,
            source_addot = :solved, source_tau = 2,
            predictor_external = external)

A motion in which each particle moves under its forces (INSPIRAL.md,
"Equations of motion"): `a`, `ȧ` and `ä` of every particle are solved
together from the state, from the equation of motion

    m γ a = F_grid + F_own + Σ_b F_b + (I - v vᵀ)(F_ext + F_N),   m = M₀ + QΦ̄ₑ,

and its first two time derivatives, by Newton's method. `F_grid` is the
remainder's force ([`particle_forces`](@ref)), `F_own` the particle's
own near field's ([`own_force`](@ref)), `F_b` the other particles'
fields' ([`other_force`](@ref)), `F_ext = external[k](z, v)` an
`isbits` callable such as [`SpringForce`](@ref), and `F_N` the
`interaction` with the other particles, such as the binary's
[`NewtonianForce`](@ref), along their jets. `a⃛` and `a⃜` come from
the predictor ([`predicted_jets`](@ref)), which takes the same
interaction and, unless `predictor_external` says otherwise, the same
external forces. `predictor_external` is for experiments on the
predictor alone, such as a [`DipoleDrag`](@ref) in place of the radiation
reaction it leaves out; the solve reads them only through the source's
`a⃛` and `a⃜`, and its `ä` with `source_addot = :predicted` ("The
predictor's drag" under "The breathing instability" in `CODE.md`).

`past` is a [`Prescribed`](@ref) motion: the initial data are its
retarded field, and the force takes over from it as
`E = (1 - S) M (a - a_past) + S E_phys`, `S = smoothstep(n2, t/onset)`,
so that at `t ≤ 0` the jets are the past's, and with `n2 = 4` they leave
it with `a`, `ȧ` and `ä` continuous (`CODE.md`, "Assumed pasts").
`onset = 0` hands over at once. `tol` is Newton's relative tolerance
on the jets. The source's projection near the particles, the solve's
costly part, is found once at the first guess and held while Newton
converges, and found again at the result only if the jets moved more
than `source_tol` from where it was found, relative to their size: the
source reads the jets so weakly that never finding it again moves the
force by `3·10⁻⁸ F_RR` while the force takes over, where the jets move
up to 9% from the guess (`CODE.md`, "The equations of motion").

`source_addot = :predicted` writes the source, and projects it in the
solve, with the predictor's `ä` (blended with the past's as `a⃛` is) in
place of the solved one, which the solve still finds and
`m.scratch.last.trajs` records. The blended predictor's `ä` is in
`m.scratch.last.addot_predicted` with either choice. It began as an
experiment on what the solved `ä`'s jumps at element faces do to the run
(`CODE.md`, "The binary" under "Measured results"), and is a candidate
cure of the binary's breathing instability ("The breathing
instability"). The solved `ä` departs from the predictor's by about
`0.005q²` of it along itself and `−0.035q²` toward `ȧ`; before the
predictor's inertia took the other charge's potential, by `0.07q²` along
itself ("The predictor with the potential").

`source_addot = :corrected` writes it with the predictor's `ä` plus a
running mean of that bias: each solve fits the solved `ä`'s departure
from the predictor's as `α ä_p + β (|ä_p|/|ȧ_p|) ȧ_p`, with `ä_p` and
`ȧ_p` the blended predictor's, which on a circle is constant, and the
mean of `(α, β)` follows it as `dμ/dt = (fit − μ)/τ`, `τ = source_tau`,
brought forward at each solve later in time than the last, as the
stages of a step are, once the force has taken over: during the onset the
solved jets follow the blend with the past, many times the bias at a
short onset, and the mean is held at zero. The source takes `μ` as it
stood before the solve. It is an experiment: no more accurate than
`:predicted`, with a margin that shrinks as `q²/τ`, and its mean is
state outside the state vector ("The corrected `ä`").
A `τ` long beside the breathing's period, `2π/8`, keeps the loop open
(`bin/breathing_loop.jl --source-addot=corrected`). The mean is in
`m.scratch.bias` with any choice, and the source's `ä` in
`m.scratch.last.addot_source`.
"""
struct Dynamic{X<:Tuple,Y<:Tuple,I,P<:Prescribed,T}
    external::X
    predictor_external::Y
    interaction::I
    past::P
    onset::T
    n2::Int
    rule::SupportRule{T}
    tol::T
    source_tol::T
    source_addot::Symbol
    source_tau::T
    scratch::DynamicScratch
end

function Dynamic(external::Tuple, past::Prescribed; interaction=NoInteraction(),
                 onset::Real=0, n2::Integer=4, rule=nothing, tol::Real=1e-10,
                 source_tol::Real=0.1, source_addot::Symbol=:solved,
                 source_tau::Real=2, predictor_external::Tuple=external)
    T = eltype(first(past.path(0.0)).z)
    source_addot in (:solved, :predicted, :corrected) || throw(ArgumentError(
        "source_addot must be :solved, :predicted or :corrected, got :$source_addot"))
    source_tau > 0 || throw(ArgumentError("source_tau must be positive, got $source_tau"))
    onset >= 0 || throw(ArgumentError("the onset must not be negative, got $onset"))
    length(external) == length(past.path(zero(T))) || throw(ArgumentError(
        "$(length(external)) external forces for $(length(past.path(zero(T)))) particles"))
    length(predictor_external) == length(external) || throw(ArgumentError(
        "$(length(predictor_external)) predictor forces for $(length(external)) particles"))
    rule === nothing && (rule = SupportRule(T))
    isbits(interaction) || throw(ArgumentError(
        "the interaction must be isbits, got a $(typeof(interaction))"))
    return Dynamic(external, predictor_external, interaction, past, T(onset), Int(n2),
                   rule, T(tol), T(source_tol), source_addot, T(source_tau),
                   DynamicScratch())
end

retarded_path(m::Dynamic) = m.past.path

# How far the force has taken over at `t`, which may be a dual number.
handover(m::Dynamic, t) = m.onset > 0 ? smoothstep(m.n2, t / m.onset) : one(t)

function trajectories(m::Dynamic, p::InspiralProblem, u, t)
    u === nothing && return m.past.path(t)
    fu = field_state(u, p)
    sc = m.scratch
    sc.du === nothing && (sc.du = similar(fu))
    wave_rhs!(sc.du, fu, p.wave, t)
    return solve_jets(m, p, u, sc.du, t)
end

rhs_trajectories(m::Dynamic, p::InspiralProblem, u, fdu, t) = solve_jets(m, p, u, fdu, t)

kernel_trajectories(m::Dynamic, p::InspiralProblem, u, t) =
    map((zv) -> Trajectory(; z=zv[1], v=zv[2]), particle_state(u, p))

"""
    dynamic_initial_state(p::InspiralProblem; t = 0, retarded, passes = 3)

[`inspiral_initial_state`](@ref) for a [`Dynamic`](@ref) motion: the
past's retarded field, and its jets at `t`. Where the force has taken
over at `t` (an `onset` of 0), the jets are solved from that state and
the near field `Ψ̂` rebuilt with them, `passes` times, so that the data
take the jump from the past's jets to the solved ones (`CODE.md`,
"Assumed pasts").
"""
function dynamic_initial_state(p::InspiralProblem{T}; t::Real=0, retarded=nothing,
                               passes::Integer=3) where {T}
    m = p.motion
    m isa Dynamic || throw(ArgumentError("dynamic_initial_state takes a Dynamic motion"))
    u = inspiral_initial_state(p; t, retarded)
    iszero(handover(m, T(t))) && return u
    for _ in 1:passes
        trajs = trajectories(m, p, u, T(t))
        u = inspiral_initial_state(p; t, retarded, trajs)
    end
    return u
end

# `LΠ_r`, the wave operator of the field's rate, as variable 2 of the
# rate of the state `(Π_r, 0)`.
function rate_laplacian(sc::DynamicScratch, p::InspiralProblem, fu, t)
    fs = p.wave.fs
    if sc.w === nothing
        sc.w, sc.dw = similar(fu), similar(fu)
    end
    wa, ua = statearray(sc.w, fs), statearray(fu, fs)
    v = ndims(ua) - 1
    selectdim(wa, v, 1) .= selectdim(ua, v, 2)
    selectdim(wa, v, 2) .= 0
    wave_rhs!(sc.dw, sc.w, p.wave, t)
    return sc.dw
end

"""
    dynamic_residual(m::Dynamic, p, fu, fdu, LΠ, zs, vs, higher, past, t, X)

The residuals of the equations of motion of every particle for the
unknown jets `X[k] = (a, ȧ, ä)`, a tuple per particle of
`(E, Ė, Ë)`: `E(τ)` along the jets' Taylor polynomial, blended with the
past's as [`Dynamic`](@ref) says, and its first two derivatives in `τ`
at 0. The grid's part reads the field `fu`, its rate `fdu` before the
source and `LΠ` through Taylor kernels, `⟨K_τ, Φ_r(τ)⟩` with
`Φ_r(τ) = Φ + τΠ + τ²∂ₜΠ/2` and `∂ₜΠ = LΦ - A P S`, and
`Π_r(τ) = Π + τ∂ₜΠ + τ²∂ₜ²Π/2` with `∂ₜ²Π = LΠ - A P Ṡ`.
"""
function dynamic_residual(m::Dynamic, p::InspiralProblem{T,K}, fu, fdu, LΠ, zs, vs,
                          higher, past, t, X, sources) where {T,K}
    jets = ntuple(k -> Trajectory(; z=zs[k], v=vs[k], a=X[k][1], ȧ=X[k][2], ä=X[k][3],
                                  a⃛=higher[k][1], a⃜=higher[k][2]), Val(K))
    pair(u, K, s, var) = pairing(p.wave, p.tables, u, K, s; var, continuous=p.continuous)
    return ntuple(Val(K)) do k
        mk = p.models[k]
        s = ParticleSupport(mk, jets[k])
        ρ(x) = particle_density(mk, x, zs[k], vs[k])
        PS, Ṡρ = sources[k]
        B4 = pair(LΠ, ρ, s, 2) - Ṡρ
        function E(τ)
            jτ = map(j -> trajectory_at(j, τ), jets)
            tr = jτ[k]
            Kτ(x) = density_kernel(mk, x, tr)
            P1 = pair(fu, Kτ, s, 1)
            P2 = pair(fu, Kτ, s, 2)
            P3 = pair(fdu, Kτ, s, 2) - pair_projected(p.tables, PS, Kτ, s)
            IΦ = P1 + τ * P2 + (τ^2 / 2) * P3
            IΠ = P2[4] + τ * P3[4] + (τ^2 / 2) * B4
            γ = lorentz_factor(tr.v)
            F = SVector(IΦ[1], IΦ[2], IΦ[3]) - tr.v * IΠ
            Q = γ * IΦ[4]
            Fo, Qo = own_force(mk, tr, m.rule; order=p.order)
            F += Fo
            Q += Qo
            for b in 1:K
                b == k && continue
                Fb, Qb = other_force(mk, tr, p.models[b], jτ[b], m.rule; order=p.order)
                F += Fb
                Q += Qb
            end
            Fx = m.external[k](tr.z, tr.v) +
                 pair_force(m.interaction, p.models, map(j -> j.z, jτ), k)
            Eφ = (mk.M₀ + Q) * γ * tr.a - F - (Fx - dot(tr.v, Fx) * tr.v)
            σ = handover(m, t + τ)
            return σ * Eφ + (1 - σ) * mk.M * (tr.a - trajectory_at(past[k], τ).a)
        end
        (E(zero(T)), ForwardDiff.derivative(E, zero(T)), second_derivative(E, zero(T)))
    end
end

# What the residual reads of the source of the jets `X`, per particle:
# `A P S` on the support's elements ([`projected_source`](@ref
# TreeWaveOLDG.projected_source)), and `⟨ρ, A P Ṡ⟩`, the source's rate
# along the jets. They are the expensive part, and depend weakly on the
# jets, so the solve holds them while it converges and finds them again
# at its result.
function source_projections(p::InspiralProblem{T,K}, zs, vs, higher, X) where {T,K}
    jets = ntuple(k -> Trajectory(; z=zs[k], v=vs[k], a=X[k][1], ȧ=X[k][2], ä=X[k][3],
                                  a⃛=higher[k][1], a⃜=higher[k][2]), Val(K))
    # One projection of the source along the jets at `δ = 0 + ε`: its
    # values are `A P S`, its partials `A P Ṡ`.
    δ = ForwardDiff.Dual{typeof(ForwardDiff.Tag(source_projections, T))}(zero(T), one(T))
    dm = map(q -> ParticleModel{typeof(δ)}(q.Q, q.ℓ, q.M), p.models)
    Sδ = ParticleSource(dm, p.window, map(j -> trajectory_at(j, δ), jets); order=p.order)
    return ntuple(Val(K)) do k
        mk = p.models[k]
        s = ParticleSupport(mk, jets[k])
        ρ(x) = particle_density(mk, x, zs[k], vs[k])
        proj = projected_source(p.wave, p.tables, Sδ, s; continuous=p.continuous,
                                rnear=p.rnear)
        PS = map(((lo, hi, g),) -> (lo, hi, map(ForwardDiff.value, g)), proj)
        (PS, ForwardDiff.partials(pair_projected(p.tables, proj, ρ, s), 1))
    end
end

"""
    solve_jets(m::Dynamic, p, u, fdu, t)

The particles' [`Trajectory`](@ref)s at `t` of the state `u`, whose
field's rate before the source is `fdu`: `z` and `v` from the state,
`a`, `ȧ` and `ä` solved from [`dynamic_residual`](@ref) by Newton's
method with a Jacobian of finite differences, from the blend of the
past's and the predictor's jets, and `a⃛`, `a⃜` that blend. At `t ≤ 0`,
before the force takes over, the past's jets. What the solve did is in
`m.scratch.last`.
"""
function solve_jets(m::Dynamic, p::InspiralProblem{T,K}, u, fdu, t) where {T,K}
    past = m.past.path(T(t))
    m.onset > 0 && t <= 0 && return past
    return tally(() -> solve_handed_over(m, p, u, fdu, t, past), m.scratch, :solve)
end

function solve_handed_over(m::Dynamic, p::InspiralProblem{T,K}, u, fdu, t,
                           past) where {T,K}
    fu = field_state(u, p)
    zv = particle_state(u, p)
    zs, vs = map(first, zv), map(last, zv)
    pred = predicted_jets(p.models, m.predictor_external, zs, vs;
                          interaction=m.interaction)
    σ = handover(m, T(t))
    mix(f) = ntuple(k -> (1 - σ) * getfield(past[k], f) + σ * getfield(pred[k], f), Val(K))
    higher = ntuple(k -> (mix(:a⃛)[k], mix(:a⃜)[k]), Val(K))
    sc = m.scratch
    if sc.bias === nothing
        sc.bias = ntuple(k -> SVector{2,T}(0, 0), Val(K))
        sc.bias_t = T(t)
    end
    # The basis of the predictor's bias, `ä_p` and `ȧ_p` scaled to its size.
    basis = ntuple(Val(K)) do k
        äp, ȧp = mix(:ä)[k], mix(:ȧ)[k]
        nȧ = norm(ȧp)
        (äp, nȧ > 0 ? (norm(äp) / nȧ) * ȧp : zero(ȧp))
    end
    correction(k) = sc.bias[k][1] * basis[k][1] + sc.bias[k][2] * basis[k][2]
    # The jets the source is written with: the solved ones, or with
    # `source_addot = :predicted` the predictor's `ä` in their place, with
    # `:corrected` that plus the mean of its bias.
    addot_source = ntuple(k -> m.source_addot === :corrected ? mix(:ä)[k] + correction(k) :
                               mix(:ä)[k], Val(K))
    sourced(X) = m.source_addot === :solved ? X :
                 ntuple(k -> (X[k][1], X[k][2], addot_source[k]), Val(K))
    LΠ = tally(() -> rate_laplacian(m.scratch, p, fu, T(t)), m.scratch, :laplacian)
    pack(X) = reduce(vcat, (vcat(x...) for x in X))
    unpack(x) = ntuple(k -> ntuple(j -> SVector{3,T}(x[9(k - 1) + 3(j - 1) .+ (1:3)]), 3),
                       Val(K))
    # The first guess: the last solve's jets, advanced to `t` along their
    # Taylor polynomials, if it was near in time; the blend otherwise.
    near = sc.last !== nothing && length(sc.last.trajs) == K &&
           abs(T(t) - sc.last.t) <= 4 * sc.last.reach
    x = if near
        Vector{T}(pack(ntuple(Val(K)) do k
            j = trajectory_at(sc.last.trajs[k], T(t) - sc.last.t)
            (j.a, j.ȧ, j.ä)
        end))
    else
        Vector{T}(pack(ntuple(k -> (mix(:a)[k], mix(:ȧ)[k], mix(:ä)[k]), Val(K))))
    end
    project(x) = tally(() -> source_projections(p, zs, vs, higher, sourced(unpack(x))),
                       sc, :projection)
    sources = project(x)
    xp = copy(x)                        # where the sources were found
    resid(x) = tally(sc, :residual) do
        pack(dynamic_residual(m, p, fu, fdu, LΠ, zs, vs, higher, past, T(t), unpack(x),
                              sources))
    end
    r = resid(x)
    n = length(x)
    # The residual is nearly affine in the jets, and its Jacobian changes
    # slowly with the state: it is kept from the last solve, and found
    # again by finite differences when there is none or the chord
    # iteration converges slowly with it.
    jacobian() = tally(jacobian_fd, sc, :jacobian)
    function jacobian_fd()
        J = Matrix{T}(undef, n, n)
        scale = pack(ntuple(k -> (pred[k].a, pred[k].ȧ, pred[k].ä), Val(K)))
        for i in 1:n
            # The step of each jet from its size, or from the particle's
            # acceleration where a jet vanishes, as `ȧ` does under a
            # constant force: the residual is nearly affine.
            blk = 3 * ((i - 1) ÷ 3)
            acc = 9 * ((i - 1) ÷ 9)
            h = sqrt(eps(T)) * max(norm(scale[(blk + 1):(blk + 3)]),
                                   norm(x[(blk + 1):(blk + 3)]),
                                   norm(x[(acc + 1):(acc + 3)]), floatmin(T) / eps(T))
            e = zeros(T, n)
            e[i] = h
            J[:, i] = (resid(x + e) - r) / h
        end
        return J
    end
    # A Jacobian found while the past still held the jets is a blend's:
    # wrong by `(γ - 1)` and the grid's self-inertia, the chord iteration
    # converges with it all the same, so it would be kept for good. It is
    # found again when the blend has moved by a quarter, and when the
    # force has taken over.
    fresh = sc.renew || sc.last === nothing || size(sc.last.jacobian) != (n, n) ||
            (σ != sc.last.σ && (σ == 1 || abs(σ - sc.last.σ) > 1 // 4))
    sc.renew = false
    J = fresh ? jacobian() : sc.last.jacobian
    σJ = fresh ? σ : sc.last.σ          # the blend at which `J` was found
    iterations = 0
    projections = 1
    converged = false
    previous = T(Inf)
    projected = false                   # the sources are those of the last iterate
    moves = T[]                         # how far the jets moved from each source
    for it in 1:20
        # The ranks of a distributed forest take rank 0's step, as LAPACK
        # may differ in its last bits between CPUs ("Running distributed").
        dx = rank_agree(p.wave.fs, J \ r)
        x -= dx
        r = resid(x)
        iterations = it
        step = norm(dx)
        if step <= m.tol * norm(x)
            # Converged with the source held. Done if the source is that of
            # jets this near (it reads them very weakly, `Dynamic`);
            # otherwise find it here.
            (projected || norm(x - xp) <= m.source_tol * norm(x)) &&
                (converged = true; break)
            push!(moves, norm(x - xp) / norm(x))
            sources = project(x)
            xp = copy(x)
            projections += 1
            r = resid(x)
            projected = true
            previous = T(Inf)
            continue
        end
        projected = false
        if step > previous / 10 && !fresh
            # The chord iteration converges slowly with the old Jacobian,
            # as it does while the force takes over.
            J = jacobian()
            fresh = true
            σJ = σ
            previous = T(Inf)
        else
            previous = step
        end
    end
    converged || @warn "the dynamic particles' jets did not converge at t = $t: " *
                       "residual $(norm(r))"
    X = unpack(x)
    trajs = ntuple(k -> Trajectory(; z=zs[k], v=vs[k], a=X[k][1], ȧ=X[k][2], ä=X[k][3],
                                   a⃛=higher[k][1], a⃜=higher[k][2]), Val(K))
    # How far in time a guess from these jets is trusted: an element
    # crossing, at the slowest of the particles' speeds.
    hₑ = frame_hₑ(p.wave.fs.forest, p.frame, order(p.wave.element) + 1)
    reach = hₑ / max(maximum(v -> norm(v), vs), eps(T))
    push!(moves, norm(x - xp) / norm(x))
    # The solved `ä`'s departure from the predictor's, fitted on the basis
    # by least squares, and the mean brought forward to `t`.
    fit = ntuple(Val(K)) do k
        b1, b2 = basis[k]
        δ = X[k][3] - mix(:ä)[k]
        G = SMatrix{2,2,T}(dot(b1, b1), dot(b2, b1), dot(b1, b2), dot(b2, b2))
        g = SVector{2,T}(dot(b1, δ), dot(b2, δ))
        # Where the predictor's `ä` or `ȧ` vanishes, as from rest under a
        # constant force, there is no bias to fit along it.
        iszero(G[1, 1]) ? zero(SVector{2,T}) :
        iszero(G[2, 2]) ? SVector{2,T}(g[1] / G[1, 1], 0) : G \ g
    end
    # While the force takes over, the solved jets follow the blend with
    # the past, not the predictor's bias, and the mean is held.
    if T(t) > sc.bias_t
        if σ == 1
            w = -expm1(-(T(t) - sc.bias_t) / m.source_tau)
            sc.bias = ntuple(k -> sc.bias[k] + w * (fit[k] - sc.bias[k]), Val(K))
        end
        sc.bias_t = T(t)
    end
    sc.last = (; t=T(t), iterations, projections, residual=norm(r), jacobian=J, σ=σJ,
               fresh, trajs, reach=min(reach, T(Inf)), moves, addot_predicted=mix(:ä),
               addot_source, fit)
    m.source_addot === :solved && return trajs
    S = sourced(X)
    return ntuple(k -> Trajectory(; z=zs[k], v=vs[k], a=S[k][1], ȧ=S[k][2], ä=S[k][3],
                                  a⃛=higher[k][1], a⃜=higher[k][2]), Val(K))
end

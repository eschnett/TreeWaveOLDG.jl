# The coupled problem of INSPIRAL.md: the remainder field on the OLDG mesh
# and the particles, stepped together. This is its first part
# (INSPIRAL-PLAN.md, M3): particles on a prescribed path, the source `P S`
# subtracted from `v`'s rate after `wave_rhs!`, and the remainder's force
# on each particle as a diagnostic that does not act. See "The right-hand
# side with particles" in `CODE.md`.
#
# The state is one plain vector: the field's state vector, `u` and `v`
# interleaved block by block as `statearray` has them, followed by `z` and
# `v` of each particle, six numbers per particle. IMEXRungeKutta's stage
# arithmetic by block owner takes a plain `Array` only, so the particles
# go at the end, owned by thread 1 in the partition, rather than into an
# `ArrayPartition` (`HISTORY.md`, "Particles").

"""
    Prescribed(path)

A motion given in advance: `path(t)` returns a tuple of
[`Trajectory`](@ref)s, one per particle, at time `t`, with `a`, `ȧ`, `ä`
and `a⃛` exact. The source and the forces read the particles from it, and
the particle part of the state follows it, `ż = v` and `v̇ = a`, as a
check. `path` must be `isbits`.
"""
struct Prescribed{F}
    path::F
end

# Uniform motion through `z` at `t = 0` with velocity `v`, per particle.
struct UniformPath{K,T}
    z::NTuple{K,SVector{3,T}}
    v::NTuple{K,SVector{3,T}}
end

(p::UniformPath)(t) = map((z, v) -> Trajectory(; z=z + t * v, v=v), p.z, p.v)

"""
    uniform_motion(z, v)

A [`Prescribed`](@ref) motion in which particle `k` moves uniformly
through `z[k]` at `t = 0` with velocity `v[k]`, for tuples `z` and `v` of
3-vectors.
"""
function uniform_motion(z::Tuple, v::Tuple)
    length(z) == length(v) || throw(ArgumentError(
        "got $(length(z)) positions and $(length(v)) velocities"))
    T = float(promote_type(map(eltype, z)..., map(eltype, v)...))
    return Prescribed(UniformPath(map(SVector{3,T}, z), map(SVector{3,T}, v)))
end

"""
    InspiralProblem(wave::WaveProblem, models, window, motion; order = 3,
                    tables = TransferTables(T, P), rnear = 1)

Everything the coupled right-hand side needs, built once: the
three-dimensional [`WaveProblem`](@ref) of the remainder, the particle
models (a tuple of [`ParticleModel`](@ref)s), the [`Window`](@ref), the
motion (a [`Prescribed`](@ref) one until M6a), the near-field `order`
(0–3) of the particle field, the [`TransferTables`](@ref) and the
projection's `rnear`, and the projection's scratch array.

The window has to hold the particles: its ball and every particle's
support lie inside the box, which the transfer checks when it runs.
"""
struct InspiralProblem{T,K,WP<:WaveProblem{T,3},TT,MO,S}
    wave::WP
    models::NTuple{K,ParticleModel{T}}
    window::Window{T}
    tables::TT
    order::Int
    motion::MO
    rnear::T
    scratch::S
end

function InspiralProblem(wave::WaveProblem{T,3}, models::Tuple, window::Window,
                         motion::Prescribed; order::Int=3, tables=nothing,
                         rnear::Real=1) where {T}
    check_order(order)
    tables === nothing && (tables = TransferTables(T, TreeWaveOLDG.order(wave.element)))
    check_tables(wave, tables)
    K = length(models)
    trajs = motion.path(zero(T))
    length(trajs) == K || throw(ArgumentError(
        "the motion moves $(length(trajs)) particles, and there are $K models"))
    models = map(m -> ParticleModel{T}(m.Q, m.ℓ, m.M), models)
    window = Window{T}(window.centre, window.R₁, window.R₂, window.m)
    scratch = projection_scratch(wave, tables)
    return InspiralProblem{T,K,typeof(wave),typeof(tables),typeof(motion),typeof(scratch)}(
        wave, models, window, tables, order, motion, T(rnear), scratch)
end

nparticles(::InspiralProblem{T,K}) where {T,K} = K

# The field's part of the state vector, and where the particles start.
fieldlength(p::InspiralProblem) = statelength(p.wave.fs)

"""
    field_state(u, p::InspiralProblem)

The field's part of the inspiral's state vector `u`, as a view: the
remainder's `u = Φ_r` and `v = Π_r`, in the layout of
`statevector(p.wave.fs)`. Pass it with `p.wave` to anything that takes a
field state.
"""
field_state(u::AbstractVector, p::InspiralProblem) = view(u, 1:fieldlength(p))

"""
    particle_state(u, p::InspiralProblem)

The particles' part of the state vector `u`: a tuple of `(z, v)` pairs of
3-vectors, one per particle.
"""
function particle_state(u::AbstractVector, p::InspiralProblem{T,K}) where {T,K}
    L = fieldlength(p)
    return ntuple(Val(K)) do k
        o = L + 6(k - 1)
        (SVector{3,T}(u[o + 1], u[o + 2], u[o + 3]),
         SVector{3,T}(u[o + 4], u[o + 5], u[o + 6]))
    end
end

function set_particle_state!(u::AbstractVector, p::InspiralProblem{T,K}, k, z,
                             v) where {T,K}
    o = fieldlength(p) + 6(k - 1)
    u[(o + 1):(o + 3)] = z
    u[(o + 4):(o + 6)] = v
    return u
end

# The particles' trajectories at time `t` of the state `u`. A prescribed
# motion gives them whatever the state.
trajectories(p::InspiralProblem{T}, u, t) where {T} = trajectories(p.motion, p, u, T(t))
trajectories(m::Prescribed, p::InspiralProblem, u, t) = m.path(t)

# The remainder of a particle field, `(1 - W)Ψ̂` for variable 1 and
# `(1 - W)𝒟Ψ̂` for variable 2, as `fill_at_nodes!` takes it. Inside the
# window's inner ball it is zero, and `Ψ̂` is not evaluated there.
struct RemainderData{T,K}
    models::NTuple{K,ParticleModel{T}}
    trajs::NTuple{K,Trajectory{T}}
    window::Window{T}
    order::Int
end

function (f::RemainderData{T,K})(x, var) where {T,K}
    X = SVector{3,T}(x)
    W = f.window(X)
    isone(W) && return zero(T)
    s = zero(T)
    for k in 1:K
        m, tr = f.models[k], f.trajs[k]
        s += var == 1 ? particle_field(m, X, tr; order=f.order) :
             particle_rate(m, X, tr; order=f.order)
    end
    return (1 - W) * s
end

"""
    inspiral_initial_state(p::InspiralProblem; t = 0)

The state vector at time `t` of the motion: the remainder of the
particle field, `Φ_r = (1 - W)Ψ̂` and `Π_r = (1 - W)𝒟Ψ̂` at the nodes, with
`Π_r` truncated as [`initial_state`](@ref) truncates `v`, and each
particle's `z` and `v` from the motion (INSPIRAL.md, "Initial data",
step 2). The total field is then the particle field everywhere.

For a particle in uniform motion this is the exact remainder at every
`t`, in an unbounded domain: `Φ_C` solves the wave equation with the
particle's source, so the remainder of a uniformly moving particle is
`(1 - W)Φ_C(z + vt)`.
"""
function inspiral_initial_state(p::InspiralProblem{T,K}; t::Real=0) where {T,K}
    trajs = trajectories(p, nothing, t)
    fs = p.wave.fs
    fill_at_nodes!(RemainderData{T,K}(p.models, trajs, p.window, p.order), fs,
                   p.wave.element)
    L = fieldlength(p)
    u = Vector{T}(undef, L + 6K)
    field = field_state(u, p)
    # The field's pages are first written by their blocks' owners, as
    # `statevector` writes them.
    TreeAMR.zerofill!(statearray(field, fs), get_backend(fs.work))
    gather!(field, fs)
    truncate_state!(field, p.wave)
    for k in 1:K
        set_particle_state!(u, p, k, trajs[k].z, trajs[k].v)
    end
    return u
end

"""
    inspiral_rhs!(du, u, p::InspiralProblem, t)

The coupled right-hand side, in the in-place form `f(du, u, p, t)`, with
the steps of INSPIRAL.md's "The right-hand side" that need no solve:

    wave_rhs! on the field part                 # du = v, dv = ∇²u
    dv -= P S                                   # the projected source
    ż = v, v̇ = a                               # each particle, a from the motion

The source is that of the particles' trajectories at `t`. It does not
truncate: [`inspiral_stage!`](@ref) does, as the integrator's limiter.
`u` is never written, and nothing is carried between calls.
"""
function inspiral_rhs!(du, u, p::InspiralProblem{T,K}, t) where {T,K}
    fu, fdu = field_state(u, p), field_state(du, p)
    wave_rhs!(fdu, fu, p.wave, t)
    trajs = trajectories(p, u, t)
    S = ParticleSource(p.models, p.window, trajs; order=p.order)
    supports = map(ParticleSupport, p.models, trajs)
    project_source!(fdu, p.wave, p.tables, S, supports;
                    ball=(p.window.centre, p.window.R₂), rnear=p.rnear,
                    scratch=p.scratch)
    particles = particle_state(u, p)
    for k in 1:K
        set_particle_state!(du, p, k, particles[k][2], trajs[k].a)
    end
    return nothing
end

"""
    inspiral_stage!(u, integrator, p::InspiralProblem, t)

The inspiral's limiter: [`truncate_state!`](@ref) on the field part of
`u`, leaving the particles alone. [`inspiral_integrator`](@ref) installs
it as the stage and the step limiter, as [`wave_integrator`](@ref)
installs [`truncate_stage!`](@ref).
"""
function inspiral_stage!(u, integrator, p::InspiralProblem, t)
    truncate_state!(field_state(u, p), p.wave)
    return nothing
end

"""
    inspiral_partition(p::InspiralProblem, u)

The block-ownership partition of the inspiral's state vector, as
[`state_partition`](@ref) gives it for the field, with the particles'
entries added to thread 1's ranges; `nothing` for a state that is not a
CPU `Array`.
"""
function inspiral_partition(p::InspiralProblem{T,K}, u) where {T,K}
    u isa Array || return nothing
    L = fieldlength(p)
    length(u) == L + 6K || throw(DimensionMismatch(
        "the state vector has $(length(u)) entries, and this problem's has " *
        "$(L + 6K)"))
    parts = [[r] for r in owner_ranges(p.wave.fs)]
    push!(parts[1], (L + 1):(L + 6K))
    return parts
end

"""
    inspiral_integrator(p::InspiralProblem, u0, t0, t1; dt, alg = RK4(),
                        alias_u0 = false, reuse = nothing)

An IMEXRungeKutta integrator for [`inspiral_rhs!`](@ref) from `t0` to `t1`
in `⌈(t1 - t0)/dt⌉` equal steps of explicit `alg`, with
[`inspiral_stage!`](@ref) as its stage and step limiter and its stage
arithmetic split by [`inspiral_partition`](@ref): the inspiral's
[`wave_integrator`](@ref). Take `dt` from `oldg_timestep(p.wave; cfl)`.
"""
function inspiral_integrator(p::InspiralProblem, u0, t0, t1; dt, alg=RK4(),
                             alias_u0::Bool=false, reuse=nothing)
    prob = IMEXProblem(inspiral_rhs!, nothing, u0, (t0, t1), p)
    return init(prob, alg; dt=dt, stage_limiter=inspiral_stage!,
                step_limiter=inspiral_stage!, partition=inspiral_partition(p, u0),
                alias_u0=alias_u0, reuse=reuse)
end

"""
    particle_forces(p::InspiralProblem, u, t)

The remainder's force on each particle, a tuple of 3-vectors: the part
of INSPIRAL.md's "Equations of motion" that the grid supplies,

    F = -(Q/γ)(∇̄Φ_r + v Π̄_r) = ∫ (∇ρ) Φ_r,h - v ∫ ρ Π_r,h,

through the adjoint pairing over the elements that meet the particle's
support ([`pairing`](@ref)), with `∇` the distributional gradient of the
discontinuous field. The analytic parts, the other particle's field and
the particle's own near field, are M6a's. In M3 it is a diagnostic: the
force does not act on the particle.
"""
function particle_forces(p::InspiralProblem{T,K}, u, t) where {T,K}
    fu = field_state(u, p)
    trajs = trajectories(p, u, t)
    return ntuple(Val(K)) do k
        m, tr = p.models[k], trajs[k]
        s = ParticleSupport(m, tr)
        ρ(x) = particle_density(m, x, tr.z, tr.v)
        ∇ρ(x) = ForwardDiff.gradient(ρ, x)
        pairing(p.wave, p.tables, fu, ∇ρ, s; var=1) -
        tr.v * pairing(p.wave, p.tables, fu, ρ, s; var=2)
    end
end

# A fold's `rf` for `ball_norm`: `Σ w |x|^p` and `Σ w` over the nodes
# within the ball, the node's position from its block's origin, the
# fold's tag.
struct BallFold{EL,D,T}
    el::EL
    centre::NTuple{D,T}
    radius::T
    p::Int
end

function (bf::BallFold{EL,D})(acc, q) where {EL,D}
    x = node_position(q.tag, q.h, q.I, bf.el)
    r² = sum(ntuple(d -> (x[d] - bf.centre[d])^2, Val(D)))
    r² < bf.radius^2 || return acc
    w = node_weight(q.I, q.h, bf.el, Val(D))
    s = zero(w)
    for v in q.u
        s += abs(v)^bf.p
    end
    return TreeAMR.combine(acc, (Sum(w * s), Sum(w * length(q.u))))
end

"""
    ball_norm(wave::WaveProblem, u, centre, radius; p = 2, vars = 1:2)
    ball_norm(prob::InspiralProblem, u, centre, radius; p = 2, vars = 1:2)

[`quadrature_norm`](@ref) restricted to the nodes within `radius` of
`centre`, normalized by the ball's quadrature volume; `p = Inf` is the
largest value there. For an [`InspiralProblem`](@ref), of the field part
of `u`.

In a periodic box the field of a charge is not periodic, so every error
norm of an inspiral run is taken in the ball that is still causally
clean of the box's faces ("The right-hand side with particles" in
`CODE.md`).
"""
function ball_norm(wave::WaveProblem{T,D}, u::AbstractVector, centre, radius;
                   p::Real=2, vars=1:2) where {T,D}
    fs = wave.fs
    forest = fs.forest
    isinf(p) || (isinteger(p) && p >= 1) || throw(ArgumentError(
        "ball_norm takes an integer p ≥ 1 or Inf, got p = $p"))
    q = isinf(p) ? 1 : Int(p)
    rf = BallFold{typeof(wave.element),D,T}(wave.element, NTuple{D,T}(Tuple(centre)),
                                            T(radius), q)
    tag = key -> block_origin(T, forest, key)
    if isinf(p)
        rfmax = (acc, pt) -> begin
            x = node_position(pt.tag, pt.h, pt.I, wave.element)
            sum(ntuple(d -> (x[d] - rf.centre[d])^2, Val(D))) < rf.radius^2 || return acc
            TreeAMR.combine(acc, Max(maximum(abs, pt.u)))
        end
        return mesh_fold(rfmax, TreeAMR.combine, Max(zero(T)), fs, u; vars=vars,
                         tag=tag).value
    end
    total, volume = mesh_fold(rf, TreeAMR.combine, (Sum(zero(T)), Sum(zero(T))), fs, u;
                              vars=vars, tag=tag)
    volume.value == 0 && return zero(T)
    return (total.value / volume.value)^inv(T(q))
end

ball_norm(prob::InspiralProblem, u::AbstractVector, centre, radius; kwargs...) =
    ball_norm(prob.wave, field_state(u, prob), centre, radius; kwargs...)

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
# `ArrayPartition` (`HISTORY.md`, "Particles"). On a device the whole
# vector is there, and the particles' tail is copied to and from the host
# in one piece.

"""
    Prescribed(path)

A motion given in advance: `path(t)` returns a tuple of
[`Trajectory`](@ref)s, one per particle, at time `t`, with `a`, `ȧ`, `ä`,
`a⃛` and `a⃜` exact. The source and the forces read the particles from it, and
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
                    tables = TransferTables(T, P), rnear = 1, continuous = true)

Everything the coupled right-hand side needs, built once: the
three-dimensional [`WaveProblem`](@ref) of the remainder, the particle
models (a tuple of [`ParticleModel`](@ref)s), the [`Window`](@ref), the
motion (a [`Prescribed`](@ref) one until M6a), the near-field `order`
(0–3) of the particle field, the [`TransferTables`](@ref) and the
projection's `rnear`, the projection's scratch array, and where the
source is averaged ([`SharedNodes`](@ref)). With `continuous = true` the
source is written through the averaging over shared nodes and the forces
are read through it ([`average_shared!`](@ref)); `false` is the transfer
and its adjoint alone, as before M4's decision ("The averaging" in
`CODE.md`).

The wave problem may be on a device, in `Float64`. The right-hand side
then runs there, but for the nested rule's points, which the host lists,
and the forces ([`particle_forces`](@ref)) are read on the host from the
few blocks around each particle ("Particles on a device" in `PLAN.md`).

The window has to hold the particles: its ball and every particle's
support lie inside the box, which the transfer checks when it runs. The
forest may be refined, as the nested boxes of [`nested_cubes`](@ref) are,
but the finest level has to lie under the window's ball and every
particle's support: the transfer measures its distances in the finest
element width, which a coarser element does not have. The window's ball
and the particles' supports at `t = 0` are checked here, and every
projection and pairing checks again, since the particles move ("Nested
boxes" in `CODE.md`).
"""
struct InspiralProblem{T,K,WP<:WaveProblem{T,3},TT,MO,S,SN<:SharedNodes{3}}
    wave::WP
    models::NTuple{K,ParticleModel{T}}
    window::Window{T}
    tables::TT
    order::Int
    motion::MO
    rnear::T
    scratch::S
    continuous::Bool
    shared::SN
end

function InspiralProblem(wave::WaveProblem{T,3}, models::Tuple, window::Window,
                         motion::Prescribed; order::Int=3, tables=nothing,
                         rnear::Real=1, continuous::Bool=true) where {T}
    check_order(order)
    wave.metric === nothing && !TreeAMR.ispatched(wave.fs.forest) || throw(ArgumentError(
        "the inspiral runs on a brick in flat space, and this wave problem has " *
        "$(wave.metric === nothing ? "a forest of patches" : "a geometry"): the " *
        "transfer measures distances in the logical coordinates and locates points " *
        "in one brick (\"Multi-patch forests\" in CODE.md)"))
    tables === nothing && (tables = TransferTables(T, TreeWaveOLDG.order(wave.element)))
    check_tables(wave, tables)
    K = length(models)
    trajs = motion.path(zero(T))
    length(trajs) == K || throw(ArgumentError(
        "the motion moves $(length(trajs)) particles, and there are $K models"))
    models = map(m -> ParticleModel{T}(m.Q, m.ℓ, m.M), models)
    window = Window{T}(window.centre, window.R₁, window.R₂, window.m)
    check_finest(wave.fs, (window.centre, window.R₂), "InspiralProblem")
    foreach((m, tr) -> check_finest(wave.fs, ParticleSupport(m, tr), "InspiralProblem"),
            models, trajs)
    scratch = projection_scratch(wave, tables)
    shared = SharedNodes(wave.fs, (window.centre, window.R₂))
    return InspiralProblem{T,K,typeof(wave),typeof(tables),typeof(motion),typeof(scratch),
                           typeof(shared)}(wave, models, window, tables, order, motion,
                                           T(rnear), scratch, continuous, shared)
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
3-vectors, one per particle, on the host, read from `u` in one copy.
"""
function particle_state(u::AbstractVector, p::InspiralProblem{T,K}) where {T,K}
    L = fieldlength(p)
    tail = Array(view(u, (L + 1):(L + 6K)))
    return ntuple(Val(K)) do k
        o = 6(k - 1)
        (SVector{3,T}(tail[o + 1], tail[o + 2], tail[o + 3]),
         SVector{3,T}(tail[o + 4], tail[o + 5], tail[o + 6]))
    end
end

# Writes each particle's `z = zs[k]` and `v = vs[k]` into `u`, in one copy.
function set_particle_states!(u::AbstractVector, p::InspiralProblem{T,K}, zs,
                              vs) where {T,K}
    tail = Vector{T}(undef, 6K)
    for k in 1:K
        o = 6(k - 1)
        tail[(o + 1):(o + 3)] = zs[k]
        tail[(o + 4):(o + 6)] = vs[k]
    end
    L = fieldlength(p)
    copyto!(view(u, (L + 1):(L + 6K)), tail)
    return u
end

# The particles' trajectories at time `t` of the state `u`. A prescribed
# motion gives them whatever the state.
trajectories(p::InspiralProblem{T}, u, t) where {T} = trajectories(p.motion, p, u, T(t))
trajectories(m::Prescribed, p::InspiralProblem, u, t) = m.path(t)

# The remainder of a particle field, `(1 - W)Ψ̂` for variable 1 and
# `(1 - W)𝒟Ψ̂` for variable 2, as `fill_at_nodes!` takes it. Inside the
# window's inner ball it is zero, and `Ψ̂` is not evaluated there.
#
# With a window `retarded`, the total field is the particles' retarded
# field inside it instead: `W_x (Φ_ret - Ψ̂_pt)` is added, with `Φ_ret`
# the point charges' retarded field ([`retarded_field`](@ref)) and `Ψ̂_pt`
# the point limit of the particle field, so that the difference is
# regular at the particles and the particles' finite-size terms through
# `order` are `Ψ̂`'s. The total field is then `W_x Φ_ref + (1 - W_x)Ψ̂`,
# with `Φ_ref = Φ_ret + Ψ̂ - Ψ̂_pt` the retarded field of the extended
# particles but for their finite-size terms beyond `order`.
struct RemainderData{T,K,R,PA,PM}
    models::NTuple{K,ParticleModel{T}}
    trajs::NTuple{K,Trajectory{T}}
    window::Window{T}
    order::Int
    retarded::R                         # nothing, or the window W_x
    path::PA
    points::PM                          # the point models
    t::T
end

function RemainderData(p, trajs, t; retarded=nothing)
    T = eltype(p.window.centre)
    points = map(point_model, p.models)
    return RemainderData(p.models, trajs, p.window, p.order, retarded, p.motion.path,
                         points, T(t))
end

function (f::RemainderData{T,K})(x, var) where {T,K}
    X = SVector{3,T}(x)
    W = f.window(X)
    s = zero(T)
    if !isone(W)
        for k in 1:K
            m, tr = f.models[k], f.trajs[k]
            s += var == 1 ? particle_field(m, X, tr; order=f.order) :
                 particle_rate(m, X, tr; order=f.order)
        end
        s *= 1 - W
    end
    return s + retarded_part(f, f.retarded, X, var)
end

retarded_part(f::RemainderData{T}, ::Nothing, X, var) where {T} = zero(T)

function retarded_part(f::RemainderData{T,K}, wx::Window, X, var) where {T,K}
    w = wx(X)
    iszero(w) && return zero(T)
    s = zero(T)
    for k in 1:K
        m, tr = f.points[k], f.trajs[k]
        # Within 10⁻⁶ of the particle the difference is `O(r³)` and below
        # roundoff, and the two fields' `1/r` would not cancel to it.
        r = X - tr.z
        dot(r, r) < T(1e-12) && continue
        Φ, Π = retarded_field(f.path, k, m.Q, X, f.t)
        s += var == 1 ? Φ - particle_field(m, X, tr; order=f.order) :
             Π - particle_rate(m, X, tr; order=f.order)
    end
    return w * s
end

"""
    inspiral_initial_state(p::InspiralProblem; t = 0, retarded = nothing)

The state vector at time `t` of the motion: the remainder of the
particle field, `Φ_r = (1 - W)Ψ̂` and `Π_r = (1 - W)𝒟Ψ̂` at the nodes, with
`Π_r` truncated as [`initial_state`](@ref) truncates `v`, and each
particle's `z` and `v` from the motion (INSPIRAL.md, "Initial data",
step 2). The total field is then the particle field everywhere.

For a particle in uniform motion this is the exact remainder at every
`t`, in an unbounded domain: `Φ_C` solves the wave equation with the
particle's source, so the remainder of a uniformly moving particle is
`(1 - W)Φ_C(z + vt)`.

With a [`Window`](@ref) `retarded = W_x`, the total field is instead
the particles' retarded field where `W_x = 1`, blended into the particle
field where it falls to zero: `W_x (Φ_ret - Ψ̂_pt)` is added, with
`Φ_ret` the point charges' [`retarded_field`](@ref) along the motion's
path and `Ψ̂_pt` the point limit of the particle field, whose difference
is regular at the particles. `W_x` must stay where every past light cone
of the path reaches ("Prescribed paths" in `CODE.md`). For a path the
particles follow, this is the exact remainder at `t` in the ball where
`W_x = 1`, as far as no signal from outside it has arrived, up to the
particles' finite-size terms beyond `order`.
"""
function inspiral_initial_state(p::InspiralProblem{T,K}; t::Real=0,
                                retarded::Union{Nothing,Window}=nothing) where {T,K}
    trajs = trajectories(p, nothing, t)
    fs = p.wave.fs
    retarded === nothing || (retarded = Window{T}(retarded.centre, retarded.R₁,
                                                  retarded.R₂, retarded.m))
    data = RemainderData(p, trajs, t; retarded)
    L = fieldlength(p)
    backend = get_backend(fs.work)
    if backend isa CPU
        fill_at_nodes!(data, fs, p.wave.element)
        u = Vector{T}(undef, L + 6K)
        field = field_state(u, p)
        # The field's pages are first written by their blocks' owners, as
        # `statevector` writes them.
        TreeAMR.zerofill!(statearray(field, fs), backend)
        gather!(field, fs)
    else
        # The particle field is host code: filled on a field set of the
        # same layout on the host, and copied.
        hfs = wave_fieldset(T, fs.forest, 2, (0, 0, 0), state_statesparity(Val(3)),
                            TreeWaveOLDG.order(p.wave.element), CPU())
        fill_at_nodes!(data, hfs, p.wave.element)
        hfield = statevector(hfs)
        gather!(hfield, hfs)
        u = allocate(backend, T, (L + 6K,))
        field = field_state(u, p)
        copyto!(field, hfield)
    end
    truncate_state!(field, p.wave)
    set_particle_states!(u, p, map(tr -> tr.z, trajs), map(tr -> tr.v, trajs))
    return u
end

"""
    inspiral_rhs!(du, u, p::InspiralProblem, t)

The coupled right-hand side, in the in-place form `f(du, u, p, t)`, with
the steps of INSPIRAL.md's "The right-hand side" that need no solve:

    wave_rhs! on the field part                 # du = v, dv = ∇²u
    dv -= A P S                                 # the projected, averaged source
    ż = v, v̇ = a                               # each particle, a from the motion

The source is that of the particles' trajectories at `t`, and `A` the
averaging over shared nodes when `p.continuous`, else the identity. It
is applied to the source alone: in the ball's blocks the source is
projected into `du`, which `wave_rhs!` has just set to `v`, averaged
there, added to `dv`, and `du = v` is written again. It does not
truncate: [`inspiral_stage!`](@ref) does, as the integrator's limiter.
`u` is never written, and nothing is carried between calls.
"""
function inspiral_rhs!(du, u, p::InspiralProblem{T,K}, t) where {T,K}
    fu, fdu = field_state(u, p), field_state(du, p)
    wave_rhs!(fdu, fu, p.wave, t)
    trajs = trajectories(p, u, t)
    S = ParticleSource(p.models, p.window, trajs; order=p.order)
    supports = map(ParticleSupport, p.models, trajs)
    ball = (p.window.centre, p.window.R₂)
    if p.continuous
        fs = p.wave.fs
        ua, dua = statearray(fu, fs), statearray(fdu, fs)
        backend = get_backend(fs.work)
        range = (ntuple(_ -> fs.forest.N, Val(3))..., length(p.shared.blocks))
        TreeAMR.launch_by_owner!(ball_zero_kernel!, backend, dua, p.shared.dev_blocks;
                                 ndrange=range)
        project_source!(fdu, p.wave, p.tables, S, supports; ball, rnear=p.rnear,
                        scratch=p.scratch, var=1)
        average_shared!(dua, p.shared, 1, Val(TreeWaveOLDG.order(p.wave.element) + 1))
        TreeAMR.launch_by_owner!(ball_add_kernel!, backend, dua, ua, p.shared.dev_blocks;
                                 ndrange=range)
        synchronize(backend)
    else
        project_source!(fdu, p.wave, p.tables, S, supports; ball, rnear=p.rnear,
                        scratch=p.scratch)
    end
    particles = particle_state(u, p)
    set_particle_states!(du, p, map(last, particles), map(tr -> tr.a, trajs))
    return nothing
end

# The ball's blocks' `du = v`, which serves as the source's scratch: zeroed
# before the projection, and after it the averaged source added to `dv`
# and `du = v` written again. One work item per node of a ball's block.
@kernel function ball_zero_kernel!(dua, @Const(blocks))
    I = @index(Global, NTuple)
    @inbounds b = Int(blocks[I[end]])
    @inbounds dua[Base.front(I)..., 1, b] = zero(eltype(dua))
end

@kernel function ball_add_kernel!(dua, @Const(ua), @Const(blocks))
    I = @index(Global, NTuple)
    J = Base.front(I)
    @inbounds b = Int(blocks[I[end]])
    @inbounds dua[J..., 2, b] += dua[J..., 1, b]
    @inbounds dua[J..., 1, b] = ua[J..., 2, b]
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
force does not act on the particle. With `continuous = p.continuous`,
the default, it is read from the field made continuous by the averaging
the source is written through ([`pairing`](@ref)); `false` and `true`
read it the other way, for comparison.
"""
function particle_forces(p::InspiralProblem{T,K}, u, t;
                         continuous::Bool=p.continuous) where {T,K}
    fu = field_state(u, p)
    trajs = trajectories(p, u, t)
    return ntuple(Val(K)) do k
        m, tr = p.models[k], trajs[k]
        s = ParticleSupport(m, tr)
        ρ(x) = particle_density(m, x, tr.z, tr.v)
        ∇ρ(x) = ForwardDiff.gradient(ρ, x)
        pairing(p.wave, p.tables, fu, ∇ρ, s; var=1, continuous) -
        tr.v * pairing(p.wave, p.tables, fu, ρ, s; var=2, continuous)
    end
end

# A fold's `rf` for `ball_norm`: `Σ w |x|^p` and `Σ w` over the nodes
# within the ball and outside the hole, at the node's position, which
# TreeAMR gives an element set's fold.
struct BallFold{EL,D,T}
    el::EL
    centre::NTuple{D,T}
    radius::T
    hcentre::NTuple{D,T}
    hradius::T                          # zero: no hole
    p::Int
end

@inline function in_shell(bf::BallFold{EL,D}, x) where {EL,D}
    r² = sum(ntuple(d -> (x[d] - bf.centre[d])^2, Val(D)))
    h² = sum(ntuple(d -> (x[d] - bf.hcentre[d])^2, Val(D)))
    return r² < bf.radius^2 && h² >= bf.hradius^2
end

function (bf::BallFold{EL,D})(acc, q) where {EL,D}
    in_shell(bf, q.x) || return acc
    w = node_weight(q.I, q.h, bf.el, Val(D))
    s = zero(w)
    for v in q.u
        s += abs(v)^bf.p
    end
    return TreeAMR.combine(acc, (Sum(w * s), Sum(w * length(q.u))))
end

"""
    ball_norm(wave::WaveProblem, u, centre, radius; p = 2, vars = 1:2,
              hole = nothing)
    ball_norm(prob::InspiralProblem, u, centre, radius; p = 2, vars = 1:2,
              hole = nothing)

[`quadrature_norm`](@ref) restricted to the nodes within `radius` of
`centre`, normalized by the ball's quadrature volume; `p = Inf` is the
largest value there. For an [`InspiralProblem`](@ref), of the field part
of `u`. `hole = (centre, radius)` leaves out the nodes within that
ball, as about a particle whose near field the norm should not see.

In a periodic box the field of a charge is not periodic, so every error
norm of an inspiral run is taken in the ball that is still causally
clean of the box's faces ("The right-hand side with particles" in
`CODE.md`).
"""
function ball_norm(wave::WaveProblem{T,D}, u::AbstractVector, centre, radius;
                   p::Real=2, vars=1:2, hole=nothing) where {T,D}
    fs = wave.fs
    isinf(p) || (isinteger(p) && p >= 1) || throw(ArgumentError(
        "ball_norm takes an integer p ≥ 1 or Inf, got p = $p"))
    q = isinf(p) ? 1 : Int(p)
    hc, hr = hole === nothing ? (Tuple(centre), zero(T)) : (Tuple(hole[1]), hole[2])
    rf = BallFold{typeof(wave.element),D,T}(wave.element, NTuple{D,T}(Tuple(centre)),
                                            T(radius), NTuple{D,T}(hc), T(hr), q)
    if isinf(p)
        rfmax = (acc, pt) -> begin
            in_shell(rf, pt.x) || return acc
            TreeAMR.combine(acc, Max(maximum(abs, pt.u)))
        end
        return mesh_fold(rfmax, TreeAMR.combine, Max(zero(T)), fs, u; vars=vars).value
    end
    total, volume = mesh_fold(rf, TreeAMR.combine, (Sum(zero(T)), Sum(zero(T))), fs, u;
                              vars=vars)
    volume.value == 0 && return zero(T)
    return (total.value / volume.value)^inv(T(q))
end

ball_norm(prob::InspiralProblem, u::AbstractVector, centre, radius; kwargs...) =
    ball_norm(prob.wave, field_state(u, prob), centre, radius; kwargs...)

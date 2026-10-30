# The analytic particle of INSPIRAL.md: a scalar charge `Q` of physical
# size `ℓ`, its field `Ψ` (the boosted static field plus the near-field
# terms of orders 1–3), the field's time derivative `𝒟Ψ` along the
# particle's motion, the residual of its wave equation, and the window
# that cuts it off far from the particles. See "The particle", "The near
# field" and "The window" in `CODE.md`.
#
# The closed forms (the profile, the radial potentials, the near-field
# terms and their Laplacians, the self-force coefficients) are generated
# into `particle_forms.jl` from the Wolfram Language derivations in
# `derivations/`. This file is the hand-written part on top: the
# particle's state, the boost, and the time derivatives, which are dual
# numbers along the trajectory's Taylor polynomial.
#
# Until the lab-slice re-expansion (INSPIRAL-PLAN.md, M5), the near-field
# terms are the rest-frame ones evaluated at the rest-frame position
# `y = Λ(v)(x - z)` with the proper acceleration `α′ = γ²Λ(v)a`.

"""
    ParticleModel(; Q, ℓ, M)

A scalar particle of charge `Q`, size `ℓ` and physical (dressed) mass
`M`. Its bare mass is `M₀ = M - δm(ℓ)`, with `δm = 693Q²/4199πℓ` the
self-inertia of its own near field.

The constructor refuses `M₀ ≤ 0`, a particle smaller than the model's
classical radius `693Q²/4199πM ≈ 0.0525 Q²/M`. Below it the coupled
particle and field have runaway solutions, as an extended classical
charge does (INSPIRAL.md, "Why the near field must be analytic").
"""
struct ParticleModel{T}
    Q::T
    ℓ::T
    M::T
    M₀::T

    function ParticleModel{T}(Q, ℓ, M) where {T}
        ℓ > 0 || throw(ArgumentError("the particle size ℓ must be positive, got ℓ = $ℓ"))
        M₀ = M - self_inertia(T(Q), T(ℓ))
        M₀ > 0 || throw(ArgumentError(
            "the bare mass M₀ = M - δm(ℓ) = $M₀ is not positive: the particle size " *
            "ℓ = $ℓ is below the model's classical radius 693Q²/4199πM = " *
            "$(self_inertia(T(Q), one(T)) / M), where the coupled particle and field " *
            "have runaway solutions. Choose a larger ℓ, a larger mass or a smaller " *
            "charge (INSPIRAL.md, \"Why the near field must be analytic\")"))
        return new{T}(Q, ℓ, M, M₀)
    end
end

function ParticleModel(; Q, ℓ, M)
    T = float(promote_type(typeof(Q), typeof(ℓ), typeof(M)))
    return ParticleModel{T}(Q, ℓ, M)
end

self_inertia(m::ParticleModel) = self_inertia(m.Q, m.ℓ)

"""
    Trajectory(; z, v, a = 0, ȧ = 0, ä = 0, a⃛ = 0)

A particle's position, velocity, acceleration and the acceleration's
first three time derivatives, as `SVector{3}`s: the Taylor polynomial in `t`
that the time derivatives of [`particle_field`](@ref) act on.
[`trajectory_at`](@ref) evaluates it at a time `t`, which may be a dual
number. A `Trajectory` of an `isbits` type is `isbits`.
"""
struct Trajectory{T}
    z::SVector{3,T}
    v::SVector{3,T}
    a::SVector{3,T}
    ȧ::SVector{3,T}
    ä::SVector{3,T}
    a⃛::SVector{3,T}
end

function Trajectory(; z, v, a=zero(z), ȧ=zero(z), ä=zero(z), a⃛=zero(z))
    T = float(promote_type(eltype(z), eltype(v), eltype(a), eltype(ȧ), eltype(ä),
                           eltype(a⃛)))
    return Trajectory{T}(z, v, a, ȧ, ä, a⃛)
end

"""
    trajectory_at(traj, t)

The trajectory's Taylor polynomial, and its derivatives, at time `t`
from the moment `traj` describes.
"""
@inline function trajectory_at(traj::Trajectory, t)
    (; z, v, a, ȧ, ä, a⃛) = traj
    return Trajectory(z + t * (v + t * (a / 2 + t * (ȧ / 6 + t * (ä / 24 + t * a⃛ / 120)))),
                      v + t * (a + t * (ȧ / 2 + t * (ä / 6 + t * a⃛ / 24))),
                      a + t * (ȧ + t * (ä / 2 + t * a⃛ / 6)),
                      ȧ + t * (ä + t * a⃛ / 2),
                      ä + t * a⃛,
                      a⃛ + zero(t) * a⃛)
end

# The boost: `Λ(v) r = r + (γ - 1)(v̂·r)v̂`, with `(γ - 1)/v² = γ²/(γ + 1)`,
# which needs no `v̂` at rest.
@inline function lorentz_factor(v)
    return inv(sqrt(1 - dot(v, v)))
end
@inline function boost(v, r)
    γ = lorentz_factor(v)
    return r + (γ^2 / (γ + 1) * dot(v, r)) * v
end

"""
    particle_density(m, x, z, v)

The particle's density `ρ(x) = ρ₀(|Λ(v)(x - z)|)` for a particle at `z`
moving with velocity `v`: the rest-frame profile carried along with the
particle, a Lorentz scalar (INSPIRAL.md, "Fields and particles").
"""
@inline function particle_density(m::ParticleModel, x, z, v)
    y = boost(v, x - z)
    return m.Q * _rho0(dot(y, y), m.ℓ)
end

"""
    boosted_static(m, x, z, v)

`Φ_C`, the field of a particle moving uniformly through `z` with velocity
`v`: the static field `Φ₀` at the rest-frame position `y = Λ(v)(x - z)`.
It satisfies `∇²Φ_C - (v·∇)²Φ_C = ρ` exactly.
"""
@inline function boosted_static(m::ParticleModel, x, z, v)
    y = boost(v, x - z)
    return m.Q * _phi0(dot(y, y), m.ℓ)
end

"""
    near_field(m, Val(N), x, traj)

The near-field term of order `N = 1, 2, 3` of a particle on `traj`
(INSPIRAL.md, "The near field of an accelerated particle"): the
rest-frame term at `y = Λ(v)(x - z)`, with the proper acceleration
`α′ = γ²Λ(v)a` in place of `a`, and `ä` as given. At rest this is the
time-symmetric field of the extended particle, order by order. Moving,
it lacks the lab-slice re-expansion, which is `INSPIRAL-PLAN.md`'s M5.
"""
@inline function near_field(m::ParticleModel, ::Val{N}, x, traj::Trajectory) where {N}
    y, α′ = rest_frame(traj, x)
    return near_field_term(Val(N), m.Q, m.ℓ, y, α′, traj.ä)
end

@inline function rest_frame(traj::Trajectory, x)
    (; z, v, a) = traj
    return boost(v, x - z), lorentz_factor(v)^2 * boost(v, a)
end

function check_order(order)
    0 <= order <= 3 || throw(ArgumentError(
        "the near field is carried to order 0, 1, 2 or 3, got order = $order"))
    return nothing
end

"""
    particle_field(m, x, traj; order = 3)

`Ψ`: the boosted static field plus the near-field terms of orders `1` to
`order` (INSPIRAL.md, "The particle field"). `order = 0` is the particle
paper's boosted static field alone.
"""
@inline function particle_field(m::ParticleModel, x, traj::Trajectory; order::Int=3)
    check_order(order)
    Ψ = boosted_static(m, x, traj.z, traj.v)
    order >= 1 && (Ψ += near_field(m, Val(1), x, traj))
    order >= 2 && (Ψ += near_field(m, Val(2), x, traj))
    order >= 3 && (Ψ += near_field(m, Val(3), x, traj))
    return Ψ
end

"""
    particle_rate(m, x, traj; order = 3)

`𝒟Ψ`, the time derivative of [`particle_field`](@ref) along the
trajectory's Taylor polynomial: a dual-number derivative in `t`.
"""
function particle_rate(m::ParticleModel, x, traj::Trajectory{T}; order::Int=3) where {T}
    return ForwardDiff.derivative(t -> particle_field(m, x, trajectory_at(traj, t); order),
                                  zero(T))
end

@inline second_derivative(f, t) =
    ForwardDiff.derivative(s -> ForwardDiff.derivative(f, s), t)

"""
    source_bracket(m, x, traj; order = 3)

The residual of the particle field's wave equation along the motion,
`𝒟²Ψ - ∇²Ψ + ρ`, for one particle: the bracket of the effective source
in INSPIRAL.md, "The remainder equations". See
[`ParticleSource`](@ref) for how it is assembled.
"""
function source_bracket(m::ParticleModel, x, traj::Trajectory; order::Int=3)
    check_order(order)
    return source_terms(ParticleJets(m, traj), order, x)[2]
end

# The tags of the source's nested dual numbers, seeded in `lift`: their
# product carries the acceleration's part of `𝒟²`, and a second outer
# partial a directional derivative.
struct SourceOuterTag end
struct SourceInnerTag end

# A field that depends on `x` through `q = |y|²`, `α′·y`, `|α′|²` and
# `ä·y`, and on `t` through them along the motion. Of each invariant `I`
# the bracket needs its value, its rate `İᵤ` in uniform motion (`z`
# moving with `v`, and `v`, `a`, `ä` held), and the acceleration's parts
# `δİ = İ - İᵤ` and `δÏ = Ï - Ïᵤ`, each formed directly and not as a
# difference. With them,
#
#     𝒟²G - (v·∂_z)²G = Σᵢ G_i δÏᵢ + Σᵢⱼ G_ij δİᵢ (2İᵤ + δİ)ⱼ,
#
# the coefficient of `ε₁ε₂` in `G(I + ε₁ δİ + ε₂ (2İᵤ + δİ) + ε₁ε₂ δÏ)`.
# `d` holds directional derivatives `∇I·n` to carry in further outer
# partials.
@inline function lift(I, İᵤ, δİ, δÏ, d::Tuple)
    I, İᵤ, δİ, δÏ = promote(I, İᵤ, δİ, δÏ)
    Inner = ForwardDiff.Dual{SourceInnerTag}
    return ForwardDiff.Dual{SourceOuterTag}(Inner(I, 2İᵤ + δİ), Inner(δİ, δÏ),
                                           map(dᵢ -> Inner(oftype(I, dᵢ), zero(I)), d)...)
end

"""
    ParticleJets(m, traj)

What [`ParticleSource`](@ref) needs of one particle, at the moment
`traj` describes, that does not depend on the point: the invariants'
`x`-independent factors and their first two time derivatives. With
`r = x - z`, `y = Λ(v)r` and `M = Λ(v)² = I + γ²vvᵀ`,

    |y|² = rᵀMr,   α′·y = k·r with k = γ²Ma,   |α′|² = γ⁴aᵀMa,
    ä·y = e·r with e = Λ(v)ä,

and `ṙ = -v`, `r̈ = -a`.
"""
struct ParticleJets{T}
    Q::T
    ℓ::T
    z::SVector{3,T}
    M::NTuple{3,SMatrix{3,3,T,9}}   # M, Ṁ, M̈
    uq::SVector{3,T}                # q̇ᵤ = uq·r
    bq::SVector{3,T}                # δq̈ = bq·r + rᵀM̈r
    k::NTuple{3,SVector{3,T}}       # k, k̇, k̈
    k̇ᵤ::T                           # d(α′·y)/dt in uniform motion
    k̈₀::T                           # δ d²(α′·y)/dt² = k̈·r + k̈₀
    aa::NTuple{3,T}                 # |α′|² and its derivatives
    e::NTuple{3,SVector{3,T}}       # e, ė, ë
    ėᵤ::T                           # the same for ä·y
    ë₀::T
end

function ParticleJets(m::ParticleModel{T}, traj::Trajectory{T}) where {T}
    jet(f) = (g = t -> f(trajectory_at(traj, t));
              (g(zero(T)), ForwardDiff.derivative(g, zero(T)), second_derivative(g, zero(T))))
    γ²(v) = inv(1 - dot(v, v))
    Mof(v) = one(SMatrix{3,3,eltype(v)}) + γ²(v) * v * v'
    M = jet(tr -> Mof(tr.v))
    k = jet(tr -> γ²(tr.v) * (Mof(tr.v) * tr.a))
    aa = jet(tr -> γ²(tr.v)^2 * dot(tr.a, Mof(tr.v) * tr.a))
    e = jet(tr -> boost(tr.v, tr.ä))
    (; z, v, a) = traj
    # q = rᵀMr: q̇ = 2ṙᵀMr + rᵀṀr, q̈ = 2r̈ᵀMr + 2ṙᵀMṙ + 4ṙᵀṀr + rᵀM̈r, of
    # which uniform motion keeps -2vᵀMr and 2vᵀMv.
    uq = -2 * (M[1] * v)
    bq = -2 * (M[1] * a) - 4 * (M[2] * v)
    linear(k) = (-dot(k[1], v), -2dot(k[2], v) - dot(k[1], a))
    k̇ᵤ, k̈₀ = linear(k)
    ėᵤ, ë₀ = linear(e)
    return ParticleJets{T}(m.Q, m.ℓ, z, M, uq, bq, k, k̇ᵤ, k̈₀, aa, e, ėᵤ, ë₀)
end

# Ψ as a function of the invariants.
@inline function invariant_field(p::ParticleJets, order, q, ay, aa, äy)
    Ψ = p.Q * _phi0(q, p.ℓ)
    order >= 1 && (Ψ += _near_field_term(Val(1), p.Q, p.ℓ, q, ay, aa, äy))
    order >= 2 && (Ψ += _near_field_term(Val(2), p.Q, p.ℓ, q, ay, aa, äy))
    order >= 3 && (Ψ += _near_field_term(Val(3), p.Q, p.ℓ, q, ay, aa, äy))
    return Ψ
end

# ∇²_y of the near-field terms, as a function of the invariants.
@inline function invariant_laplacian(p::ParticleJets, order, q, ay, aa, äy)
    L = zero(q)
    order >= 1 && (L += _near_field_laplacian(Val(1), p.Q, p.ℓ, q, ay, aa, äy))
    order >= 2 && (L += _near_field_laplacian(Val(2), p.Q, p.ℓ, q, ay, aa, äy))
    order >= 3 && (L += _near_field_laplacian(Val(3), p.Q, p.ℓ, q, ay, aa, äy))
    return L
end

# One particle's `(Ψ, 𝒟²Ψ - ∇²Ψ + ρ, n·∇Ψ)` at `x`, with `n·∇Ψ` only
# where `n` is given.
#
# `ρ` cancels against `□_vΦ_C = ∇²Φ_C - (v·∇)²Φ_C` analytically, and
# each near-field term has `∇²_x = ∇²_y + γ²(v·∇_y)²` with
# `γ²(v·∇_y)² = (v·∂_z)²`, so the bracket is `𝒟²Ψ - (v·∂_z)²Ψ`, which
# `lift` forms from the acceleration's parts alone, minus the generated
# `∇²_y` of the near-field terms.
@inline function source_terms(p::ParticleJets, order, x, n...)
    r = x - p.z
    M, M′, M″ = p.M
    Mr = M * r
    q = lift(dot(r, Mr), dot(p.uq, r), dot(r, M′ * r), dot(p.bq, r) + dot(r, M″ * r),
             map(nᵢ -> 2dot(nᵢ, Mr), n))
    k, k′, k″ = p.k
    ay = lift(dot(k, r), p.k̇ᵤ, dot(k′, r), dot(k″, r) + p.k̈₀, map(nᵢ -> dot(nᵢ, k), n))
    aa = lift(p.aa[1], zero(p.aa[1]), p.aa[2], p.aa[3], map(nᵢ -> zero(p.aa[1]), n))
    e, e′, e″ = p.e
    äy = lift(dot(e, r), p.ėᵤ, dot(e′, r), dot(e″, r) + p.ë₀, map(nᵢ -> dot(nᵢ, e), n))
    G = invariant_field(p, order, q, ay, aa, äy)
    value = ForwardDiff.value
    partial = ForwardDiff.partials
    ∇²y = invariant_laplacian(p, order, value(value(q)), value(value(ay)),
                              value(value(aa)), value(value(äy)))
    return (value(value(G)), partial(partial(G, 1), 1) - ∇²y,
            map(i -> value(partial(G, i + 1)), ntuple(identity, length(n)))...)
end

"""
    Window(centre, R₁, R₂; continuity)

`W(x) = w(|x - centre|)`: one for `r ≤ R₁`, zero for `r ≥ R₂`, and in
between `1 - S_m((r - R₁)/(R₂ - R₁))`, with `S_m` the polynomial step of
degree `2m + 1` whose first `m` derivatives vanish at both ends, so that
`W` is `C^m` and no smoother. INSPIRAL.md asks for `C^{P+1}` or better:
pass `continuity = P + 1`. See "The window" in `CODE.md`.

`window_terms(w, x)` gives `W`, `∇W` and `∇²W` in closed form, and
`w(x)` gives `W`.
"""
struct Window{T}
    centre::SVector{3,T}
    R₁::T
    R₂::T
    m::Int
    c::T                        # (2m+1)!/(m!)², S_m′(ξ) = c ξ^m (1-ξ)^m

    function Window{T}(centre, R₁, R₂, m) where {T}
        0 < R₁ < R₂ || throw(ArgumentError(
            "the window needs 0 < R₁ < R₂, got R₁ = $R₁, R₂ = $R₂"))
        m >= 0 || throw(ArgumentError("the continuity must be at least 0, got $m"))
        return new{T}(centre, R₁, R₂, m, T((2m + 1) * binomial(2m, m)))
    end
end

function Window(centre, R₁, R₂; continuity::Integer)
    T = float(promote_type(eltype(centre), typeof(R₁), typeof(R₂)))
    return Window{T}(centre, R₁, R₂, continuity)
end

# S_m(ξ) = ξ^{m+1} Σₖ C(m+k, k) (1-ξ)^k for 0 ≤ ξ ≤ 1, summed upward with
# C(m+k+1, k+1) = C(m+k, k) (m+k+1)/(k+1), exact in integers.
function smoothstep(m, ξ)
    y = 1 - ξ
    yᵏ = one(ξ)
    C = 1
    acc = zero(ξ)
    for k in 0:m
        acc += C * yᵏ
        yᵏ *= y
        C = C * (m + k + 1) ÷ (k + 1)
    end
    return ξ^(m + 1) * acc
end

function (w::Window)(x)
    r² = dot(x - w.centre, x - w.centre)
    r² <= w.R₁^2 && return one(r²)
    r² >= w.R₂^2 && return zero(r²)
    return 1 - smoothstep(w.m, (sqrt(r²) - w.R₁) / (w.R₂ - w.R₁))
end

"""
    window_terms(w, x)

`(W, ∇W, ∇²W)` at `x`, in closed form.
"""
function window_terms(w::Window, x)
    d = x - w.centre
    r² = dot(d, d)
    zero3 = zero(d) * zero(r²)
    r² <= w.R₁^2 && return (one(r²), zero3, zero(r²))
    r² >= w.R₂^2 && return (zero(r²), zero3, zero(r²))
    r = sqrt(r²)
    Δ = w.R₂ - w.R₁
    ξ = (r - w.R₁) / Δ
    m = w.m
    S = smoothstep(m, ξ)
    p = m == 0 ? zero(ξ) : (ξ * (1 - ξ))^(m - 1)
    S′ = m == 0 ? w.c : w.c * p * (ξ * (1 - ξ))
    S″ = w.c * m * p * (1 - 2ξ)
    return (1 - S, -(S′ / (Δ * r)) * d, -(S″ / Δ^2 + 2S′ / (Δ * r)))
end

"""
    ParticleSource(models, w, trajs; order = 3)

The effective source of the remainder equations (INSPIRAL.md, "The
remainder equations") for particles `models[i]` on `trajs[i]`, as a
callable `S(x)`:

    S = W Σₐ [𝒟²Ψₐ - ∇²Ψₐ + ρₐ] - 2∇W·∇Ψ̂ - Ψ̂ ∇²W,    Ψ̂ = Σₐ Ψₐ.

It vanishes outside the window's outer radius. It is `isbits` for an
`isbits` `T`, so that it can be passed to a kernel, and holds each
particle's [`ParticleJets`](@ref), which do not depend on `x`.

Near a particle `∇²Ψ`, `(v·∇)²Ψ` and `ρ` are each of size `Q/ℓ³`, and
the bracket is assembled without subtracting them (CODE.md, "The
split"). `Ψ` depends on `x` and `t` through four invariants, `|y|²`,
`α′·y`, `|α′|²` and `ä·y`. `ρ` cancels against `□_vΦ_C` analytically and
`∇²_x` of each near-field term is `∇²_y + (v·∂_z)²`, so the bracket is
`𝒟²Ψ - (v·∂_z)²Ψ` minus the generated `∇²_y` of the near-field terms,
and `𝒟²Ψ - (v·∂_z)²Ψ` is formed from the acceleration's parts of the
invariants' rates, which are formed directly. One evaluation of `Ψ` on
nested dual numbers gives it, and in the window's transition also `Ψ`
and `∇W·∇Ψ`.
"""
struct ParticleSource{T,K}
    particles::NTuple{K,ParticleJets{T}}
    window::Window{T}
    order::Int
end

function ParticleSource(models::Tuple, w::Window, trajs::Tuple; order::Int=3)
    length(models) == length(trajs) || throw(ArgumentError(
        "got $(length(models)) particle models and $(length(trajs)) trajectories"))
    check_order(order)
    return ParticleSource(map(ParticleJets, models, trajs), w, order)
end

function (s::ParticleSource)(x)
    W, ∇W, ∇²W = window_terms(s.window, x)
    S = zero(W)
    iszero(W) && return S
    if iszero(∇W) && iszero(∇²W)
        for p in s.particles
            S += W * source_terms(p, s.order, x)[2]
        end
    else
        for p in s.particles
            Ψ, bracket, ∇WΨ = source_terms(p, s.order, x, ∇W)
            S += W * bracket - 2∇WΨ - Ψ * ∇²W
        end
    end
    return S
end

"""
    particle_source(models, w, x, trajs; order = 3)

The effective source at `x`, `ParticleSource(models, w, trajs; order)(x)`;
see [`ParticleSource`](@ref), which is the form to evaluate at many
points.
"""
particle_source(models::Tuple, w::Window, x, trajs::Tuple; order::Int=3) =
    ParticleSource(models, w, trajs; order)(x)

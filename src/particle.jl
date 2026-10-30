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

# The trajectory with every derivative of the velocity scaled by `s`:
# `s = 0` is uniform motion, `s = 1` the actual one.
@inline deform(traj::Trajectory, s) =
    Trajectory(traj.z + zero(s) * traj.z, traj.v + zero(s) * traj.v,
               s * traj.a, s * traj.ȧ, s * traj.ä, s * traj.a⃛)

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
in INSPIRAL.md, "The remainder equations".

Near the particle `∇²Ψ`, `(v·∇)²Ψ` and `ρ` are each of size `Q/ℓ³`, and
the bracket is assembled without subtracting them:

- `ρ` cancels against `□_vΦ_C = ∇²Φ_C - (v·∇)²Φ_C` analytically, which
  leaves `A := 𝒟²Φ_C - (v·∂_z)²Φ_C`, the part of `Φ_C`'s second time
  derivative that the acceleration causes. `A` is computed as
  derivatives in a deformation parameter `s` of the motion with every
  derivative of the velocity scaled by `s`: `∂ₜ²Φ_C` along it is
  quadratic in `s`, and `A` is its first derivative plus half its second
  at `s = 0`, not the difference of two second time derivatives.
- Each near-field term contributes `𝒟²Φ⁽ᴺ⁾ - ∇²Φ⁽ᴺ⁾`, with
  `∇²_x = ∇²_y + γ²(v·∇_y)²` and `∇²_y` the generated closed form from
  the hierarchy.
"""
function source_bracket(m::ParticleModel, x, traj::Trajectory{T}; order::Int=3) where {T}
    check_order(order)
    # ∂ₜ²Φ_C along the deformed motion, as a function of s.
    h(s) = second_derivative(
        t -> (tr = trajectory_at(deform(traj, s), t); boosted_static(m, x, tr.z, tr.v)),
        zero(s))
    A = ForwardDiff.derivative(h, zero(T)) + second_derivative(h, zero(T)) / 2
    order >= 1 && (A += near_field_residual(m, Val(1), x, traj))
    order >= 2 && (A += near_field_residual(m, Val(2), x, traj))
    order >= 3 && (A += near_field_residual(m, Val(3), x, traj))
    return A
end

# `𝒟²Φ⁽ᴺ⁾ - ∇²Φ⁽ᴺ⁾` for one near-field term.
function near_field_residual(m::ParticleModel, ::Val{N}, x, traj::Trajectory{T}) where {N,T}
    D² = second_derivative(t -> near_field(m, Val(N), x, trajectory_at(traj, t)), zero(T))
    y, α′ = rest_frame(traj, x)
    w = lorentz_factor(traj.v) * traj.v
    ∇²y = near_field_laplacian(Val(N), m.Q, m.ℓ, y, α′, traj.ä)
    vv = second_derivative(ε -> near_field_term(Val(N), m.Q, m.ℓ, y + ε * w, α′, traj.ä),
                           zero(T))
    return D² - ∇²y - vv
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

# S_m(ξ) = ξ^{m+1} Σₖ C(m+k, k) (1-ξ)^k for 0 ≤ ξ ≤ 1.
function smoothstep(m, ξ)
    acc = zero(ξ)
    for k in m:-1:0
        acc = acc * (1 - ξ) + binomial(m + k, k)
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
    S′ = w.c * (ξ * (1 - ξ))^m
    S″ = m == 0 ? zero(ξ) : w.c * m * (ξ * (1 - ξ))^(m - 1) * (1 - 2ξ)
    return (1 - S, -(S′ / (Δ * r)) * d, -(S″ / Δ^2 + 2S′ / (Δ * r)))
end

"""
    particle_source(models, w, x, trajs; order = 3)

The effective source of the remainder equations (INSPIRAL.md, "The
remainder equations") at `x`, for particles `models[i]` on `trajs[i]`:

    S = W Σₐ [𝒟²Ψₐ - ∇²Ψₐ + ρₐ] - 2∇W·∇Ψ̂ - Ψ̂ ∇²W,    Ψ̂ = Σₐ Ψₐ

with each bracket from [`source_bracket`](@ref). It vanishes outside the
window's outer radius.
"""
function particle_source(models::Tuple, w::Window, x, trajs::Tuple; order::Int=3)
    length(models) == length(trajs) || throw(ArgumentError(
        "got $(length(models)) particle models and $(length(trajs)) trajectories"))
    W, ∇W, ∇²W = window_terms(w, x)
    S = zero(W)
    iszero(W) && return S
    for (m, traj) in zip(models, trajs)
        S += W * source_bracket(m, x, traj; order)
        if !iszero(∇²W) || !iszero(∇W)
            Ψ = ξ -> particle_field(m, ξ, traj; order)
            S -= 2dot(∇W, ForwardDiff.gradient(Ψ, x)) + Ψ(x) * ∇²W
        end
    end
    return S
end

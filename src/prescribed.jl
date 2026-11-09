# Prescribed paths with known accelerations, INSPIRAL-PLAN.md's M4
# (INSPIRAL.md, "Validation", step 4): a particle in hyperbolic motion,
# whose retarded field at `t = 0` is time-symmetric, and a particle on a
# circle at small speed, whose remainder's force is radiation reaction.
# Also the exact retarded field of a point charge on any path, by
# root-finding for the retarded time, which the hyperbolic run's initial
# data and its reference take. See "Prescribed paths" in `CODE.md`.

# A circle about `centre` in the plane spanned by `e₁` and `e₂`, of radius
# `r`, at angular velocity `Ω`, at angle `φ₀` at `t = 0`.
struct CirclePath{T}
    centre::SVector{3,T}
    e₁::SVector{3,T}
    e₂::SVector{3,T}
    r::T
    Ω::T
    φ₀::T
end

function (p::CirclePath)(t)
    φ = p.φ₀ + p.Ω * t
    c, s = cos(φ), sin(φ)
    ρ = c * p.e₁ + s * p.e₂             # outward
    τ = -s * p.e₁ + c * p.e₂            # along the motion
    r, Ω = p.r, p.Ω
    return (Trajectory(; z=p.centre + r * ρ, v=r * Ω * τ, a=-r * Ω^2 * ρ,
                       ȧ=-r * Ω^3 * τ, ä=r * Ω^4 * ρ, a⃛=r * Ω^5 * τ,
                       a⃜=-r * Ω^6 * ρ),)
end

"""
    circular_motion(centre, radius, speed; phase = 0, normal = (0, 0, 1))

A [`Prescribed`](@ref) motion of one particle on a circle about `centre`
of the given `radius` at the given `speed`, in the plane normal to
`normal`, counterclockwise about it, at angle `phase` at `t = 0` from
the plane's first axis (`x` for the default normal). Its acceleration
and the acceleration's derivatives are exact:
`ȧ = -Ω²v` with `Ω = speed/radius`, so that the radiation-reaction force
of a point charge, `(Q²/12π)ȧ`, opposes the motion.
"""
function circular_motion(centre, radius::Real, speed::Real; phase::Real=0,
                         normal=(0, 0, 1))
    T = float(promote_type(eltype(centre), typeof(radius), typeof(speed), typeof(phase)))
    0 < speed < 1 || throw(ArgumentError("the speed must lie in (0, 1), got $speed"))
    radius > 0 || throw(ArgumentError("the radius must be positive, got $radius"))
    n = SVector{3,T}(normal)
    n = n / sqrt(dot(n, n))
    # The plane's first axis: `x` projected onto it, or `y` if `n ∥ x`.
    seed = abs(n[1]) < T(0.9) ? SVector{3,T}(1, 0, 0) : SVector{3,T}(0, 1, 0)
    e₁ = seed - dot(seed, n) * n
    e₁ = e₁ / sqrt(dot(e₁, e₁))
    e₂ = SVector(n[2] * e₁[3] - n[3] * e₁[2], n[3] * e₁[1] - n[1] * e₁[3],
                 n[1] * e₁[2] - n[2] * e₁[1])
    return Prescribed(CirclePath{T}(SVector{3,T}(centre), e₁, e₂, T(radius),
                                    T(speed) / T(radius), T(phase)))
end

# Hyperbolic motion along the unit vector `n` through `z₀`, at rest there at
# `t = 0`, with proper acceleration `g`.
struct HyperbolicPath{T}
    z₀::SVector{3,T}
    n::SVector{3,T}
    g::T
end

# The distance travelled, `X(t) = (√(1 + g²t²) - 1)/g`, and its first six
# derivatives, in closed form.
@inline function hyperbolic_jet(g, t)
    s² = 1 + (g * t)^2
    s = sqrt(s²)
    X = (g * t)^2 / (g * (s + 1))       # (s - 1)/g without cancellation
    X1 = g * t / s
    X2 = g / (s * s²)
    X3 = -3g^3 * t / (s * s²^2)
    X4 = -3g^3 * (1 - 4(g * t)^2) / (s * s²^3)
    X5 = 15g^5 * t * (3 - 4(g * t)^2) / (s * s²^4)
    X6 = 45g^5 * (1 - 12(g * t)^2 + 8(g * t)^4) / (s * s²^5)
    return X, X1, X2, X3, X4, X5, X6
end

function (p::HyperbolicPath)(t)
    X, X1, X2, X3, X4, X5, X6 = hyperbolic_jet(p.g, t)
    n = p.n
    return (Trajectory(; z=p.z₀ + X * n, v=X1 * n, a=X2 * n, ȧ=X3 * n, ä=X4 * n,
                       a⃛=X5 * n, a⃜=X6 * n),)
end

"""
    hyperbolic_motion(z₀, g; direction = (0, 0, 1))

A [`Prescribed`](@ref) motion of one particle in hyperbolic motion, the
motion of uniform proper acceleration `g` along `direction`, at rest at
`z₀` at `t = 0`: it travels `(√(1 + g²t²) - 1)/g` by time `t`, at speed
`gt/√(1 + g²t²)`. At `t = 0`, `a = g`, `ȧ = 0` and `ä = -3g³`, and the
retarded field of a point charge is time-symmetric there
(INSPIRAL.md, "The near field of an accelerated particle").
"""
function hyperbolic_motion(z₀, g::Real; direction=(0, 0, 1))
    T = float(promote_type(eltype(z₀), typeof(g)))
    g > 0 || throw(ArgumentError("the proper acceleration must be positive, got $g"))
    n = SVector{3,T}(direction)
    return Prescribed(HyperbolicPath{T}(SVector{3,T}(z₀), n / sqrt(dot(n, n)), T(g)))
end

"""
    retarded_field(path, k, Q, x, t)

The retarded field of a point charge `Q` on particle `k`'s worldline of
`path` (a function `t -> (Trajectory, …)`, as [`Prescribed`](@ref)
takes), and its time derivative, `(Φ, ∂ₜΦ)` at `(x, t)`. The retarded
time `s` is the root of `t - s = |x - z(s)|`, found by Newton's method
from `s = t - |x - z(t)|`; then, in this package's normalization
(`∂ₜ²Φ - ∇²Φ = -ρ`, the lab-frame charge `Q/γ`),

    Φ = -Q / (4π γ (R - R·v)),    R = x - z(s), v = v(s),

and `∂ₜΦ = (dΦ/ds) / (1 - R̂·v)`. A point that no past light cone of the
worldline reaches, as behind a particle in hyperbolic motion forever, is
refused.
"""
function retarded_field(path, k::Integer, Q, x, t)
    T = float(promote_type(eltype(x), typeof(t), typeof(Q)))
    return retarded_field(path, k, T(Q), SVector{3,T}(x), T(t))
end

function retarded_field(path, k::Integer, Q::T, X::SVector{3,T}, t::T) where {T}
    s = t - norm3(X - path(t)[k].z)
    # The scale of roundoff in `t - s - |x - z(s)|`, positions included.
    scale = max(abs(t), maximum(abs, X), one(T))
    for _ in 1:100
        tr = path(s)[k]
        R = X - tr.z
        r = norm3(R)
        Δ = (t - s - r) / (-1 + dot(R, tr.v) / r)   # f/f′, f′ < 0 for |v| < 1
        s -= Δ
        (abs(Δ) <= 8eps(T) * max(scale, abs(s)) || !isfinite(s)) && break
    end
    f = t - s - norm3(X - path(s)[k].z)
    isfinite(s) && abs(f) <= 64eps(T) * max(scale, abs(s)) ||
        no_retarded_time(X, t)
    tr = path(s)[k]
    R = X - tr.z
    ds_dt = inv(1 - dot(R, tr.v) / norm3(R))
    Φ = point_potential(path, k, Q, X, s)
    return Φ, ForwardDiff.derivative(σ -> point_potential(path, k, Q, X, σ), s) * ds_dt
end

@noinline no_retarded_time(X, t) = throw(ArgumentError(
    "no retarded time found for x = $(Tuple(X)) at t = $t: no past light cone of the " *
    "worldline reaches it"))

# `-Q/(4π γ (R - R·v))` with everything at the retarded time `s`.
@inline function point_potential(path, k, Q, X, s)
    tr = path(s)[k]
    R = X - tr.z
    γ = inv(sqrt(1 - dot(tr.v, tr.v)))
    return -Q / (4 * oftype(Q, π) * γ * (norm3(R) - dot(R, tr.v)))
end

@inline norm3(x) = sqrt(dot(x, x))

"""
    point_model(m::ParticleModel; ℓ = 1e-12)

A particle with `m`'s charge and size `ℓ` small enough to be a point at
every distance a grid resolves: its fields are the point limits of `m`'s
to `(ℓ/r)²`. Its mass is chosen above its classical radius and means
nothing.
"""
function point_model(m::ParticleModel{T}; ℓ::Real=T(1e-12)) where {T}
    return ParticleModel{T}(m.Q, T(ℓ), 2self_inertia(m.Q, T(ℓ)) + one(T))
end

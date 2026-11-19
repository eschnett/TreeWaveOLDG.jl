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

# The smoothstep of order `n` as its coefficients, `c[j + 1]` that of `xʲ`,
# exact integers rounded once to `T`, at most degree 15 (`n ≤ 7`), so that
# it is `isbits` and a device kernel evaluates it without building
# anything.
struct Smoothstep{T}
    n::Int
    c::NTuple{16,T}
end

function Smoothstep{T}(n::Integer) where {T}
    0 <= n <= 7 || throw(ArgumentError("the smoothstep's order must lie in 0:7, got $n"))
    # S_n(x) = x^{n+1} Σₖ (-1)^k C(n+k, k) C(2n+1, n-k) xᵏ.
    c = ntuple(16) do i
        k = i - 1 - (n + 1)
        0 <= k <= n || return zero(T)
        return T((-1)^k * binomial(n + k, k) * binomial(2n + 1, n - k))
    end
    return Smoothstep{T}(n, c)
end

# The `d`-th derivative at `x`, `d = -1` the integral from 0; 0 below
# `x = 0`, and above `x = 1` the constant 1 and the integral `x - 1/2`.
# Continuous from the right: at `x = 0` the polynomial's derivatives, at
# `x = 1` the constant's. Above `x = 1/2` it is evaluated from the
# mirror image, `S(x) = 1 - S(1 - x)`, so that near either end the
# polynomial's lowest power `x^{n+1}` keeps what vanishes there small to
# roundoff of itself rather than of the coefficients, which reach 10⁴.
@inline function (s::Smoothstep)(x, d::Int=0)
    if x < 0
        return zero(x)
    elseif x >= 1
        return d == -1 ? x - one(x) / 2 : d == 0 ? one(x) : zero(x)
    elseif 2x > 1
        y = smoothstep_poly(s, 1 - x, d)
        # I(x) = x - 1/2 + I(1 - x), S^{(d)}(x) = (-1)^{d+1} S^{(d)}(1 - x).
        return d == -1 ? x - one(x) / 2 + y : d == 0 ? 1 - y : isodd(d) ? y : -y
    end
    return smoothstep_poly(s, x, d)
end

@inline function smoothstep_poly(s::Smoothstep, x, d::Int)
    r = zero(x)
    if d == -1
        for j in 15:-1:0
            r = r * x + s.c[j + 1] / (j + 1)
        end
        return r * x
    end
    for j in 15:-1:d
        f = one(eltype(s.c))             # j!/(j - d)!
        for i in (j - d + 1):j
            f *= i
        end
        r = r * x + f * s.c[j + 1]
    end
    return r
end

"""
    smoothstep(n, x, d = 0)

The smoothstep of order `n`, the polynomial `S_n` of degree `2n + 1`
that rises from 0 at `x = 0` to 1 at `x = 1` with its first `n`
derivatives zero at both ends, `S_0(x) = x`, `S_1(x) = 3x² - 2x³`; 0
below `x = 0` and 1 above `x = 1`. With `d ≥ 1` its `d`-th derivative,
with `d = -1` its integral from 0, which is `x - 1/2` above `x = 1`.
`n` is at most 7. The ramps of [`ramped_circular_motion`](@ref) are made
of it (`INSPIRAL-PLAN.md`'s M8f).
"""
smoothstep(n::Integer, x::Real, d::Integer=0) = Smoothstep{float(typeof(x))}(n)(x, Int(d))

# A circle with an assumed past and a departure (INSPIRAL-PLAN.md's M8f).
# For `-T₀ ≤ t < 0`, and for `t ≥ 0` when `ε = 0`, it is `circle` itself.
# Before `-T₀` its angular velocity is `Ω S((t + T₀ + Δ)/Δ)`, the particle
# at rest at angle `φrest` before `-T₀ - Δ`. From `t = 0` its radius is
# `r - εΔ₂ I₂(t/Δ₂)`, `I₂` the integral of `S₂`, at the circle's angle.
struct RampedCircle{T}
    circle::CirclePath{T}
    T₀::T
    Δ::T
    S::Smoothstep{T}
    φrest::T
    ε::T
    Δ₂::T
    S₂::Smoothstep{T}
end

function (p::RampedCircle)(t)
    t < -p.T₀ && return (spinup(p, t),)
    tr = p.circle(t)[1]
    (iszero(p.ε) || t < 0) && return (tr,)
    return (departure(p, tr, t),)
end

# The `k`-th derivatives, `k = 0, …, 6`, of `r(t) E(t)` from those of `r`
# and `E`, by Leibniz's rule.
@inline function leibniz(r, E)
    return (r[1] * E[1],
            r[2] * E[1] + r[1] * E[2],
            r[3] * E[1] + 2r[2] * E[2] + r[1] * E[3],
            r[4] * E[1] + 3r[3] * E[2] + 3r[2] * E[3] + r[1] * E[4],
            r[5] * E[1] + 4r[4] * E[2] + 6r[3] * E[3] + 4r[2] * E[4] + r[1] * E[5],
            r[6] * E[1] + 5r[5] * E[2] + 10r[4] * E[3] + 10r[3] * E[4] + 5r[2] * E[5] +
            r[1] * E[6],
            r[7] * E[1] + 6r[6] * E[2] + 15r[5] * E[3] + 20r[4] * E[4] + 15r[3] * E[5] +
            6r[2] * E[6] + r[1] * E[7])
end

# The derivatives of `e^{iφ(t)}` from those of `φ`: `E′ = iφ′E`, and the
# higher ones by Leibniz's rule on that product.
@inline function phase_jets(φ)
    s, c = sincos(φ[1])
    E0 = complex(c, s)
    iφ = map(f -> complex(zero(f), f), φ)
    E1 = iφ[2] * E0
    E2 = iφ[3] * E0 + iφ[2] * E1
    E3 = iφ[4] * E0 + 2iφ[3] * E1 + iφ[2] * E2
    E4 = iφ[5] * E0 + 3iφ[4] * E1 + 3iφ[3] * E2 + iφ[2] * E3
    E5 = iφ[6] * E0 + 4iφ[5] * E1 + 6iφ[4] * E2 + 4iφ[3] * E3 + iφ[2] * E4
    E6 = iφ[7] * E0 + 5iφ[6] * E1 + 10iφ[5] * E2 + 10iφ[4] * E3 + 5iφ[3] * E4 +
         iφ[2] * E5
    return (E0, E1, E2, E3, E4, E5, E6)
end

# A complex number `w` in the circle's plane, `Re(w) e₁ + Im(w) e₂`.
@inline inplane(c::CirclePath, w) = real(w) * c.e₁ + imag(w) * c.e₂

@inline function spinup(p::RampedCircle, t)
    (; circle, Δ) = p
    Ω = circle.Ω
    x = (t + p.T₀ + Δ) / Δ
    # φ = φrest + ΩΔ I(x), φ⁽ᵏ⁾ = ΩΔ^{1-k} S^{(k-1)}(x).
    φ = (p.φrest + Ω * Δ * p.S(x, -1), Ω * p.S(x, 0), Ω / Δ * p.S(x, 1),
         Ω / Δ^2 * p.S(x, 2), Ω / Δ^3 * p.S(x, 3), Ω / Δ^4 * p.S(x, 4),
         Ω / Δ^5 * p.S(x, 5))
    w = map(E -> circle.r * E, phase_jets(φ))
    return Trajectory(; z=circle.centre + inplane(circle, w[1]), v=inplane(circle, w[2]),
                      a=inplane(circle, w[3]), ȧ=inplane(circle, w[4]),
                      ä=inplane(circle, w[5]), a⃛=inplane(circle, w[6]),
                      a⃜=inplane(circle, w[7]))
end

@inline function departure(p::RampedCircle, tr, t)
    (; circle, ε, Δ₂) = p
    Ω = circle.Ω
    x = t / Δ₂
    # δr = -εΔ₂ I₂(x), δr⁽ʲ⁾ = -εΔ₂^{1-j} S₂^{(j-1)}(x).
    δr = (-ε * Δ₂ * p.S₂(x, -1), -ε * p.S₂(x, 0), -ε / Δ₂ * p.S₂(x, 1),
          -ε / Δ₂^2 * p.S₂(x, 2), -ε / Δ₂^3 * p.S₂(x, 3), -ε / Δ₂^4 * p.S₂(x, 4),
          -ε / Δ₂^5 * p.S₂(x, 5))
    # e^{iφ} with `φ = φ₀ + Ωt`, and its derivatives `(iΩ)ᵏe^{iφ}`.
    s, c = sincos(circle.φ₀ + Ω * t)
    E0 = complex(c, s)
    iΩ = complex(zero(Ω), Ω)
    E = (E0, iΩ * E0, iΩ^2 * E0, iΩ^3 * E0, iΩ^4 * E0, iΩ^5 * E0, iΩ^6 * E0)
    w = leibniz(δr, E)
    return Trajectory(; z=tr.z + inplane(circle, w[1]), v=tr.v + inplane(circle, w[2]),
                      a=tr.a + inplane(circle, w[3]), ȧ=tr.ȧ + inplane(circle, w[4]),
                      ä=tr.ä + inplane(circle, w[5]), a⃛=tr.a⃛ + inplane(circle, w[6]),
                      a⃜=tr.a⃜ + inplane(circle, w[7]))
end

"""
    ramped_circular_motion(centre, radius, speed; phase = 0, normal = (0, 0, 1),
                           past = Inf, ramp = 2π/Ω, n = 4,
                           depart = 0, onset = 2π/Ω, n2 = 4)

[`circular_motion`](@ref)'s circle, `Ω = speed/radius`, with an assumed
past and a departure from it at `t = 0` (`INSPIRAL-PLAN.md`'s M8f).

The past: on the circle for `-past ≤ t`, and before that spun up from
rest along it, at angular velocity `Ω S_n((t + past + ramp)/ramp)` with
`S_n` the [`smoothstep`](@ref) of order `n`, so at rest before
`-past - ramp`. The acceleration and its first `n - 1` derivatives are
continuous at both joints, its `n`-th jumps (`n = 0`: `a` itself); the
speed stays below `speed`. `past = Inf` is the circle forever.

The departure: from `t = 0` the radius is `radius - ε onset I(t/onset)`,
`ε = depart`, `I` the integral of `S_{n2}`, at the circle's angle
`phase + Ωt`, so the radial speed rises to `-ε` over `onset` and stays
there. The acceleration and its first `n2 - 1` derivatives are continuous
at `t = 0` and at `t = onset`, its `n2`-th jumps, by `ε/onset^{n2+1}`
times a number of order one. `depart = 0` is the circle. The radius
shrinks without end: the run must end before it reaches zero.

Every piece has exact jets, `a` through `a⃜`, in closed form, so the
path runs in a device kernel and [`retarded_field`](@ref) takes it. With
`past = Inf` and `depart = 0` it is `circular_motion`'s path bit for bit.
"""
function ramped_circular_motion(centre, radius::Real, speed::Real; phase::Real=0,
                                normal=(0, 0, 1), past::Real=Inf, ramp::Real=0,
                                n::Integer=4, depart::Real=0, onset::Real=0,
                                n2::Integer=4)
    circle = circular_motion(centre, radius, speed; phase, normal).path
    T = typeof(circle.r)
    Ω = circle.Ω
    past >= 0 || throw(ArgumentError("past must be at least 0, got $past"))
    depart >= 0 || throw(ArgumentError("depart must be at least 0, got $depart"))
    speed + depart < 1 ||
        throw(ArgumentError("speed + depart must be below 1, got $(speed + depart)"))
    ramp = iszero(ramp) ? 2T(π) / Ω : T(ramp)
    onset = iszero(onset) ? 2T(π) / Ω : T(onset)
    ramp > 0 || throw(ArgumentError("ramp must be positive, got $ramp"))
    onset > 0 || throw(ArgumentError("onset must be positive, got $onset"))
    # At `t = -past`, `φ = φ₀ - Ω past = φrest + Ω ramp/2`, the integral of
    # `S` over the ramp being one half.
    φrest = isfinite(past) ? circle.φ₀ - Ω * T(past) - Ω * ramp / 2 : circle.φ₀
    return Prescribed(RampedCircle{T}(circle, T(past), ramp, Smoothstep{T}(n), φrest,
                                      T(depart), onset, Smoothstep{T}(n2)))
end

# Several one-particle paths as one path of as many particles, particle
# `k` moving on `paths[k]`. A tuple of `isbits` paths, so a device kernel
# takes it as it takes one.
struct PathTuple{P<:Tuple}
    paths::P
end

(p::PathTuple)(t) = map(q -> q(t)[1], p.paths)

"""
    binary_circular_motion(centre, radius, speed; phase = 0, normal = (0, 0, 1),
                           past = Inf, ramp = 2π/Ω, n = 4)

A [`Prescribed`](@ref) motion of two particles on one circle, half an
orbit apart: particle 1 on [`ramped_circular_motion`](@ref)'s path at
`phase`, particle 2 on the same at `phase + π`, both with the same past.
The charges' dipoles cancel, so their radiation starts at `ℓ = 2`. It
takes `ramped_circular_motion`'s refusals; there is no departure.
"""
function binary_circular_motion(centre, radius::Real, speed::Real; phase::Real=0,
                                normal=(0, 0, 1), past::Real=Inf, ramp::Real=0,
                                n::Integer=4)
    at(φ) = ramped_circular_motion(centre, radius, speed; phase=φ, normal, past, ramp,
                                   n).path
    a = at(phase)
    return Prescribed(PathTuple((a, at(phase + oftype(a.circle.φ₀, π)))))
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
    Φ, Π, found = retarded_root(path, k, Q, X, t)
    found || no_retarded_time(X, t)
    return Φ, Π
end

# `retarded_field` as `(Φ, ∂ₜΦ, found)`, with `found` false where no
# retarded time is found, in place of the error, which a device kernel
# cannot build.
@inline function retarded_root(path, k::Integer, Q::T, X::SVector{3,T}, t::T) where {T}
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
    found = isfinite(s) && abs(f) <= 64eps(T) * max(scale, abs(s))
    tr = path(s)[k]
    R = X - tr.z
    ds_dt = inv(1 - dot(R, tr.v) / norm3(R))
    Φ = point_potential(path, k, Q, X, s)
    return Φ, ForwardDiff.derivative(σ -> point_potential(path, k, Q, X, σ), s) * ds_dt,
           found
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

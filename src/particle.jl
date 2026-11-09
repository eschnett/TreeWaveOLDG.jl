# The analytic particle of INSPIRAL.md: a scalar charge `Q` of physical
# size `ℓ`, its field `Ψ` (the boosted static field plus the near-field
# terms and the lab slice of orders 1–3), the field's time derivative `𝒟Ψ` along the
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
# The near-field terms are the rest-frame ones, evaluated at the
# rest-frame position `y = Λ(v)(x - z)` with the rest frame's derivatives
# of the worldline (`rest_frame_derivatives`); the lab slice of each order
# is what the field on the lab's time slice adds to them
# ([`lab_slice`](@ref); CODE.md, "The near field").

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
    Trajectory(; z, v, a = 0, ȧ = 0, ä = 0, a⃛ = 0, a⃜ = 0)

A particle's position, velocity, acceleration and the acceleration's
first four time derivatives, as `SVector{3}`s: the Taylor polynomial in `t`
that the time derivatives of [`particle_field`](@ref) act on. The order-3
term reads `ä` through the rest-frame `C`, so the source, which takes its
second time derivative, needs `a⃜`, the sixth derivative of `z`; without
it the source along a path whose `a⃜` does not vanish is that of the
path's Taylor polynomial, not of the path (`CODE.md`, "The near
field").
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
    a⃜::SVector{3,T}
end

function Trajectory(; z, v, a=zero(z), ȧ=zero(z), ä=zero(z), a⃛=zero(z), a⃜=zero(z))
    T = float(promote_type(eltype(z), eltype(v), eltype(a), eltype(ȧ), eltype(ä),
                           eltype(a⃛), eltype(a⃜)))
    return Trajectory{T}(z, v, a, ȧ, ä, a⃛, a⃜)
end

"""
    trajectory_at(traj, t)

The trajectory's Taylor polynomial, and its derivatives, at time `t`
from the moment `traj` describes.
"""
@inline function trajectory_at(traj::Trajectory, t)
    (; z, v, a, ȧ, ä, a⃛, a⃜) = traj
    return Trajectory(z + t * (v + t * (a / 2 + t * (ȧ / 6 + t * (ä / 24 + t * (a⃛ / 120 +
                                                                          t * a⃜ / 720))))),
                      v + t * (a + t * (ȧ / 2 + t * (ä / 6 + t * (a⃛ / 24 + t * a⃜ / 120)))),
                      a + t * (ȧ + t * (ä / 2 + t * (a⃛ / 6 + t * a⃜ / 24))),
                      ȧ + t * (ä + t * (a⃛ / 2 + t * a⃜ / 6)),
                      ä + t * (a⃛ + t * a⃜ / 2),
                      a⃛ + t * a⃜,
                      a⃜ + zero(t) * a⃜)
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
rest-frame term at `y = Λ(v)(x - z)`, with the rest frame's derivatives
of the worldline, [`rest_frame_derivatives`](@ref): the proper
acceleration `α′ = γ²Λ(v)a` for `a` and `C` for `ä`. At rest this is
the time-symmetric field of the extended particle, order by order.
Moving, the particle field adds the lab slice, [`lab_slice`](@ref).
"""
@inline function near_field(m::ParticleModel, ::Val{N}, x, traj::Trajectory) where {N}
    y = boost(traj.v, x - traj.z)
    α′, _, C = rest_frame_derivatives(traj)
    return near_field_term(Val(N), m.Q, m.ℓ, y, α′, C)
end

"""
    rest_frame_derivatives(traj)

`(α′, B, C)`: the second, third and fourth derivatives of the particle's
worldline in its instantaneous rest frame, with respect to that frame's
coordinate time `t′`, at the moment `traj` describes. The rest frame is
the lab boosted by `v`, `t′ = γ(t - v·r)` and `y = Λ(v)r - γvt`, and
`α′ = γ²Λ(v)a` is the proper acceleration. These are the `a`, `ȧ` and
`ä` of the near field and the lab slice (`derivations/nearfield.wl`,
`labslice.wl`), which hold in that frame.

Along the worldline `r(t) = vt + ½at² + ⅙ȧt³ + (1/24)ät⁴`, `y(t)` has no
linear term and `t′(t) = t/γ - ½γ(v·a)t² - ⅙γ(v·ȧ)t³ + …`. Inverting
`t′(t)` as a series, `t = s₁t′ + s₂t′² + s₃t′³`, and composing gives the
derivatives of `y` in `t′`. Generic in the number type, so that it
works on dual numbers along the motion.
"""
@inline function rest_frame_derivatives(traj::Trajectory)
    (; v, a, ȧ, ä) = traj
    γ = lorentz_factor(v)
    y₂, y₃, y₄ = boost(v, a) / 2, boost(v, ȧ) / 6, boost(v, ä) / 24
    τ₂, τ₃ = -γ * dot(v, a) / 2, -γ * dot(v, ȧ) / 6
    s₁ = γ
    s₂ = -τ₂ * γ^3
    s₃ = γ^5 * (2τ₂^2 - τ₃ / γ)
    α′ = 2s₁^2 * y₂
    B = 6 * (2s₁ * s₂ * y₂ + s₁^3 * y₃)
    C = 24 * ((s₂^2 + 2s₁ * s₃) * y₂ + 3s₁^2 * s₂ * y₃ + s₁^4 * y₄)
    return α′, B, C
end

"""
    lab_slice(m, Val(N), x, traj)

`Λ_N`, the lab slice of order `N = 1, 2, 3`: what the particle field of
order `N` on the lab's time slice adds to the rest-frame term
[`near_field`](@ref) (`INSPIRAL.md`, "Moving particles and the lab
slice"; `derivations/labslice.wl`). It is the time-symmetric field of
the lab-sliced density in the instantaneous rest frame, at rest-frame
time `t′ = -v·y`, from which the rest-frame term at `t′ = 0` is taken
away. It vanishes at `v = 0`. `Λ₁` is the `t′²` term of `Φ_C`,

    T₂ = -½ t′² (α′·∇Φ₀(y)) = -(v·y)² (α′·y) Φ₀′(|y|²),

which outside the particle is `-(Q/8π) v²(v̂·n̂)²(α′·n̂)`, of order `r⁰`,
plus the field of the density `½(v·y)²(α′·∇ρ₀)` by which the lab-sliced
particle differs from the rest-frame-sliced one, a dipole outside.
"""
@inline function lab_slice(m::ParticleModel, ::Val{N}, x, traj::Trajectory) where {N}
    y = boost(traj.v, x - traj.z)
    α′, B, C = rest_frame_derivatives(traj)
    return lab_slice_term(Val(N), m.Q, m.ℓ, y, traj.v, α′, B, C)
end

function check_order(order)
    0 <= order <= 3 || throw(ArgumentError(
        "the near field is carried to order 0, 1, 2 or 3, got order = $order"))
    return nothing
end

"""
    particle_field(m, x, traj; order = 3)

`Ψ`: the boosted static field plus, for each order `N` from `1` to
`order`, the near-field term and the lab slice of order `N`
([`near_field`](@ref), [`lab_slice`](@ref); INSPIRAL.md, "The particle
field"). `order = 0` is the particle paper's boosted static field alone.
"""
@inline function particle_field(m::ParticleModel, x, traj::Trajectory; order::Int=3)
    check_order(order)
    Ψ = boosted_static(m, x, traj.z, traj.v)
    order >= 1 && (Ψ += near_field(m, Val(1), x, traj) + lab_slice(m, Val(1), x, traj))
    order >= 2 && (Ψ += near_field(m, Val(2), x, traj) + lab_slice(m, Val(2), x, traj))
    order >= 3 && (Ψ += near_field(m, Val(3), x, traj) + lab_slice(m, Val(3), x, traj))
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

# A field that depends on `x` through `q = |y|²` and the four invariants
# `u·y` linear in `y`, and on `t` through them and six `x`-independent
# ones along the motion ([`ParticleJets`](@ref)). Of each invariant `I`
# the bracket needs its value, its rate `İᵤ` in uniform motion (`z`
# moving with `v`, and `v` and the worldline's rest-frame derivatives
# held), and the acceleration's parts `δİ = İ - İᵤ` and `δÏ = Ï - Ïᵤ`,
# each formed directly and not as a difference. With them,
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

# The pairs of rest-frame vectors `(v, α′, B, C)` whose dot products are
# the `x`-independent invariants, in the generated forms' order:
# `v²`, `v·α′`, `|α′|²`, `v·B`, `α′·B`, `v·C`.
const CONSTANT_PAIRS = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 4))

"""
    ParticleJets(m, traj)

What [`ParticleSource`](@ref) needs of one particle, at the moment
`traj` describes, that does not depend on the point: the invariants'
`x`-independent factors and their first two time derivatives. With
`r = x - z`, `y = Λ(v)r`, `M = Λ(v)² = I + γ²vvᵀ`, and the rest-frame
vectors `u = v, α′, B, C` of [`rest_frame_derivatives`](@ref),

    |y|² = rᵀMr,   u·y = (Λ(v)u)·r,

with `Λ(v)v = γv`, and `ṙ = -v`, `r̈ = -a`; and the six dot products of
the `u` that the lab slice reads (`v²`, `v·α′`, `|α′|²`, `v·B`, `α′·B`,
`v·C`), which do not depend on `x`. The lab slice depends on these only
through the coefficients of its groups' polynomials
(`_lab_coefficients`, generated), which are held here once per particle,
with their first two time derivatives for the term and alone for its
Laplacian, so that a point evaluates one polynomial per group.
"""
struct ParticleJets{T}
    Q::T
    ℓ::T
    z::SVector{3,T}
    M::NTuple{3,SMatrix{3,3,T,9}}   # M, Ṁ, M̈
    uq::SVector{3,T}                # q̇ᵤ = uq·r
    bq::SVector{3,T}                # δq̈ = bq·r + rᵀM̈r
    L::NTuple{4,NTuple{3,SVector{3,T}}}  # Λ(v)u, its first two rates, u = v, α′, B, C
    L̇ᵤ::NTuple{4,T}                 # d(u·y)/dt in uniform motion
    L̈₀::NTuple{4,T}                 # δ d²(u·y)/dt² = L̈·r + L̈₀
    c::NTuple{6,NTuple{3,T}}        # the dot products and their rates
    Ct::LabTermJets{T}              # the lab slice's coefficients and rates
    Cl::LabLaplacianCoefficients{T} # its Laplacian's coefficients
end

function ParticleJets(m::ParticleModel{T}, traj::Trajectory{T}) where {T}
    jet(f) = (g = t -> f(trajectory_at(traj, t));
              (g(zero(T)), ForwardDiff.derivative(g, zero(T)), second_derivative(g, zero(T))))
    Mof(v) = one(SMatrix{3,3,eltype(v)}) + v * v' / (1 - dot(v, v))
    rest(tr) = (tr.v, rest_frame_derivatives(tr)...)
    M = jet(tr -> Mof(tr.v))
    L = ntuple(i -> jet(tr -> boost(tr.v, rest(tr)[i])), Val(4))
    constants(tr) = (u = rest(tr); map(ij -> dot(u[ij[1]], u[ij[2]]), CONSTANT_PAIRS))
    c = map(ij -> jet(tr -> (u = rest(tr); dot(u[ij[1]], u[ij[2]]))), CONSTANT_PAIRS)
    ℓ = m.ℓ
    Ct = (jet(tr -> _lab_coefficients(Val(1), Val(:term), ℓ, constants(tr)...)),
          jet(tr -> _lab_coefficients(Val(2), Val(:term), ℓ, constants(tr)...)),
          jet(tr -> _lab_coefficients(Val(3), Val(:term), ℓ, constants(tr)...)))
    c₀ = map(first, c)
    Cl = (_lab_coefficients(Val(1), Val(:laplacian), ℓ, c₀...),
          _lab_coefficients(Val(2), Val(:laplacian), ℓ, c₀...),
          _lab_coefficients(Val(3), Val(:laplacian), ℓ, c₀...))
    (; z, v, a) = traj
    # q = rᵀMr: q̇ = 2ṙᵀMr + rᵀṀr, q̈ = 2r̈ᵀMr + 2ṙᵀMṙ + 4ṙᵀṀr + rᵀM̈r, of
    # which uniform motion keeps -2vᵀMr and 2vᵀMv.
    uq = -2 * (M[1] * v)
    bq = -2 * (M[1] * a) - 4 * (M[2] * v)
    # u·y = k·r: d/dt = k̇·r - k·v, d²/dt² = k̈·r - 2k̇·v - k·a, of which
    # uniform motion keeps -k·v.
    L̇ᵤ = map(k -> -dot(k[1], v), L)
    L̈₀ = map(k -> -2dot(k[2], v) - dot(k[1], a), L)
    return ParticleJets{T}(m.Q, m.ℓ, z, M, uq, bq, L, L̇ᵤ, L̈₀, c, Ct, Cl)
end

# Coefficient `i` of the lab slice's generated `_lab_evaluate`: as it is,
# or, from its value and first two time derivatives, lifted as the
# bracket reads it, an invariant that does not depend on `x`, with the
# directions `d` of the further outer partials. Read in place, so that
# the table is not copied for every point.
@inline _coefficient(c::SVector, i, d) = c[i]
@inline function _coefficient(jets::NTuple{3,SVector}, i, d)
    c, ċ, c̈ = jets[1][i], jets[2][i], jets[3][i]
    return lift(c, zero(c), ċ, c̈, map(_ -> zero(c), d))
end

# Ψ as a function of the invariants: `q`, the four `u·y` (`vy`, `ay`,
# `by`, `cy`), `|α′|²` for the near field, and the lab slice's
# coefficients, which hold the other invariants, lifted with the
# directions `d`.
@inline function invariant_field(p::ParticleJets, order, q, lin, aa, d)
    vy, ay, by, cy = lin
    C = p.Ct
    Ψ = p.Q * _phi0(q, p.ℓ)
    order >= 1 && (Ψ += _near_field_term(Val(1), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(1), Val(:term), p.Q, p.ℓ, q, lin..., C[1], d))
    order >= 2 && (Ψ += _near_field_term(Val(2), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(2), Val(:term), p.Q, p.ℓ, q, lin..., C[2], d))
    order >= 3 && (Ψ += _near_field_term(Val(3), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(3), Val(:term), p.Q, p.ℓ, q, lin..., C[3], d))
    return Ψ
end

# ∇²_y of the near-field terms and the lab slice, as a function of the
# invariants.
@inline function invariant_laplacian(p::ParticleJets, order, q, lin, aa)
    vy, ay, by, cy = lin
    C = p.Cl
    L = zero(q)
    order >= 1 && (L += _near_field_laplacian(Val(1), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(1), Val(:laplacian), p.Q, p.ℓ, q, lin..., C[1]))
    order >= 2 && (L += _near_field_laplacian(Val(2), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(2), Val(:laplacian), p.Q, p.ℓ, q, lin..., C[2]))
    order >= 3 && (L += _near_field_laplacian(Val(3), p.Q, p.ℓ, q, ay, aa, cy) +
                        _lab_evaluate(Val(3), Val(:laplacian), p.Q, p.ℓ, q, lin..., C[3]))
    return L
end

# One particle's `(Ψ, 𝒟²Ψ - ∇²Ψ + ρ, n·∇Ψ)` at `x`, with `n·∇Ψ` only
# where `n` is given.
#
# `ρ` cancels against `□_vΦ_C = ∇²Φ_C - (v·∇)²Φ_C` analytically, and
# each near-field and lab-slice term has `∇²_x = ∇²_y + γ²(v·∇_y)²` with
# `γ²(v·∇_y)² = (v·∂_z)²`, so the bracket is `𝒟²Ψ - (v·∂_z)²Ψ`, which
# `lift` forms from the acceleration's parts alone, minus the generated
# `∇²_y` of those terms.
@inline function source_terms(p::ParticleJets, order, x, n...)
    r = x - p.z
    M, M′, M″ = p.M
    Mr = M * r
    q = lift(dot(r, Mr), dot(p.uq, r), dot(r, M′ * r), dot(p.bq, r) + dot(r, M″ * r),
             map(nᵢ -> 2dot(nᵢ, Mr), n))
    lin = map(p.L, p.L̇ᵤ, p.L̈₀) do (k, k′, k″), k̇ᵤ, k̈₀
        return lift(dot(k, r), k̇ᵤ, dot(k′, r), dot(k″, r) + k̈₀, map(nᵢ -> dot(nᵢ, k), n))
    end
    a, ȧ, ä = p.c[3]                            # |α′|²
    aa = lift(a, zero(a), ȧ, ä, map(nᵢ -> zero(a), n))
    G = invariant_field(p, order, q, lin, aa, n)
    value = ForwardDiff.value
    partial = ForwardDiff.partials
    v₀(I) = value(value(I))
    ∇²y = invariant_laplacian(p, order, v₀(q), map(v₀, lin), a)
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
split"). `Ψ` depends on `x` and `t` through eleven invariants, `|y|²`,
the four `u·y` and the six dot products of [`ParticleJets`](@ref). `ρ`
cancels against `□_vΦ_C` analytically and `∇²_x` of each near-field and
lab-slice term is `∇²_y + (v·∂_z)²`, so the bracket is
`𝒟²Ψ - (v·∂_z)²Ψ` minus the generated `∇²_y` of those terms,
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

# The particles are summed by constant index, with the terms inlined and
# `s` passed as an argument, never captured, so that each particle's jets
# are read in place: a loop over `s.particles`, a `map` over it, or a
# closure capturing `s` copied the 12.9 KB `ParticleJets` for every point,
# a quarter of the source at order 3 and nearly all of it at order 0
# (CODE.md, "Prescribed paths" under "Measured results"). The call itself
# is not inlined, so that the source is the same to the bit wherever it is
# evaluated: inlined into each projection the compiler compiles, it fused
# a `muladd` in one and not in another, and the inspiral's right-hand
# side, which projects into a view, differed from a projection into an
# array by an ulp.
@noinline function (s::ParticleSource{T,K})(x) where {T,K}
    W, ∇W, ∇²W = window_terms(s.window, x)
    iszero(W) && return zero(W)
    iszero(∇W) && iszero(∇²W) && return W * _sum_particles(_inner_term, Val(K), s, x)
    return _sum_particles(_transition_term, Val(K), s, x, W, ∇W, ∇²W)
end

@inline _inner_term(i, s, x) = source_terms(s.particles[i], s.order, x)[2]
@inline function _transition_term(i, s, x, W, ∇W, ∇²W)
    Ψ, bracket, ∇WΨ = source_terms(s.particles[i], s.order, x, ∇W)
    return W * bracket - 2∇WΨ - Ψ * ∇²W
end

# `f(1, args...) + … + f(K, args...)`, unrolled, with each index a
# constant.
@inline _sum_particles(f::F, ::Val{1}, args...) where {F} = f(1, args...)
@inline _sum_particles(f::F, ::Val{K}, args...) where {F,K} =
    _sum_particles(f, Val(K - 1), args...) + f(K, args...)

"""
    particle_source(models, w, x, trajs; order = 3)

The effective source at `x`, `ParticleSource(models, w, trajs; order)(x)`;
see [`ParticleSource`](@ref), which is the form to evaluate at many
points.
"""
particle_source(models::Tuple, w::Window, x, trajs::Tuple; order::Int=3) =
    ParticleSource(models, w, trajs; order)(x)

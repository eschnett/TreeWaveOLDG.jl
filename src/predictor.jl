# The acceleration predictor of INSPIRAL.md ("The acceleration and its
# derivatives"; INSPIRAL-PLAN.md, M6a): the instantaneous acceleration
# of each particle under the external forces, the particles' Newtonian
# attraction (M6b) and the other particles' boosted static fields in
# point form, with the physical masses, and its
# time derivatives along the motion it predicts. A dynamic particle
# solves `a`, `ȧ` and `ä` from its equation of motion; the predictor
# gives it `a⃛` and `a⃜`, which the source reads and the equation of
# motion cannot. See "The equations of motion" in `CODE.md`.

"""
    SpringForce(centre, k)

An external force `F(z, v) = -k (z - centre)` that holds a particle on a
circle about `centre`: at radius `r` and speed `v` when
`k = Mγv²/r²`, since the equation of motion
`Mγ a = (I - v vᵀ) F` gives `|a| = k r/(Mγ)` for a force normal to `v`.
It is `isbits`, as an external force must be.
"""
struct SpringForce{T}
    centre::SVector{3,T}
    k::T
end

function SpringForce(centre, k)
    T = float(promote_type(eltype(centre), typeof(k)))
    return SpringForce{T}(SVector{3,T}(Tuple(centre)), T(k))
end

(f::SpringForce)(z, v) = -f.k * (z - f.centre)

"""
    NoForce()

The external force of a free particle, zero.
"""
struct NoForce end
(::NoForce)(z, v) = zero(z)

"""
    NewtonianForce(G)

The instantaneous Newtonian attraction between the particles, which binds
INSPIRAL.md's binary ("Assumptions", "Equations of motion"): on particle
`k` from particle `b`,

    F_N = G Mₖ M_b (z_b - zₖ)/|z_b - zₖ|³,

with the physical masses, as the orbital energy has them
("Diagnostics"). An interaction is called as `f(Mₖ, zₖ, M_b, z_b)` and
gives the force on `k`; it enters the equation of motion as `F_ext` does,
through `(I - v vᵀ)`. It is `isbits`.
"""
struct NewtonianForce{T}
    G::T
end

NewtonianForce(G::Real) = NewtonianForce{float(typeof(G))}(G)

function (f::NewtonianForce)(Mk, zk, Mb, zb)
    r = zb - zk
    return (f.G * Mk * Mb / norm(r)^3) * r
end

"""
    NoInteraction()

No force between the particles but their fields', the interaction of a
[`Dynamic`](@ref) motion and of the predictor unless one is given.
"""
struct NoInteraction end
(::NoInteraction)(Mk, zk, Mb, zb) = zero(zk)

# The force of `interaction` on particle `k` from all the others, at the
# positions `zs`.
@inline function pair_force(interaction, models::NTuple{K,ParticleModel}, zs::NTuple{K},
                            k::Int) where {K}
    F = zero(zs[k])
    for b in 1:K
        b == k && continue
        F += interaction(models[k].M, zs[k], models[b].M, zs[b])
    end
    return F
end

"""
    spring_circle(model, centre, radius, speed; phase = 0, normal = (0, 0, 1))

The [`SpringForce`](@ref) that holds a particle of `model`'s physical
mass on [`circular_motion`](@ref)`(centre, radius, speed; phase,
normal)`, and that motion, as `(force, motion)`.
"""
function spring_circle(m::ParticleModel{T}, centre, radius::Real, speed::Real;
                       phase::Real=0, normal=(0, 0, 1)) where {T}
    motion = circular_motion(centre, radius, speed; phase, normal)
    γ = inv(sqrt(1 - T(speed)^2))
    return SpringForce(SVector{3,T}(centre), m.M * γ * T(speed)^2 / T(radius)^2), motion
end

"""
    newtonian_binary(models, separation, speed; centre = (0, 0, 0), phase = 0,
                     normal = (0, 0, 1), past = Inf, ramp = 0, n = 4)

The [`NewtonianForce`](@ref) that holds the two particles of `models`, of
equal physical mass, on [`binary_circular_motion`](@ref)`(centre,
separation/2, speed; phase, normal, past, ramp, n)`, and that motion, as
`(force, motion)`: INSPIRAL.md's "Initial data", step 1, at
`Ω = 2 speed/separation`. `G` is the one for which the predictor's
acceleration ([`predicted_acceleration`](@ref)), from Newton's attraction
and the other particle's boosted static field, is the circle's
centripetal one, `v²/r`. That acceleration is affine in `G`, so two
evaluations give it exactly; by the orbit's symmetry under the half-turn
about `centre` the same `G` holds both particles. The masses are equal
and the velocities opposite, so `Σ γMv = 0` and the centre of mass is
`centre`. Unequal masses would need two radii, which
`binary_circular_motion` does not have, and are refused; so is a pair of
charges that alone attract more than the circle needs.
"""
function newtonian_binary(models::NTuple{2,ParticleModel{T}}, separation::Real,
                          speed::Real; centre=(0, 0, 0), phase::Real=0, normal=(0, 0, 1),
                          past::Real=Inf, ramp::Real=0, n::Integer=4) where {T}
    models[1].M == models[2].M || throw(ArgumentError(
        "newtonian_binary holds particles of equal mass only, got $(models[1].M) and " *
        "$(models[2].M)"))
    separation > 0 || throw(ArgumentError("the separation must be positive, got $separation"))
    motion = binary_circular_motion(centre, T(separation) / 2, T(speed); phase, normal,
                                    past, ramp, n)
    trs = motion.path(zero(T))
    ext = (NoForce(), NoForce())
    a₀ = predicted_acceleration(models, ext, trs)[1]
    a₁ = predicted_acceleration(models, ext, trs; interaction=NewtonianForce(one(T)))[1]
    a = trs[1].a
    n̂ = a / norm(a)
    G = dot(a - a₀, n̂) / dot(a₁ - a₀, n̂)
    G > 0 || throw(ArgumentError(
        "the charges alone attract more than the circle needs (G = $G); no Newtonian " *
        "force holds them on it"))
    return NewtonianForce(G), motion
end

# The acceleration of a particle of physical mass `M` and charge `Q` at
# `(z, v)` under a force `F` from outside and a field whose gradient and
# rate at the particle are `∇Φ` and `Π` (INSPIRAL.md, "Equations of
# motion", in point form with `m = M`):
#     Mγ a = -(Q/γ)(∇Φ + v Π) + (I - v vᵀ) F.
@inline function point_acceleration(M, Q, v, F, ∇Φ, Π)
    γ = lorentz_factor(v)
    return (-(Q / γ) * (∇Φ + v * Π) + F - dot(v, F) * v) / (M * γ)
end

"""
    predicted_acceleration(models, external, trajs; interaction = NoInteraction())

Each particle's acceleration at the positions and velocities of `trajs`
(a tuple of [`Trajectory`](@ref)s, whose other jets are not read):
`external[k](z, v)`, the `interaction` with the other particles (such as
[`NewtonianForce`](@ref)), and the other particles' boosted static fields
`Φ_C` at the particle, with their rate `-v_b·∇Φ_C` of uniform motion,
through the point equation of motion with the physical mass `M`. A
tuple of 3-vectors.
"""
function predicted_acceleration(models::NTuple{K,ParticleModel}, external::NTuple{K,Any},
                                trajs::NTuple{K,Trajectory};
                                interaction=NoInteraction()) where {K}
    zs = map(tr -> tr.z, trajs)
    return ntuple(Val(K)) do k
        m, tr = models[k], trajs[k]
        z, v = tr.z, tr.v
        ∇Φ = zero(z)
        Π = zero(eltype(z))
        for b in 1:K
            b == k && continue
            mb, tb = models[b], trajs[b]
            g = ForwardDiff.gradient(x -> boosted_static(mb, x, tb.z, tb.v), z)
            ∇Φ += g
            Π -= dot(tb.v, g)
        end
        F = external[k](z, v) + pair_force(interaction, models, zs, k)
        point_acceleration(m.M, m.Q, v, F, ∇Φ, Π)
    end
end

# The `j`-th derivative at 0 of `f`, a function of one real variable that
# returns a vector, by `j` nested dual numbers.
@inline _derivative(f, ::Val{0}, t) = f(t)
@inline _derivative(f, ::Val{j}, t) where {j} =
    ForwardDiff.derivative(s -> _derivative(f, Val(j - 1), s), t)

# The particles' accelerations along the Taylor polynomials of `trajs`
# at `t`, stacked into one `SVector{3K}`.
function _stacked_acceleration(models, external, interaction, trajs::NTuple{K}, t) where {K}
    as = predicted_acceleration(models, external, map(tr -> trajectory_at(tr, t), trajs);
                                interaction)
    return vcat(as...)
end

"""
    predicted_jets(models, external, zs, vs; interaction = NoInteraction())

The [`Trajectory`](@ref)s of the motion the predictor gives from the
positions `zs` and velocities `vs`: `a` from
[`predicted_acceleration`](@ref) with `interaction`, and `ȧ`, `ä`, `a⃛`, `a⃜` its time
derivatives along that motion. They are found by iterating on the
trajectories' Taylor polynomials: each pass differentiates the
acceleration along the polynomial of the last, which makes one more
derivative exact, since `a⁽ʲ⁾` reads the position's derivatives
through `j` and so the acceleration's through `j - 2`. Four passes
after the first make `a⃜` exact.
"""
function predicted_jets(models::NTuple{K,ParticleModel}, external::NTuple{K,Any},
                        zs::NTuple{K}, vs::NTuple{K};
                        interaction=NoInteraction()) where {K}
    T = promote_type(map(eltype, zs)..., map(eltype, vs)...)
    trajs = map((z, v) -> Trajectory(; z=SVector{3,T}(z), v=SVector{3,T}(v)), zs, vs)
    pick(s, k) = SVector{3,T}(s[3k - 2], s[3k - 1], s[3k])
    for pass in 1:5
        f = let trajs = trajs
            t -> _stacked_acceleration(models, external, interaction, trajs, t)
        end
        t0 = zero(T)
        a = _derivative(f, Val(0), t0)
        ȧ = pass >= 2 ? _derivative(f, Val(1), t0) : zero(a)
        ä = pass >= 3 ? _derivative(f, Val(2), t0) : zero(a)
        a⃛ = pass >= 4 ? _derivative(f, Val(3), t0) : zero(a)
        a⃜ = pass >= 5 ? _derivative(f, Val(4), t0) : zero(a)
        trajs = ntuple(k -> Trajectory(; z=trajs[k].z, v=trajs[k].v, a=pick(a, k),
                                       ȧ=pick(ȧ, k), ä=pick(ä, k), a⃛=pick(a⃛, k),
                                       a⃜=pick(a⃜, k)), Val(K))
    end
    return trajs
end

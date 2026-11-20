# The binary's diagnostics (INSPIRAL.md, "Diagnostics"; INSPIRAL-PLAN.md,
# M6b): its orbit from the particles' positions and velocities, its
# energy and centre of mass, the leading-order inspiral, and the
# eccentricity from the separation's oscillation. The flux through
# spheres is the extraction's (`sphere_flux`). See "Diagnostics" in
# `CODE.md`.

"""
    binding_constant(models, interaction)

`K = G M₁M₂ + Q₁Q₂/4π` of two particles of `models` under `interaction`,
a [`NewtonianForce`](@ref) or [`NoInteraction`](@ref): the attraction is
`K/d²` and the potential energy `-K/d` (INSPIRAL.md, "Diagnostics"). Like
charges attract in a scalar field.
"""
function binding_constant(models::NTuple{2,ParticleModel}, interaction)
    G = interaction isa NewtonianForce ? interaction.G : zero(models[1].M)
    return G * models[1].M * models[2].M + models[1].Q * models[2].Q / 4π
end

"""
    orbit_diagnostics(models, interaction, zs, vs; normal = (0, 0, 1))

The orbit of two particles of `models` at positions `zs` and velocities
`vs`, as a named tuple:

- `d`, the separation `|z₁ - z₂|`, and `ddot`, its rate;
- `phase`, the angle of `z₁ - z₂` about `normal`, from the first of two
  axes normal to it, and `omega`, its rate, `((z₁ - z₂) × (v₁ - v₂))⋅n̂/d²`;
- `energy`, the orbital energy with the physical masses and the analytic
  interaction, `Σ M(γ - 1) - K/d` ([`binding_constant`](@ref));
- `centre`, the centre of mass `Σ γMz / Σ γM`, and `momentum`, `Σ γMv`.

The phase is in `(-π, π]`; unwrap it along a run.
"""
function orbit_diagnostics(models::NTuple{2,ParticleModel}, interaction, zs, vs;
                           normal=(0, 0, 1))
    T = typeof(models[1].M)
    z1, z2 = SVector{3,T}(zs[1]), SVector{3,T}(zs[2])
    v1, v2 = SVector{3,T}(vs[1]), SVector{3,T}(vs[2])
    n̂ = SVector{3,T}(normal) / norm(SVector{3,T}(normal))
    a = abs(n̂[1]) < 0.9 ? SVector{3,T}(1, 0, 0) : SVector{3,T}(0, 1, 0)
    e₁ = a - dot(a, n̂) * n̂
    e₁ /= norm(e₁)
    e₂ = cross(n̂, e₁)
    r, w = z1 - z2, v1 - v2
    d = norm(r)
    γ1, γ2 = lorentz_factor(v1), lorentz_factor(v2)
    m1, m2 = γ1 * models[1].M, γ2 * models[2].M
    K = binding_constant(models, interaction)
    return (; d, ddot=dot(r, w) / d, phase=atan(dot(r, e₂), dot(r, e₁)),
            omega=dot(cross(r, w), n̂) / d^2,
            energy=models[1].M * (γ1 - 1) + models[2].M * (γ2 - 1) - K / d,
            centre=(m1 * z1 + m2 * z2) / (m1 + m2), momentum=m1 * v1 + m2 * v2)
end

"""
    dipole_ddot(models, interaction, d)

The leading-order rate of the separation of a circular orbit of
separation `d`, `ḋ = -Δq² K/(6π d²)` with `Δq = Q₁/M₁ - Q₂/M₂`, from the
dipole power `|d̈|²/12π` and the Newtonian orbital energy `-K/2d`
(INSPIRAL.md, "Diagnostics"); correct to `O(v²)`.
"""
function dipole_ddot(models::NTuple{2,ParticleModel}, interaction, d)
    Δq = models[1].Q / models[1].M - models[2].Q / models[2].M
    return -Δq^2 * binding_constant(models, interaction) / (6π * d^2)
end

"""
    eccentricity_fit(t, d, omega)

The eccentricity of an orbit from its separation `d` at times `t`: the
least-squares fit `d(t) = c₀ + c₁t + c₂t² + A cos(ωt) + B sin(ωt)` at the
orbital frequency `omega`, a secular trend and the radial oscillation
about it, gives `e = √(A² + B²)/c₀` at the fit's start. Returns
`(; e, amplitude, trend)`, the trend `(c₀, c₁, c₂)`. The samples should
cover two orbits or more, or the oscillation and the trend's curvature
cannot be told apart.
"""
function eccentricity_fit(t::AbstractVector, d::AbstractVector, omega::Real)
    length(t) == length(d) || throw(DimensionMismatch(
        "$(length(t)) times for $(length(d)) separations"))
    length(t) >= 5 || throw(ArgumentError(
        "the fit has five parameters, got $(length(t)) samples"))
    t0 = first(t)
    τ = t .- t0
    A = hcat(one.(τ), τ, τ .^ 2, cos.(omega .* τ), sin.(omega .* τ))
    c = A \ d
    amplitude = hypot(c[4], c[5])
    return (; e=amplitude / c[1], amplitude, trend=(c[1], c[2], c[3]))
end

# The seven-patch ball's exact solution and its convergence driver.
#
# A plane wave `u = sin(k⋅x - |k|t + φ)` crosses every glued face of the
# ball, the cube–shell ones and the shell–shell quarter turns, and is the
# Dirichlet data on the outer sphere. It is the one solution of the free
# wave equation that needs no special functions and is the same expression
# in every patch. See "The seven-patch ball" in `CODE.md`.

"""
    PlaneWave{T}(k, φ)
    PlaneWave(k, φ)

The plane wave `u = sin(k⋅x - |k|t + φ)` of wave vector `k` (a `D`-tuple)
and phase `φ`, a solution of `∂ₜ²u = ∇²u` in physical coordinates. As
`wave(x, t)` it returns `(u, ∂₁u, …, ∂_D u)` at the physical position `x`,
the form of [`WaveProblem`](@ref)'s `boundary`; [`sevenpatch_exact`](@ref)
gives `u` and `v = ∂ₜu` at the nodes. It is `isbits`.
"""
struct PlaneWave{T,D}
    k::NTuple{D,T}
    ω::T
    φ::T
end

PlaneWave{T}(k, φ) where {T} =
    PlaneWave{T,length(k)}(T.(Tuple(k)), T(sqrt(sum(abs2, k))), T(φ))
PlaneWave(k, φ) = PlaneWave{float(promote_type(eltype(k), typeof(φ)))}(k, φ)

@inline plane_phase(w::PlaneWave, x, t) =
    sum(ntuple(d -> w.k[d] * x[d], length(w.k))) - w.ω * t + w.φ

@inline function (w::PlaneWave)(x, t)
    θ = plane_phase(w, x, t)
    s, c = sincos(θ)
    return (s, ntuple(d -> w.k[d] * c, length(w.k))...)
end

# `u` and `v` of a plane wave at logical `x` of patch `p`, for
# `fill_at_nodes!` through `WithPatch`.
struct PlaneWaveState{G,W,T}
    geometry::G
    wave::W
    t::T
end

@inline function (s::PlaneWaveState)(x, p, var)
    θ = plane_phase(s.wave, physical_coordinates(s.geometry, x, p), s.t)
    return var == 1 ? sin(θ) : -s.wave.ω * cos(θ)
end

"""
    sevenpatch_exact(geometry, wave::PlaneWave, t)

The plane wave `wave` at time `t` on the patches of `geometry`, `u` and
`v = ∂ₜu`, as a `WithPatch` callback for [`fill_at_nodes!`](@ref) and
[`initial_state`](@ref).
"""
sevenpatch_exact(geo::Geometry, wave::PlaneWave{T}, t) where {T} =
    WithPatch(PlaneWaveState(geo, wave, T(t)))

"""
    sevenpatch_errors(T=Float64; P, E, roots=(2, 2), Lc=1, R1=3,
                      k=(1.1, -0.7, 0.5), φ=0.3, cfl=1/2, tend=1, alg=RK4(),
                      truncate=true, refined=false, grading=0, Rm=nothing,
                      backend=CPU())

Evolve a plane wave ([`PlaneWave`](@ref)) on the seven-patch ball of
[`sevenpatch_forest`](@ref) to `tend`, with the wave itself as the
Dirichlet data on the outer sphere. Return the `√g`-weighted L2 errors of
`u` and `v` together and apart, the nodal L∞ error, the cube's physical
spacing `h` and the element width `hₑ = (P+1)h`, the step count, the
number of blocks and the initial and final energies.

The time step is [`oldg_timestep`](@ref), which on the ball divides each
block's spacing by its largest logical wave speed. `refined` refines the
cube ([`sevenpatch_forest`](@ref)), and `h` is then its finest level's;
`grading` grades the shells' radial elements ([`SevenPatchBall`](@ref)).
With a middle radius `Rm` the ball is instead the thirteen-patch one of
[`thirteenpatch_forest`](@ref), whose `roots` are `(n, m, m′)`, and
`grading` must be `0`. The run, and the exact solution it is compared
with, are on `backend`.
"""
function sevenpatch_errors(::Type{T}=Float64; P::Integer, E::Integer, roots=(2, 2),
                           Lc=1, R1=3, k=(1.1, -0.7, 0.5), φ=0.3, cfl=T(1//2), tend=1,
                           alg=RK4(), truncate::Bool=true, refined=false,
                           grading=0, Rm=nothing, backend::Backend=CPU()) where {T}
    forest, geo = ball_forest(T, P, E; roots=roots, Lc=Lc, Rm=Rm, R1=R1, refined=refined,
                              grading=grading)
    el = Element(T, P)
    wave = PlaneWave{T}(k, φ)
    problem = WaveProblem(forest, el; truncate=truncate, geometry=geo, boundary=wave,
                          backend=backend)
    t_end = T(tend)
    u0 = initial_state(problem, sevenpatch_exact(geo, wave, 0))
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps
    integ = wave_integrator(problem, u0, zero(T), t_end; dt=dt, alg=alg)
    solve!(integ)
    u = integ.u

    exact = wave_fieldset(T, forest, 2, (0, 0, 0), state_statesparity(Val(3)), P,
                          backend)
    fill_at_nodes!(sevenpatch_exact(geo, wave, t_end), exact, el)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = u .- uexact
    # The logical spacing is the same at every root count; the one returned
    # is the cube's physical spacing, `2Lc/n` per root.
    h = minimum_spacing(forest) * 2 * geo.Lc / geo.roots[1]
    return (l2=wave_norm(problem, err), l2u=wave_norm(problem, err; vars=1),
            l2v=wave_norm(problem, err; vars=2), linf=wave_norm(problem, err; p=Inf),
            h=h, hₑ=(P + 1) * h, nsteps=nsteps, dt=dt, tend=t_end,
            nblocks=nleaves(forest), energy0=wave_energy(problem, u0; t=zero(T)),
            energy=wave_energy(problem, u; t=t_end))
end

# The seven-patch ball, or with a middle radius `Rm` the thirteen-patch one.
function ball_forest(::Type{T}, P, E; roots, Lc, Rm, R1, refined, grading=0) where {T}
    Rm === nothing &&
        return sevenpatch_forest(T, P, E; roots=roots, Lc=Lc, R1=R1, refined=refined,
                                 grading=grading)
    iszero(grading) || throw(ArgumentError(
        "the thirteen-patch ball's shells are not graded, got grading = $grading"))
    return thirteenpatch_forest(T, P, E; roots=roots, Lc=Lc, Rm=Rm, R1=R1,
                                refined=refined)
end

"""
    OutgoingPulse{T}(w)
    OutgoingPulse(w)

The spherical wave `u = (f(t - r) - f(t + r))/r` with `f(s) = exp(-s²/w²)`,
a solution of `∂ₜ²u = ∇²u` that is regular at the origin. At `t = 0`
`u` vanishes and `v` is a pulse of width `w` about the origin; from then
on the wave is the outgoing `f(t - r)/r` up to the incoming
`f(t + r)/r`, which is below `exp(-R²/w²)` on a sphere of radius `R`. So
on the ball's outer sphere it satisfies Sommerfeld's condition
`∂ₜu + ∂ᵣu + u/r = 0` to that size, and a [`RadiativeBoundary`](@ref)
lets it leave. As `pulse(x, t)` it returns `(u, v)` at the physical
position `x`.
"""
struct OutgoingPulse{T}
    w::T
    OutgoingPulse{T}(w) where {T} = new{T}(T(w))
end

OutgoingPulse(w) = OutgoingPulse{float(typeof(w))}(w)

# `(f(t - r) - f(t + r))/r` and its time derivative. Below `r = 10⁻³ w` the
# quotient loses digits, and the Taylor series in `r` takes over:
# `-2 Σ_k f^{(2k+1)}(t) r^{2k}/(2k+1)!` and its derivative, through `r⁴`,
# whose next term is below `10⁻¹⁸` relative. `f^{(n)}(s) = (-1/w)ⁿ
# Hₙ(s/w) f(s)` with `Hₙ` the physicists' Hermite polynomials.
@inline function (pulse::OutgoingPulse{T})(x, t) where {T}
    w = pulse.w
    r = sqrt(sum(abs2, x))
    f(s) = exp(-(s / w)^2)
    df(s) = -2 * s / w^2 * f(s)
    if r > w / 1000
        return ((f(t - r) - f(t + r)) / r, (df(t - r) - df(t + r)) / r)
    end
    y = t / w
    H = (2y, 4y^2 - 2, 8y^3 - 12y, 16y^4 - 48y^2 + 12, 32y^5 - 160y^3 + 120y,
         64y^6 - 480y^4 + 720y^2 - 120)
    d(n) = (-1 / w)^n * H[n] * f(t)
    ρ2 = r^2
    u = -2 * (d(1) + ρ2 * d(3) / 6 + ρ2^2 * d(5) / 120)
    v = -2 * (d(2) + ρ2 * d(4) / 6 + ρ2^2 * d(6) / 120)
    return (u, v)
end

# `u` and `v` of the pulse at logical `x` of patch `p`, for
# `fill_at_nodes!` through `WithPatch`.
struct PulseState{G,W,T}
    geometry::G
    pulse::W
    t::T
end

@inline (s::PulseState)(x, p, var) =
    s.pulse(physical_coordinates(s.geometry, x, p), s.t)[var]

"""
    radiative_errors(T=Float64; P, E, roots=(2, 2), Lc=1, R1=3, w=1/2,
                     κ=1, inverse_r=true, cfl=1/2, tend=2, alg=RK4(),
                     truncate=true, refined=false, grading=0, Rm=nothing,
                     backend=CPU())

Evolve an [`OutgoingPulse`](@ref) of width `w` on the seven-patch ball of
[`sevenpatch_forest`](@ref) to `tend`, with a [`RadiativeBoundary`](@ref)
of `κ` and `inverse_r` on the outer sphere. Return the `√g`-weighted L2
errors of `u` and `v` together and apart, the nodal L∞ error, the cube's
physical spacing `h` and the element width `hₑ`, the step count, the
number of blocks and the initial and final energies, the boundary's own
included ([`wave_energy`](@ref)).

The pulse reaches the sphere `r = R1` at `t ≈ R1 - 2w` and has left it
by `t ≈ R1 + 2w`; what it leaves behind is the reflection. `refined`
refines the cube ([`sevenpatch_forest`](@ref)), and `h` is then its finest
level's; `grading` grades the shells' radial elements
([`SevenPatchBall`](@ref)). With a middle radius `Rm` the ball is instead
the thirteen-patch one of [`thirteenpatch_forest`](@ref), whose `roots`
are `(n, m, m′)`, and `grading` must be `0`. The run, and the exact
solution it is compared with, are on `backend`.
"""
function radiative_errors(::Type{T}=Float64; P::Integer, E::Integer, roots=(2, 2),
                          Lc=1, R1=3, w=T(1//2), κ=1, inverse_r::Bool=true,
                          cfl=T(1//2), tend=2, alg=RK4(), truncate::Bool=true,
                          refined=false, grading=0, Rm=nothing,
                          backend::Backend=CPU()) where {T}
    forest, geo = ball_forest(T, P, E; roots=roots, Lc=Lc, Rm=Rm, R1=R1, refined=refined,
                              grading=grading)
    el = Element(T, P)
    pulse = OutgoingPulse{T}(w)
    problem = WaveProblem(forest, el; truncate=truncate, geometry=geo,
                          boundary=RadiativeBoundary(; κ=T(κ), inverse_r=inverse_r),
                          backend=backend)
    t_end = T(tend)
    u0 = initial_state(problem, WithPatch(PulseState(geo, pulse, zero(T))))
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps
    integ = wave_integrator(problem, u0, zero(T), t_end; dt=dt, alg=alg)
    solve!(integ)
    u = integ.u

    exact = wave_fieldset(T, forest, 2, (0, 0, 0), state_statesparity(Val(3)), P,
                          backend)
    fill_at_nodes!(WithPatch(PulseState(geo, pulse, t_end)), exact, el)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = u .- uexact
    h = minimum_spacing(forest) * 2 * geo.Lc / geo.roots[1]
    return (l2=wave_norm(problem, err), l2u=wave_norm(problem, err; vars=1),
            l2v=wave_norm(problem, err; vars=2), linf=wave_norm(problem, err; p=Inf),
            h=h, hₑ=(P + 1) * h, nsteps=nsteps, dt=dt, tend=t_end,
            nblocks=nleaves(forest), energy0=wave_energy(problem, u0),
            energy=wave_energy(problem, u))
end

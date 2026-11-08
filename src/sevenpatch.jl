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
                      truncate=true)

Evolve a plane wave ([`PlaneWave`](@ref)) on the seven-patch ball of
[`sevenpatch_forest`](@ref) to `tend`, with the wave itself as the
Dirichlet data on the outer sphere. Return the `√g`-weighted L2 errors of
`u` and `v` together and apart, the nodal L∞ error, the cube's physical
spacing `h` and the element width `hₑ = (P+1)h`, the step count, the
number of blocks and the initial and final energies.

The time step is [`oldg_timestep`](@ref), which on the ball divides by
the largest logical wave speed.
"""
function sevenpatch_errors(::Type{T}=Float64; P::Integer, E::Integer, roots=(2, 2),
                           Lc=1, R1=3, k=(1.1, -0.7, 0.5), φ=0.3, cfl=T(1//2), tend=1,
                           alg=RK4(), truncate::Bool=true) where {T}
    forest, geo = sevenpatch_forest(T, P, E; roots=roots, Lc=Lc, R1=R1)
    el = Element(T, P)
    wave = PlaneWave{T}(k, φ)
    problem = WaveProblem(forest, el; truncate=truncate, geometry=geo, boundary=wave)
    t_end = T(tend)
    u0 = initial_state(problem, sevenpatch_exact(geo, wave, 0))
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps
    integ = wave_integrator(problem, u0, zero(T), t_end; dt=dt, alg=alg)
    solve!(integ)
    u = integ.u

    exact = wave_fieldset(T, forest, 2, (0, 0, 0), state_statesparity(Val(3)), P, CPU())
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

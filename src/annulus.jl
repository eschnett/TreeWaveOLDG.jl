# The rotating Bessel mode on the annulus, the exact solution the
# four-patch annulus is measured against (TreeAMR's M17 acceptance).
#
# On `r0 ≤ r ≤ r1` with solid (Neumann) walls,
#
#     u = R(kr) cos(mθ - kt),   R(z) = J_m(z) Y_m′(kr0) - Y_m(z) J_m′(kr0)
#
# solves the wave equation when `k` is a root of
# `J_m′(kr0) Y_m′(kr1) - J_m′(kr1) Y_m′(kr0)`, so that `R′` vanishes on
# both walls. The pattern turns at the angular speed `k/m`, so it carries
# flux through every glued face of the ring. See "The annulus" in
# `CODE.md`.

using SpecialFunctions: besselj, bessely

# `Z_m′(z)` for `Z = J` or `Y`, by `Z_m′ = (Z_{m-1} - Z_{m+1})/2`.
dbesselj(m, z) = (besselj(m - 1, z) - besselj(m + 1, z)) / 2
dbessely(m, z) = (bessely(m - 1, z) - bessely(m + 1, z)) / 2

annulus_cross(m, k, r0, r1) =
    dbesselj(m, k * r0) * dbessely(m, k * r1) - dbesselj(m, k * r1) * dbessely(m, k * r0)

"""
    annulus_wavenumber(m, n, r0, r1)

The `n`-th positive root `k` of `J_m′(kr0) Y_m′(kr1) - J_m′(kr1) Y_m′(kr0)`:
the wavenumber, and the angular frequency, of the Neumann mode of angular
number `m` with radial number `n` on the annulus `r0 ≤ r ≤ r1`. Bracketed
by a scan in steps of a fiftieth of `π/(r1 - r0)`, the roots' spacing far
out, and bisected to the last bit, in `Float64`.
"""
function annulus_wavenumber(m::Integer, n::Integer, r0::Real, r1::Real)
    m >= 0 && n >= 1 || throw(ArgumentError(
        "the mode needs m ≥ 0 and n ≥ 1, got m = $m and n = $n"))
    r0, r1 = Float64(r0), Float64(r1)
    0 < r0 < r1 || throw(ArgumentError("the annulus needs 0 < r0 < r1"))
    f(k) = annulus_cross(m, k, r0, r1)
    step = π / (r1 - r0) / 50
    a = step / 8
    fa = f(a)
    found = 0
    while true
        b = a + step
        fb = f(b)
        if fa == 0 || sign(fa) != sign(fb)
            found += 1
            if found == n
                lo, hi = a, b
                flo = fa
                while true
                    mid = (lo + hi) / 2
                    (mid == lo || mid == hi) && return lo
                    fm = f(mid)
                    if fm == 0
                        return mid
                    elseif sign(fm) == sign(flo)
                        lo, flo = mid, fm
                    else
                        hi = mid
                    end
                end
            end
        end
        a, fa = b, fb
    end
end

"""
    AnnulusMode{T}

The rotating mode `u = R(kr) cos(mθ - kt)`, `v = k R(kr) sin(mθ - kt)`, as
a callback `(x, p, v) -> value` of the logical position `x` of patch `p`:
wrap it in `WithPatch` for [`fill_at_nodes!`](@ref) or
[`initial_state`](@ref), as [`annulus_exact`](@ref) does. `R` is scaled to
`R(kr0) = 1`. It evaluates the Bessel functions in `Float64`, so it runs
on the CPU.
"""
struct AnnulusMode{T,G}
    geometry::G
    m::Int
    k::Float64
    jr0::Float64                 # J_m′(kr0)
    yr0::Float64                 # Y_m′(kr0)
    scale::Float64
    t::Float64
end

function (mode::AnnulusMode{T})(x, p, var) where {T}
    geo = mode.geometry
    r = Float64(annulus_radius(geo, x[2]))
    θ = Float64(annulus_angle(geo, x[1], p))
    kr = mode.k * r
    R = mode.scale * (besselj(mode.m, kr) * mode.yr0 - bessely(mode.m, kr) * mode.jr0)
    phase = mode.m * θ - mode.k * mode.t
    value = var == 1 ? R * cos(phase) : mode.k * R * sin(phase)
    return oftype(x[1], value)
end

"""
    annulus_exact(geometry::PolarAnnulus, m, k, t)

The rotating Neumann mode of angular number `m` and wavenumber `k` (see
[`annulus_wavenumber`](@ref)) at time `t` on the annulus of `geometry`,
as a `WithPatch` callback for [`fill_at_nodes!`](@ref) and
[`initial_state`](@ref).
"""
function annulus_exact(geo::PolarAnnulus{T}, m::Integer, k::Real, t::Real) where {T}
    k = Float64(k)
    r0 = Float64(geo.r0)
    jr0, yr0 = dbesselj(m, k * r0), dbessely(m, k * r0)
    # `J_m Y_m′ - Y_m J_m′ = 2/(πz)`, so `R(kr0) = 2/(π k r0)`.
    scale = π * k * r0 / 2
    return WithPatch(AnnulusMode{T,typeof(geo)}(geo, Int(m), k, jr0, yr0, scale,
                                                Float64(t)))
end

"""
    annulus_errors(T=Float64; P, E, roots=(2, 1), r0=1, r1=2, m=1, n=1, cfl=1/2,
                   tend=nothing, alg=RK4(), truncate=true)

Evolve the rotating Neumann mode ([`annulus_exact`](@ref)) on the
four-patch annulus of [`annulus_forest`](@ref) to `tend`, by default a
quarter turn of its pattern, `(π/2)m/k`, after which every point of it
has crossed a glued face. Return the `√g`-weighted L2 errors of `u` and
`v` together and apart, the nodal L∞ error, the radial spacing `h` in
physical units and the element width `hₑ = (P+1)h`, the step count, the
number of blocks, the wavenumber and the initial and final energies.

The time step is [`oldg_timestep`](@ref), which on the annulus divides by
the largest logical wave speed, at the inner wall.
"""
function annulus_errors(::Type{T}=Float64; P::Integer, E::Integer, roots=(2, 1),
                        r0=1, r1=2, m::Integer=1, n::Integer=1, cfl=T(1//2),
                        tend=nothing, alg=RK4(), truncate::Bool=true) where {T}
    forest, geo = annulus_forest(T, P, E; roots=roots, r0=r0, r1=r1)
    el = Element(T, P)
    problem = WaveProblem(forest, el; truncate=truncate, geometry=geo)
    k = annulus_wavenumber(m, n, r0, r1)
    t_end = tend === nothing ? T(π / 2 * max(m, 1) / k) : T(tend)
    u0 = initial_state(problem, annulus_exact(geo, m, k, 0))
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps
    integ = wave_integrator(problem, u0, zero(T), t_end; dt=dt, alg=alg)
    solve!(integ)
    u = integ.u

    exact = wave_fieldset(T, forest, 2, (0, 0), state_statesparity(Val(2)), P, CPU())
    fill_at_nodes!(annulus_exact(geo, m, k, t_end), exact, el)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = u .- uexact
    # The logical spacing is the same at every root count, so the one
    # returned is the physical radial spacing, `r′h`.
    h = minimum_spacing(forest) * geo.dr
    return (l2=wave_norm(problem, err), l2u=wave_norm(problem, err; vars=1),
            l2v=wave_norm(problem, err; vars=2), linf=wave_norm(problem, err; p=Inf),
            h=h, hₑ=(P + 1) * h, nsteps=nsteps, dt=dt, tend=t_end,
            nblocks=nleaves(forest), k=k, energy0=wave_energy(problem, u0),
            energy=wave_energy(problem, u))
end

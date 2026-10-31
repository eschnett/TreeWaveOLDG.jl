# The standing sine mode, the one initial condition of the prototype.
#
# On a periodic box of side L the separable standing mode
#
#     u(x,t) = cos(ωt) ∏ sin(2πm x_d / L),   ω = 2πm √D / L
#
# is an exact solution of the wave equation in any number of dimensions,
# so it measures the order of the discretization — in `h` at fixed `P`,
# and in `P` at fixed `h`. Taken from TreeWave unchanged, except that it is
# sampled at the Gauss–Lobatto nodes rather than at cell centres.

"""
Angular frequency of the `m`-th sine mode on a box of side `L`, built in
the type of `L` so that a `Float32` box does not get a `Float64`
frequency.
"""
function wave_omega(D, L, m)
    T = typeof(float(L))
    return 2 * T(π) * m * sqrt(T(D)) / L
end

"""
The exact solution, as an `(x, v) -> value` callback for
[`fill_at_nodes!`](@ref) or [`initial_state`](@ref). It is a kernel
argument, so it captures scalars only, and the product over dimensions is
an accumulating loop starting from `one(twoπ)` rather than `one(T)`.
"""
function wave_exact(D, L, m, t)
    T = typeof(float(L))
    ω = wave_omega(D, L, m)
    twoπ = 2 * T(π)
    return function (x, var)
        shape = one(twoπ)
        for d in 1:D
            shape *= sin(twoπ * m * x[d] / L)
        end
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
Angular frequency of the `m`-th reflected standing mode on a reflecting box
of side `L`, built in the type of `L`. It is the same `mπ√D/L` as the
plugged (separable) mode of a periodic box of side `2L`, which the even
reflection of `cos` reproduces on `[0,L]`.
"""
function reflect_omega(D, L, m)
    T = typeof(float(L))
    return T(π) * m * sqrt(T(D)) / L
end

"""
The exact solution on a reflecting box, as an `(x, v) -> value` callback
for [`fill_at_nodes!`](@ref) or [`initial_state`](@ref). The standing mode

    u(x,t) = cos(ωt) ∏_d cos(mπ x_d / L),   ω = mπ√D/L

satisfies `∂_n u = 0` on every wall of the box `[0,L]^D` (each cosine
has zero derivative at the two ends of its dimension), so it is the exact
Neumann (solid-wall) counterpart of [`wave_exact`](@ref). A `cos` needs
no mirror images, unlike the periodic box's `sin`, so the shape passes to
the wall as-is.
"""
function reflect_exact(D, L, m, t)
    T = typeof(float(L))
    ω = reflect_omega(D, L, m)
    return function (x, var)
        shape = one(T)
        for d in 1:D
            shape *= cospi(m * x[d] / L)
        end
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
    wave_forest(T, Val(D), P, E; roots=4, L=1, reflecting=false, refined=false)

A box of `roots^D` blocks, each holding `E^D` elements of order `P`, so
`N = E(P+1)` points per dimension. `E` has to be even, so that a block of
a refined forest covers whole elements of its parent; see
[`check_layout`](@ref TreeWaveOLDG.check_layout).

The box is periodic by default. With `reflecting=true` every face is a
solid (Neumann, `∂_n u = 0`) wall, declared to TreeAMR as a
reflection symmetry: each face is `reflecting` on the forest, and
[`WaveProblem`](@ref) derives the per-variable parities the wall imposes
(`u` and `v` even, `∂_d u` odd in `d`) for the field sets' ghosts.
Periodic remains the default; see "Reflecting walls" in `CODE.md`.

The box is uniform by default. `refined` makes it a static two-level
forest, as TreeWave's is: the blocks whose centres lie inside a region
are refined once, then the forest is balanced. `refined=true` takes the
middle half of the box in every dimension, `(L/4, 3L/4)^D`, which needs
`roots ≥ 3` to hold a block centre; a tuple of `D` intervals `(lo, hi)`
gives the region itself, so that refinement can reach a wall of a
reflecting box. A region that holds no block centre is refused.
"""
function wave_forest(::Type{T}, ::Val{D}, P::Integer, E::Integer; roots=4,
                     L=one(T), reflecting::Bool=false, refined=false) where {T,D}
    N = E * (P + 1)
    iseven(E) || throw(ArgumentError(
        "E = $E elements of order P = $P per block is odd, and E must be even: a " *
        "block of a refined forest covers half of its parent and has to cover " *
        "whole elements of it, so TreeAMR's element centering needs N = $N to be a " *
        "multiple of 2(P+1) = $(2(P + 1))"))
    periodic = ntuple(_ -> !reflecting, D)
    forest = if reflecting
        Forest{T}(ntuple(_ -> roots, D); N=N, periodic=periodic,
                  reflecting=ntuple(_ -> (true, true), D),
                  extents=ntuple(_ -> (zero(T), L), D))
    else
        Forest{T}(ntuple(_ -> roots, D); N=N, periodic=periodic,
                  extents=ntuple(_ -> (zero(T), L), D))
    end
    refined === false && return forest
    region = refined === true ? ntuple(_ -> (L / 4, 3L / 4), D) : refined
    length(region) == D && all(r -> length(r) == 2 && r[1] < r[2], region) ||
        throw(ArgumentError(
            "refined must be false, true, or a tuple of $D intervals (lo, hi) with " *
            "lo < hi, one per dimension; got $(repr(refined))"))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> region[d][1] < (ext[d][1] + ext[d][2]) / 2 < region[d][2], 1:D)
    end
    isempty(targets) && throw(ArgumentError(
        "no block centre lies inside the region $(repr(region)) to be refined; " *
        "with refined=true the region is the middle half of the box, which holds " *
        "a block centre only from roots = 3 on (roots = $roots)"))
    refine!(forest, targets)
    balance!(forest)
    return forest
end

"""
    wave_errors(T, Val(D); P, E, roots=4, L=1, m=1, cfl=1/2, periods=1/4,
                alg=RK4(), truncate=true, backend=CPU(), reflecting=false,
                refined=false)

Evolve the standing mode to `t_end` with fixed-step `alg` and return the
quadrature-weighted L2 and nodal L∞ errors of `u` and `v` together, the
mesh spacing `h`, the element width `hₑ = (P+1)h`, the step count and the
initial and final energies.

On a periodic box (the default) the mode is the sine [`wave_exact`](@ref);
on a reflecting box (`reflecting=true`) it is the Neumann standing mode
[`reflect_exact`](@ref), and the forest's walls are solid. Both measure the
order of the discretization. `refined` is passed to [`wave_forest`](@ref):
the box is uniform by default, and a static two-level forest otherwise.
The spacings returned are the finest level's.

The time step is [`oldg_timestep`](@ref), `cfl · hₑ/(P+1)²`. At RK4's
fourth order its error falls more slowly than the spatial error once
`P > 4`, so a convergence study at higher order needs a smaller `cfl` or
a higher-order `alg`; see "Measured results" in `CODE.md`.

`alg` is an explicit IMEXRungeKutta tableau. The run goes through
[`wave_integrator`](@ref), which installs [`truncate_stage!`](@ref) as its
stage and step limiter, so that truncation cannot be left out.
"""
function wave_errors(::Type{T}, ::Val{D}; P::Integer, E::Integer, roots=4, L=one(T),
                     m=1, cfl=T(1//2), periods=T(1//4), alg=RK4(),
                     truncate::Bool=true, backend::Backend=CPU(),
                     reflecting::Bool=false, refined=false) where {T,D}
    forest = wave_forest(T, Val(D), P, E; roots=roots, L=L, reflecting=reflecting,
                         refined=refined)
    el = Element(T, P)
    problem = WaveProblem(forest, el; truncate=truncate, backend=backend)

    exact_mode = reflecting ? reflect_exact : wave_exact
    omega = reflecting ? reflect_omega : wave_omega
    u0 = initial_state(problem, exact_mode(D, L, m, zero(T)))

    t_end = periods * 2 * T(π) / omega(D, L, m)
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps                          # land exactly on t_end

    integ = wave_integrator(problem, u0, zero(T), t_end; dt=dt, alg=alg)
    solve!(integ)
    u = integ.u

    exact = wave_fieldset(T, forest, 2, ntuple(_ -> 0, D), state_statesparity(Val(D)),
                          P, backend)
    fill_at_nodes!(exact_mode(D, L, m, t_end), exact, el)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = u .- uexact
    h = minimum_spacing(forest)
    return (l2=quadrature_norm(problem.fs, el, err),
            linf=quadrature_norm(problem.fs, el, err; p=Inf),
            h=h, hₑ=(P + 1) * h, nsteps=nsteps, nblocks=nleaves(forest),
            energy0=wave_energy(problem, u0), energy=wave_energy(problem, u))
end

"""
    reflect_errors(T, Val(D); kwargs...)

[`wave_errors`](@ref) on a reflecting box: a shorthand kept so the
standing Neumann mode's own name reads in a caller. Accepts every keyword
`wave_errors` does except `reflecting`, which is fixed at `true`;
`refined` with a region that reaches a wall refines the box there.
"""
reflect_errors(args...; kwargs...) = wave_errors(args...; reflecting=true, kwargs...)

wave_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; kwargs...)
reflect_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; reflecting=true, kwargs...)

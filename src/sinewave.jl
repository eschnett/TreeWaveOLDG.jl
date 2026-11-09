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

The box is uniform by default. `refined` makes it a static refined
forest, fixed before the run:

- `refined=true` refines the middle half of the box in every dimension,
  `(L/4, 3L/4)^D`, once, which needs `roots ≥ 3` to hold a block centre,
  as TreeWave's two-level forest does;
- a tuple of `D` intervals `(lo, hi)` refines that region once, so that
  refinement can reach a wall of a reflecting box;
- a vector of such regions, one per level above the root level, refines
  level by level: region `k` refines the leaves of level `k - 1` whose
  centres lie in it, and the forest is balanced after each.
  [`nested_cubes`](@ref) builds the nested boxes of the inspiral this
  way. An empty vector leaves the box uniform.

A region that holds no centre of a leaf of the level below it is refused.
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
    for (k, region) in enumerate(refinement_regions(refined, Val(D), L))
        targets = filter(forest.leaves) do key
            level(key) == k - 1 || return false
            ext = block_extent(forest, key)
            all(d -> region[d][1] < (ext[d][1] + ext[d][2]) / 2 < region[d][2], 1:D)
        end
        isempty(targets) && throw(ArgumentError(
            "no block centre of level $(k - 1) lies inside the region " *
            "$(repr(region)) that is to refine it to level $k; with refined=true " *
            "the region is the middle half of the box, which holds a block centre " *
            "only from roots = 3 on (roots = $roots)"))
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

# The regions `wave_forest`'s `refined` stands for, one per level above
# the root level: the middle half for `true`, a single region for a tuple
# of `D` intervals, and the regions themselves for a vector.
function refinement_regions(refined, ::Val{D}, L) where {D}
    isregion(r) = r isa Tuple && length(r) == D &&
                  all(i -> i isa Tuple && length(i) == 2 && i[1] < i[2], r)
    refined === true && return [ntuple(_ -> (L / 4, 3L / 4), D)]
    isregion(refined) && return [refined]
    refined isa AbstractVector && all(isregion, refined) && return refined
    throw(ArgumentError(
        "refined must be false, true, a tuple of $D intervals (lo, hi) with lo < hi, " *
        "one per dimension, or a vector of such tuples, one per level; got " *
        "$(repr(refined))"))
end

"""
    nested_cubes(Val(D), w₀, levels)

The regions of a forest of `levels` nested cubes about the centre of a
box of side `L = 2^levels · w₀`, as [`wave_forest`](@ref)'s `refined`
takes them (CODE.md, "Nested boxes"). The cube of level `k`
has the half-width `w_{levels-1-k} = 2^(levels-1-k) w₀`, so the finest
level, `levels - 1`, covers the cube of half-width `w₀` and the root
level the box, of half-width `w_{levels-1}`. Each level's element width
is twice the next finer one's, so every level has the same number of
elements across, `roots·E`. `levels = 1` gives no region, a uniform box.

So that every cube's faces lie on faces of the blocks of the level below,
the root blocks across the box must be a multiple of 4:
`wave_forest(T, Val(D), P, E; roots, L = 2^levels * w₀, refined =
nested_cubes(Val(D), w₀, levels))` with `roots % 4 == 0`.
"""
function nested_cubes(::Val{D}, w₀, levels::Integer) where {D}
    levels >= 1 || throw(ArgumentError("a forest has at least one level, got $levels"))
    w₀ > 0 || throw(ArgumentError(
        "the finest cube's half-width w₀ = $w₀ must be positive"))
    c = 2^(levels - 1) * w₀
    return [ntuple(_ -> (c - 2^(levels - 1 - k) * w₀, c + 2^(levels - 1 - k) * w₀), D)
            for k in 1:(levels - 1)]
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

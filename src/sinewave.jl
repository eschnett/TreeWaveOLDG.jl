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
    wave_forest(T, Val(D), P, E; roots=4, L=1, reflecting=false)

A uniform box of `roots^D` blocks, each holding `E^D` elements of order
`P`, so `N = E(P+1)` points per dimension. TreeAMR needs `N` even, so
for even `P` the number of elements per block has to be even too.

The box is periodic by default. With `reflecting=true` every face is a
solid (Neumann, `∂_n u = 0`) wall, declared to TreeAMR as a
reflection symmetry: each face is `reflecting` on the forest, and
[`WaveProblem`](@ref) derives the per-variable parities the wall imposes
(`u` and `v` even, `∂_d u` odd in `d`) for the field sets' ghosts.
Periodic remains the default; see "Reflecting walls" in `CODE.md`.

Uniform, because that is what OLDG can run on today; see "Scope and
non-goals" in `CODE.md`.
"""
function wave_forest(::Type{T}, ::Val{D}, P::Integer, E::Integer; roots=4,
                     L=one(T), reflecting::Bool=false) where {T,D}
    N = E * (P + 1)
    iseven(N) || throw(ArgumentError(
        "E = $E elements of order P = $P make N = $N points per block, and " *
        "TreeAMR needs N even: use an even E when P is even"))
    periodic = ntuple(_ -> !reflecting, D)
    if reflecting
        return Forest{T}(ntuple(_ -> roots, D); N=N, periodic=periodic,
                         reflecting=ntuple(_ -> (true, true), D),
                         extents=ntuple(_ -> (zero(T), L), D))
    else
        return Forest{T}(ntuple(_ -> roots, D); N=N, periodic=periodic,
                         extents=ntuple(_ -> (zero(T), L), D))
    end
end

"""
    wave_errors(T, Val(D); P, E, roots=4, L=1, m=1, cfl=1/2, periods=1/4,
                alg=RK4(), truncate=true, backend=CPU(), reflecting=false)

Evolve the standing mode to `t_end` with fixed-step `alg` and return the
quadrature-weighted L2 and nodal L∞ errors of `u` and `v` together, the
mesh spacing `h`, the element width `hₑ = (P+1)h`, the step count and the
initial and final energies.

On a periodic box (the default) the mode is the sine [`wave_exact`](@ref);
on a reflecting box (`reflecting=true`) it is the Neumann standing mode
[`reflect_exact`](@ref), and the forest's walls are solid. Both measure the
order of the discretization.

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
                     reflecting::Bool=false) where {T,D}
    forest = wave_forest(T, Val(D), P, E; roots=roots, L=L, reflecting=reflecting)
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

    exact = if reflecting
        FieldSet{T}(forest, 2; G=1, centering=cellcentered(D),
                    parity=state_statesparity(Val(D)), backend=backend)
    else
        FieldSet{T}(forest, 2; G=1, centering=cellcentered(D), backend=backend)
    end
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
`wave_errors` does except `reflecting`, which is fixed at `true`.
"""
reflect_errors(args...; kwargs...) = wave_errors(args...; reflecting=true, kwargs...)

wave_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; kwargs...)
reflect_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; reflecting=true, kwargs...)

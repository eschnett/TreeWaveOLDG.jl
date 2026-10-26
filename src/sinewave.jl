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
    wave_forest(T, Val(D), P, E; roots=4, L=1)

A uniform periodic box of `roots^D` blocks, each holding `E^D` elements of
order `P`, so `N = E(P+1)` points per dimension. TreeAMR needs `N` even,
so for even `P` the number of elements per block has to be even too.

Uniform, because that is what OLDG can run on today; see "Scope and
non-goals" in `CODE.md`.
"""
function wave_forest(::Type{T}, ::Val{D}, P::Integer, E::Integer; roots=4,
                     L=one(T)) where {T,D}
    N = E * (P + 1)
    iseven(N) || throw(ArgumentError(
        "E = $E elements of order P = $P make N = $N points per block, and " *
        "TreeAMR needs N even: use an even E when P is even"))
    return Forest{T}(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                     extents=ntuple(_ -> (zero(T), L), D))
end

"""
    wave_errors(T, Val(D); P, E, roots=4, L=1, m=1, cfl=1/2, periods=1/4,
                alg=RK4(), truncate=true, backend=CPU())

Evolve the sine mode to `t_end` with fixed-step `alg` and return the
quadrature-weighted L2 and nodal L∞ errors of `u` and `v` together, the
mesh spacing `h`, the element width `hₑ = (P+1)h`, the step count and the
initial and final energies.

The time step is [`oldg_timestep`](@ref), `cfl · hₑ/(P+1)²`. At RK4's
fourth order its error falls more slowly than the spatial error once
`P > 4`, so a convergence study at higher order needs a smaller `cfl` or
a higher-order `alg`; see "Measured results" in `CODE.md`.
"""
function wave_errors(::Type{T}, ::Val{D}; P::Integer, E::Integer, roots=4, L=one(T),
                     m=1, cfl=T(1//2), periods=T(1//4), alg=RK4(),
                     truncate::Bool=true, backend::Backend=CPU()) where {T,D}
    forest = wave_forest(T, Val(D), P, E; roots=roots, L=L)
    el = Element(T, P)
    problem = WaveProblem(forest, el; truncate=truncate, backend=backend)
    u0 = initial_state(problem, wave_exact(D, L, m, zero(T)))

    t_end = periods * 2 * T(π) / wave_omega(D, L, m)
    nsteps = ceilint(t_end / oldg_timestep(problem; cfl=cfl))
    dt = t_end / nsteps                          # land exactly on t_end

    prob = ODEProblem(wave_rhs!, u0, (zero(T), t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)
    u = sol.u[end]

    exact = FieldSet{T}(forest, 2; G=1, centering=cellcentered(D), backend=backend)
    fill_at_nodes!(wave_exact(D, L, m, t_end), exact, el)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = u .- uexact
    h = minimum_spacing(forest)
    return (l2=quadrature_norm(problem.fs, el, err),
            linf=quadrature_norm(problem.fs, el, err; p=Inf),
            h=h, hₑ=(P + 1) * h, nsteps=nsteps, nblocks=nleaves(forest),
            energy0=wave_energy(problem, u0), energy=wave_energy(problem, u))
end

wave_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; kwargs...)

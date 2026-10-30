# The radial blast wave, in two dimensions.
#
# A Gaussian peak released at rest in the middle of a periodic box. It
# splits into an ingoing and an outgoing half, the ingoing half passes
# through the origin, and what survives is a single expanding ring that
# loses amplitude to geometric spreading, with a wake trailing it: in 2D
# there is no Huygens principle. Taken from TreeWave's `src/blast.jl`,
# where it exercises a refined annulus that follows the ring; here, on a
# uniform forest, it is the case where the solution is not aligned with
# the elements, so it shows what the element faces do to a feature that
# crosses them. See "The blast wave" in `CODE.md`.

supergaussian(x, σ, n) = exp(-(x / σ)^(2n))

"""
    blast_initial(L, x₀, σ)

A radial Gaussian peak at rest: `u = exp(-(r/σ)²)`, `∂ₜu = 0`, with `r`
the periodically wrapped distance from `x₀`, as the `(x, var) -> value`
callback [`fill_at_nodes!`](@ref) and [`initial_state`](@ref) take.

Released at rest rather than outgoing, because a purely outgoing radial
wave is singular at the origin unless it vanishes there.
"""
function blast_initial(L, x₀, σ)
    return function (x, var)
        var == 1 || return zero(x[1])
        r2 = zero(x[1])
        for d in 1:2
            r2 += (wrap(x[d] - x₀[d] + L / 2, L) - L / 2)^2
        end
        return supergaussian(sqrt(r2), σ, 1)
    end
end

"""
    blast_reference(L, x₀, σ; rmax, nr=2001, nk=2000)

Everything the exact solution needs that does not depend on time, built
once. In two dimensions the radially symmetric solution is the inverse
Hankel transform of the initial profile, propagated mode by mode,

    u(r,t)  =  ∫₀^∞ f̂(k) J₀(kr) cos(kt) k dk
    ∂ₜu(r,t) = -∫₀^∞ f̂(k) J₀(kr) sin(kt) k² dk

with `f̂(k) = (σ²/2) exp(-k²σ²/4)` for the Gaussian. Only `cos(kt)` and
`sin(kt)` depend on `t`, so `J₀(kr)` is tabulated here on the `(r, k)`
grid and every later time is a contraction against it. The `k` integral
is cut at `13/σ`, where the envelope is `e^(-42)`, and taken by the
midpoint rule.

The table is `Float64` whatever the run's type, since `besselj0` has
hardware-float methods only; its quadrature error, about `1e-6`, is below
every discretization error measured against it.
"""
function blast_reference(L, x₀, σ; rmax, nr=2001, nk=2000)
    L, σ, rmax = tofloat64(L), tofloat64(σ), tofloat64(rmax)
    x₀ = map(tofloat64, x₀)
    dk = (13 / σ) / nk
    ks = [(j - 0.5) * dk for j in 1:nk]
    ws = [(σ^2 / 2) * exp(-(k * σ)^2 / 4) * k * dk for k in ks]
    rs = range(0.0, rmax; length=nr)
    J = Matrix{Float64}(undef, nr, nk)
    Threads.@threads for j in 1:nk
        k = ks[j]
        @inbounds for i in 1:nr
            J[i, j] = besselj0(k * rs[i])
        end
    end
    return (L=L, x₀=x₀, σ=σ, rs=rs, dr=step(rs), rmax=rmax, ks=ks, ws=ws, J=J)
end

"""
    blast_radial_table(ref, t)

The free-space radial profiles of `u` and `∂ₜu` at time `t`, on the
reference's radius grid. Threaded over radii, so that every `us[i]` is
summed over the modes in the same order whatever the thread count.
"""
function blast_radial_table(ref, t)
    nr, nk = size(ref.J)
    us = zeros(Float64, nr)
    vs = zeros(Float64, nr)
    rows = collect(Iterators.partition(1:nr, cld(nr, Threads.nthreads())))
    Threads.@threads for c in eachindex(rows)
        for j in 1:nk
            w = ref.ws[j]
            k = ref.ks[j]
            wc, wsn = w * cos(k * t), w * sin(k * t) * k
            @inbounds @simd for i in rows[c]
                Jij = ref.J[i, j]
                us[i] += Jij * wc
                vs[i] -= Jij * wsn
            end
        end
    end
    return (us=us, vs=vs)
end

"""
    blast_exact(T, ref, t)

The exact solution at time `t` as an `(x, var) -> value` callback, in
type `T`. The box is periodic, so it sums the radial profile over the
periodic images of the source out to the reference's `rmax`; a ring
reaches the corners of the box, where the nearest image is as far away as
the source. The profile is interpolated linearly in `r`.
"""
function blast_exact(::Type{T}, ref, t) where {T}
    tab = blast_radial_table(ref, tofloat64(t))
    us, vs = map(T, tab.us), map(T, tab.vs)
    L, rmax, dr = T(ref.L), T(ref.rmax), T(ref.dr)
    x₀ = map(T, ref.x₀)
    nimages = ceil(Int, ref.rmax / ref.L)
    return function (x, var)
        u = zero(x[1])
        v = zero(x[1])
        for j1 in (-nimages):nimages, j2 in (-nimages):nimages
            r = hypot(x[1] - x₀[1] - j1 * L, x[2] - x₀[2] - j2 * L)
            r < rmax || continue
            q = r / dr
            i = floorint(q) + 1
            s = q - (i - 1)
            u += (1 - s) * us[i] + s * us[i + 1]
            v += (1 - s) * vs[i] + s * vs[i + 1]
        end
        return var == 1 ? u : v
    end
end

"""
    uniform_blast(T=Float64; P, E, roots, L=1, σ=2/25, t_end=2/5, cfl=1/2,
                  alg=RK4(), truncate=true, observer=nothing, nsnapshots=21)

Evolve the blast wave on a uniform `roots × roots` box of `E × E` elements
of order `P` per block, and return the L∞ and L2 errors at `t_end`
against [`blast_exact`](@ref), the spacing, the number of nodes, and the
energy at the start and the end.

Pass `observer` to watch the run: it is called as `observer(p, t, u, ue)`
after the steps nearest `nsnapshots` evenly spaced times, the first and
the last included, with `t` the time actually reached, `p` the
[`WaveProblem`](@ref), `u` the state and `ue` the exact state at the same
nodes. The steps are the run's own, so watching it does not change it.
`u` is the integrator's state, overwritten by the next step: an observer
that keeps it must copy it. This is what the viewer in `bin/` uses.
"""
function uniform_blast(::Type{T}=Float64; P::Integer, E::Integer, roots::Integer,
                       L=one(T), σ=T(2//25), t_end=T(2//5), cfl=T(1//2), alg=RK4(),
                       truncate::Bool=true, observer=nothing,
                       nsnapshots::Integer=21) where {T}
    forest = wave_forest(T, Val(2), P, E; roots=roots, L=L)
    el = Element(T, P)
    p = WaveProblem(forest, el; truncate=truncate)
    x₀ = (L / 2, L / 2)
    u0 = initial_state(p, blast_initial(L, x₀, σ))
    ref = blast_reference(L, x₀, σ; rmax=t_end + 6σ)

    nsteps = ceilint(t_end / oldg_timestep(p; cfl=cfl))
    dt = t_end / nsteps
    integ = wave_integrator(p, u0, zero(T), t_end; dt=dt, alg=alg)

    exact = FieldSet{T}(forest, 2; G=1, centering=cellcentered(2))
    function exact_state(t)
        fill_at_nodes!(blast_exact(T, ref, t), exact, el)
        ue = statevector(exact)
        return gather!(ue, exact)
    end
    snapshots = observer === nothing ? Int[] :
                round.(Int, range(0, integ.nsteps; length=nsnapshots))
    for k in 0:integ.nsteps
        k > 0 && step!(integ)
        k in snapshots && observer(p, integ.t, integ.u, exact_state(integ.t))
    end

    u = integ.u
    err = u .- exact_state(t_end)
    h = minimum_spacing(forest)
    return (err=quadrature_norm(p.fs, el, err; p=Inf),
            l2=quadrature_norm(p.fs, el, err), h=h, nodes=nleaves(forest) * forest.N^2,
            nsteps=nsteps, energy0=wave_energy(p, u0), energy=wave_energy(p, u))
end

# The seven-patch ball: a plane wave on TreeAMR's forest of a cube and six
# shells, glued by the identity and by quarter turns, mapped to `|x| ≤ R1`
# by `SevenPatchBall`, with the wave itself as the Dirichlet data on the
# outer sphere, at the thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/sevenpatch.jl [--P=2,3] [--scales=1,2,3] \
#         [--truncate=yes,no] [--Lc=1] [--R1=3] [--E=2] [--cfls=0.5] [--tend=1] \
#         [--boundary=dirichlet] [--w=0.5] [--kappa=1] [--inverse_r=yes]
#
# With `--boundary=radiative` the solution is instead an outgoing pulse
# `(f(t - r) - f(t + r))/r` of width `w` (`OutgoingPulse`), and the outer
# sphere has a `RadiativeBoundary` of `κ = kappa`, with the `u/R` term
# unless `--inverse_r=no`; the pulse has left the ball of `R1 = 3` by
# `t ≈ 4`, and what is left then is the reflection.
#
# Runs against the package environment.
#
# Each patch has `(2s, 2s)` roots for each scale `s` (the cube `(2s)³`, a
# shell `2s` along each dimension), each block `E³` elements of order `P`;
# RK4 at `cfl` times `oldg_timestep`, which on the ball divides by the
# largest logical speed. The wave vector is `(1.1, −0.7, 0.5)` and the
# phase `0.3`, so the wave crosses every glue.
#
# Prints a header row, then one tab-separated row per run:
#
#     threads P truncate s cfl blocks nodes steps h l2 l2u l2v linf
#     energy_change seconds
#
# with `h` the cube's physical node spacing, the errors `√g`-weighted (L2)
# and nodal (L∞) against the exact wave, `energy_change` `1 - E₁/E₀`,
# which the Dirichlet data move, or the radiative boundary lets out, and `seconds` the run's wall-clock time
# without the first run's compilation (each `(P, truncate)` is run once at
# the smallest scale first to compile, and not printed).

using TreeWaveOLDG

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

list(name, default, parse_one) = [parse_one(strip(x)) for x in
                                  split(option(name, default), ",")]

function main()
    Ps = list("P", "2,3", s -> parse(Int, s))
    scales = list("scales", "1,2,3", s -> parse(Int, s))
    truncs = list("truncate", "yes,no", s -> s == "yes")
    cfls = list("cfls", "0.5", s -> parse(Float64, s))
    Lc = parse(Float64, option("Lc", "1"))
    R1 = parse(Float64, option("R1", "3"))
    E = parse(Int, option("E", "2"))
    tend = parse(Float64, option("tend", "1"))
    radiative = option("boundary", "dirichlet") == "radiative"
    w = parse(Float64, option("w", "0.5"))
    κ = parse(Float64, option("kappa", "1"))
    inverse_r = option("inverse_r", "yes") == "yes"
    run(; kw...) = radiative ?
                   radiative_errors(; w=w, κ=κ, inverse_r=inverse_r, kw...) :
                   sevenpatch_errors(; kw...)
    println(join(("threads", "P", "truncate", "s", "cfl", "blocks", "nodes", "steps", "h",
                  "l2", "l2u", "l2v", "linf", "energy_change", "seconds"), '\t'))
    flush(stdout)
    for P in Ps, truncate in truncs
        run(; P=P, E=E, roots=(2, 2), Lc=Lc, R1=R1, tend=tend / 64, truncate=truncate)
        for s in scales, cfl in cfls
            t0 = time()
            r = run(; P=P, E=E, roots=(2s, 2s), Lc=Lc, R1=R1, cfl=cfl, tend=tend,
                    truncate=truncate)
            seconds = time() - t0
            nodes = r.nblocks * (E * (P + 1))^3
            println(join((Threads.nthreads(), P, truncate ? "yes" : "no", s, cfl,
                          r.nblocks, nodes, r.nsteps, r.h, r.l2, r.l2u, r.l2v, r.linf,
                          1 - r.energy / r.energy0, round(seconds; digits=3)), '\t'))
            flush(stdout)
        end
    end
    return nothing
end

main()

# The four-patch annulus, TreeAMR's M17 acceptance: the rotating Neumann
# Bessel mode on TreeAMR's ring of four patches glued by the identity,
# mapped to `r0 ≤ r ≤ r1` by `PolarAnnulus`, at the thread count Julia was
# started with:
#
#     julia -t N,0 --project=. bin/annulus.jl [--P=2,3,4,5] [--scales=1,2,4,8] \
#         [--truncate=yes,no] [--m=2] [--n=1] [--r0=1] [--r1=2] [--E=2] \
#         [--cfls=0.5] [--turns=0.25]
#
# Runs against the package environment; `bin/symmetry/annulus.sh` runs it
# on a Symmetry node.
#
# Each patch has `(2s, s)` roots for each scale `s`, each block `E²`
# elements of order `P`; RK4 at `cfl` times `oldg_timestep`, which on the
# annulus divides by the largest logical speed, at the inner wall. The run
# goes to `turns` full turns of the mode's pattern, `2π turns · m/k`, so a
# quarter turn has carried every point of it across a glued face.
#
# Prints a header row, then one tab-separated row per run:
#
#     threads P truncate m n s cfl blocks nodes steps h l2 l2u l2v linf
#     energy_loss seconds
#
# with `h` the physical radial node spacing, the errors `√g`-weighted
# (L2) and nodal (L∞) against the exact mode, `energy_loss` `1 - E₁/E₀`,
# and `seconds` the run's wall-clock time without the first run's
# compilation (each `(P, truncate)` is run once at the smallest scale
# first to compile, and not printed).

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
    Ps = list("P", "2,3,4,5", s -> parse(Int, s))
    scales = list("scales", "1,2,4,8", s -> parse(Int, s))
    truncs = list("truncate", "yes,no", s -> s == "yes")
    cfls = list("cfls", "0.5", s -> parse(Float64, s))
    m = parse(Int, option("m", "2"))
    n = parse(Int, option("n", "1"))
    r0 = parse(Float64, option("r0", "1"))
    r1 = parse(Float64, option("r1", "2"))
    E = parse(Int, option("E", "2"))
    turns = parse(Float64, option("turns", "0.25"))
    k = annulus_wavenumber(m, n, r0, r1)
    tend = 2π * turns * max(m, 1) / k
    println(join(("threads", "P", "truncate", "m", "n", "s", "cfl", "blocks", "nodes",
                  "steps", "h", "l2", "l2u", "l2v", "linf", "energy_loss", "seconds"),
                 '\t'))
    flush(stdout)
    for P in Ps, truncate in truncs
        annulus_errors(; P=P, E=E, roots=(2, 1), r0=r0, r1=r1, m=m, n=n,
                       tend=tend / 64, truncate=truncate)
        for s in scales, cfl in cfls
            t0 = time()
            r = annulus_errors(; P=P, E=E, roots=(2s, s), r0=r0, r1=r1, m=m, n=n,
                               cfl=cfl, tend=tend, truncate=truncate)
            seconds = time() - t0
            nodes = r.nblocks * (E * (P + 1))^2
            println(join((Threads.nthreads(), P, truncate ? "yes" : "no", m, n, s, cfl,
                          r.nblocks, nodes, r.nsteps, r.h, r.l2, r.l2u, r.l2v, r.linf,
                          1 - r.energy / r.energy0, round(seconds; digits=3)), '\t'))
            flush(stdout)
        end
    end
    return nothing
end

main()

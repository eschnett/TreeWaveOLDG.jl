# What a step on the thirteen-patch ball costs, and its parts, at the
# thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/ball_benchmark.jl [--E=4] [--Lc=4] \
#         [--Rm=12] [--R1=100] [--scan=n:m:m2,n:m:m2,…] [--reps=5]
#
# Like `bin/benchmark.jl`, which times a uniform box, this runs against the
# package environment. The forest is `thirteenpatch_forest`'s ball at
# `P = 3`, `Float64`, blocks of `E³` elements, with the radiative sphere
# (`RadiativeBoundary()`) and truncation on: the cube `[−Lc, Lc]³` of `n³`
# roots, six inner shells to `Rm` of `m` radial roots, six outer shells to
# `R1` of `m2`. `--scan` runs the table once per `n:m:m2` in one process,
# so that compilation is paid once; varying `m` and `m2` at fixed `n`
# separates the three layers' costs per node (a least-squares fit, done
# afterwards from the rows).
#
# The defaults are the benchmark ball: 16 elements across each patch
# angularly, `hₑ = 1/2` in the cube and radially in both layers of
# shells, so `Lc = 4`, `Rm = 3Lc`, `R1 = 100`, roots `(4, 4, 44)`: 4672
# blocks of `16³`, 1.9·10⁷ nodes, nine tenths of them in the outer shells.
#
# Prints a header row, then one tab-separated row per quantity:
#
#     threads E n m m2 quantity value
#
# Sizes: `nodes`, `nodes_cube`, `nodes_inner`, `nodes_outer`, `blocks`.
# Set-up, seconds: `build_s` (the forest and `WaveProblem`, the truncation
# tables included). Memory: `problem_bytes` (`Base.summarysize`). The step:
# `dt`, and `dt_flat`, the step a flat box of the cube's spacing would
# take, so that `dt/dt_flat` is what the shells' speeds cost. Times, in
# seconds, the least of `reps` after a warm-up:
#
# - `rhs`: one `wave_rhs!`; `scatter_u`, `ghosts_u` (with the radiative
#   `u` ghosts), `gradient`, `ghosts_grad` (with the radiative gradient
#   ghosts) and `laplacian` are its parts, timed alone;
# - `truncate`: one `truncate_state!`;
# - `step`: one RK4 step of `wave_integrator`;
# - `stage`: an RK4 step with an empty right-hand side and no limiters.

using TreeWaveOLDG
using TreeWaveOLDG: scatter_u_kernel!, cart_gradient_kernel!, cart_laplacian_kernel!,
                    map_blocks!, fill_ghosts!, statearray, nblocks, radiative_u!,
                    radiative_grad!, statevector, IMEXProblem, RK4, init, step!

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function phases(E, n, m, m2, Lc, Rm, R1, reps)
    P = 3
    build = @elapsed begin
        forest, geo = thirteenpatch_forest(Float64, P, E; roots=(n, m, m2), Lc=Lc, Rm=Rm,
                                           R1=R1)
        p = WaveProblem(forest, Element(Float64, P); geometry=geo,
                        boundary=RadiativeBoundary())
    end
    u = statevector(p.fs)
    u .= sin.(1:length(u))
    truncate_state!(u, p)
    du = similar(u)
    ua, dua = statearray(u, p.fs), statearray(du, p.fs)
    metric = p.metric
    dt = oldg_timestep(p; cfl=1 // 2)
    N = E * (P + 1)
    hcube = 2Lc / (n * E)
    dt_flat = (1 // 2) * hcube / (P + 1)^2
    t1 = 10^6 * dt
    integ = wave_integrator(p, copy(u), 0.0, t1; dt=dt)
    empty!(du, u, p, t) = nothing
    stage = init(IMEXProblem(empty!, nothing, copy(u), (0.0, t1), p), RK4(); dt=dt,
                 partition=state_partition(p.fs, u))
    w = copy(u)
    rows = [
        "rhs" => best(() -> wave_rhs!(du, u, p, 0.0), reps),
        "scatter_u" => best(() -> map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, ua,
                                              p.valD, Val(p.ufs.G)), reps),
        "ghosts_u" => best(() -> (fill_ghosts!(p.ufs, p.uschedule);
                                  radiative_u!(p, p.boundary)), reps),
        "gradient" => best(() -> map_blocks!(cart_gradient_kernel!, p.ufs,
                                             p.cartgrad.work, p.ufs.work, p.spacings,
                                             p.element, metric.coef, metric.flat,
                                             p.valD, Val(p.ufs.G)), reps),
        "ghosts_grad" => best(() -> (fill_ghosts!(p.cartgrad, p.cartschedule);
                                     radiative_grad!(p, p.boundary, ua)), reps),
        "laplacian" => best(() -> map_blocks!(cart_laplacian_kernel!, p.fs, dua, ua,
                                              p.cartgrad.work, p.spacings, p.element,
                                              metric.coef, metric.flat, p.valD), reps),
        "truncate" => best(() -> truncate_state!(w, p), reps),
        "step" => best(() -> step!(integ), reps),
        "stage" => best(() -> step!(stage), reps),
    ]
    sizes = [
        "nodes" => length(u) ÷ 2,
        "nodes_cube" => (n * N)^3,
        "nodes_inner" => 6 * n^2 * m * N^3,
        "nodes_outer" => 6 * n^2 * m2 * N^3,
        "blocks" => nblocks(p.fs),
        "build_s" => build,
        "problem_bytes" => Base.summarysize(p),
        "dt" => dt,
        "dt_flat" => dt_flat,
    ]
    return vcat(sizes, rows)
end

function main()
    E = parse(Int, option("E", "4"))
    Lc = parse(Float64, option("Lc", "4"))
    Rm = parse(Float64, option("Rm", "12"))
    R1 = parse(Float64, option("R1", "100"))
    reps = parse(Int, option("reps", "5"))
    scan = [parse.(Int, split(s, ":")) for s in split(option("scan", "4:4:44"), ",")]
    println(join(("threads", "E", "n", "m", "m2", "quantity", "value"), '\t'))
    for (n, m, m2) in scan
        for (q, v) in phases(E, n, m, m2, Lc, Rm, R1, reps)
            println(join((Threads.nthreads(), E, n, m, m2, q, v), '\t'))
        end
        flush(stdout)
    end
    return nothing
end

main()

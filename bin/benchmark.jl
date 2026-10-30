# What a step costs, and its parts, at the thread count Julia was started
# with:
#
#     julia -t N,0 --project=. bin/benchmark.jl [--D=3] [--P=3] \
#         [--scan=E:roots,E:roots,…] [--reps=10]
#
# Note the `--project=.`: this script runs against the *package*
# environment rather than `bin/Project.toml`, which exists for the viewer;
# a compute node should not build Cairo to time a stencil.
#
# A uniform periodic box of `roots^D` blocks of `E^D` elements of order
# `P`, the standing sine mode, `Float64`. `--scan` runs the table once per
# `E:roots` pair in one process, so that compilation is paid once. Prints
# a header row, then one tab-separated row per phase:
#
#     threads D P E roots blocks nodes phase seconds
#
# Each time is the least of `reps` after a warm-up. The phases:
#
# - `rhs`: one `wave_rhs!`; `scatter_u`, `ghosts_u`, `gradient`,
#   `ghosts_grad` and `laplacian` are its five parts, timed alone;
# - `truncate`: one `truncate_state!`;
# - `step`: one RK4 step of `wave_integrator`, four of each of the above
#   plus the stage arithmetic;
# - `stage` and `stage_broadcast`: an RK4 step with an empty right-hand
#   side and no limiters, so the stage arithmetic alone, by owner and
#   broadcast.
#
# `bin/symmetry/benchmark.sh` runs it at several thread counts on a
# Symmetry node. "Threads on Symmetry" in `CODE.md` has the tables.

using TreeWaveOLDG, TreeAMR
using TreeWaveOLDG: scatter_u_kernel!, gradient_kernel!, laplacian_kernel!
using IMEXRungeKutta: IMEXProblem, RK4, init, step!

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function phases(D, P, E, roots, reps)
    p = WaveProblem(wave_forest(Float64, Val(D), P, E; roots=roots), Element(Float64, P))
    u = initial_state(p, wave_exact(D, 1.0, 1, 0.0))
    du = similar(u)
    ua, dua = statearray(u, p.fs), statearray(du, p.fs)
    grads = map(g -> g.work, p.grads)
    dt = oldg_timestep(p; cfl=1 // 2)
    t1 = 10^6 * dt                         # far more steps than are taken
    integ = wave_integrator(p, u, 0.0, t1; dt=dt)
    empty!(du, u, p, t) = nothing
    stage = init(IMEXProblem(empty!, nothing, u, (0.0, t1), p), RK4(); dt=dt,
                 partition=state_partition(p.fs, u))
    stage_broadcast = init(IMEXProblem(empty!, nothing, u, (0.0, t1), p), RK4(); dt=dt)
    w = copy(u)
    return [
        "rhs" => best(() -> wave_rhs!(du, u, p, 0.0), reps),
        "scatter_u" => best(() -> map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, ua,
                                              p.valD, Val(p.ufs.G)), reps),
        "ghosts_u" => best(() -> fill_ghosts!(p.ufs, p.uschedule), reps),
        "gradient" => best(() -> map_blocks!(gradient_kernel!, p.ufs, grads, p.ufs.work,
                                             p.spacings, p.element, p.valD,
                                             Val(p.ufs.G)), reps),
        "ghosts_grad" => best(() -> foreach(fill_ghosts!, p.grads, p.gradschedules),
                              reps),
        "laplacian" => best(() -> map_blocks!(laplacian_kernel!, p.fs, dua, ua, grads,
                                              p.spacings, p.element, p.valD), reps),
        "truncate" => best(() -> truncate_state!(w, p), reps),
        "step" => best(() -> step!(integ), reps),
        "stage" => best(() -> step!(stage), reps),
        "stage_broadcast" => best(() -> step!(stage_broadcast), reps),
    ], nblocks(p.fs), length(u) ÷ 2
end

function main()
    D = parse(Int, option("D", "3"))
    P = parse(Int, option("P", "3"))
    reps = parse(Int, option("reps", "10"))
    scan = [parse.(Int, split(s, ":")) for s in split(option("scan", "4:8"), ",")]
    println(join(("threads", "D", "P", "E", "roots", "blocks", "nodes", "phase",
                  "seconds"), '\t'))
    for (E, roots) in scan
        rows, blocks, nodes = phases(D, P, E, roots, reps)
        for (phase, seconds) in rows
            println(join((Threads.nthreads(), D, P, E, roots, blocks, nodes, phase,
                          seconds), '\t'))
        end
        flush(stdout)
    end
    return nothing
end

main()

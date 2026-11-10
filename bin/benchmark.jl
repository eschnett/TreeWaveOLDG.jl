# What a step costs, and its parts, at the thread count Julia was started
# with:
#
#     julia -t N,0 --project=. bin/benchmark.jl [--D=3] [--P=3] \
#         [--scan=E:roots,E:roots,…] [--reps=10] [--backend=cpu|cuda|metal] \
#         [--type=f64|f32]
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
#     threads backend type D P E roots blocks nodes phase seconds
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
# With `--backend=cuda` or `metal` the problem is built on that device
# (`bin/backend.jl`), which needs an environment that adds the device
# package, and every time waits for the device to finish; by owner and
# broadcast are then the same, IMEXRungeKutta's broadcast path.
#
# `bin/symmetry/benchmark.sh` runs it at several thread counts on a
# Symmetry node. "Threads on Symmetry" in `CODE.md` has the tables.

# Everything through TreeWaveOLDG, which has TreeAMR's and IMEXRungeKutta's
# names in scope, so that an environment that only `develop`s the package
# can run this.
include("backend.jl")
using TreeWaveOLDG: scatter_u_kernel!, gradient_kernel!, laplacian_kernel!,
                    map_blocks!, fill_ghosts!, statearray, nblocks,
                    IMEXProblem, RK4, init, step!

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

function phases(backend, ::Type{T}, D, P, E, roots, reps) where {T}
    best(f, reps) = Main.best(f, backend, reps)
    p = WaveProblem(wave_forest(T, Val(D), P, E; roots=roots), Element(T, P);
                    backend=backend)
    u = initial_state(p, wave_exact(D, one(T), 1, zero(T)))
    du = similar(u)
    ua, dua = statearray(u, p.fs), statearray(du, p.fs)
    grads = map(g -> g.work, p.grads)
    dt = oldg_timestep(p; cfl=T(1 // 2))
    t1 = 10^6 * dt                         # far more steps than are taken
    integ = wave_integrator(p, u, zero(T), t1; dt=dt)
    empty!(du, u, p, t) = nothing
    stage = init(IMEXProblem(empty!, nothing, u, (zero(T), t1), p), RK4(); dt=dt,
                 partition=state_partition(p.fs, u))
    stage_broadcast = init(IMEXProblem(empty!, nothing, u, (zero(T), t1), p), RK4();
                           dt=dt)
    w = copy(u)
    return [
        "rhs" => best(() -> wave_rhs!(du, u, p, zero(T)), reps),
        "scatter_u" => best(() -> map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, ua,
                                              p.valD, Val(p.ufs.G)), reps),
        "ghosts_u" => best(() -> fill_ghosts!(p.ufs, p.uschedule), reps),
        "gradient" => best(() -> map_blocks!(gradient_kernel!, p.ufs, grads, p.ufs.work,
                                             p.spacings, p.element, nothing, p.valD,
                                             Val(p.ufs.G)), reps),
        "ghosts_grad" => best(() -> foreach(fill_ghosts!, p.grads, p.gradschedules),
                              reps),
        "laplacian" => best(() -> map_blocks!(laplacian_kernel!, p.fs, dua, ua, grads,
                                              p.spacings, p.element, nothing,
                                              p.valD), reps),
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
    bname, tname = option("backend", "cpu"), option("type", "f64")
    println(join(("threads", "backend", "type", "D", "P", "E", "roots", "blocks",
                  "nodes", "phase", "seconds"), '\t'))
    withbackend(bname, tname) do backend, T
        for (E, roots) in scan
            rows, blocks, nodes = phases(backend, T, D, P, E, roots, reps)
            for (phase, seconds) in rows
                println(join((Threads.nthreads(), bname, tname, D, P, E, roots, blocks,
                              nodes, phase, seconds), '\t'))
            end
            flush(stdout)
        end
    end
    return nothing
end

main()

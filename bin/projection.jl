# What the transfer costs, at the thread count Julia was started with:
#
#     julia -t N,0 --project=. bin/projection.jl [--hdiv=8] [--rnear=1] \
#         [--order=3] [--reps=1] [--offgrid]
#
# Like `bin/benchmark.jl`, this runs against the package environment.
#
# The first run of INSPIRAL.md ("Parameters for a first run"): two
# opposite charges `q² = 0.005`, `ℓ = 10⁻³`, `M = 1`, on a circle of
# diameter `d = 1` at `v = 0.2`, the window `R₁ = 1.25`, `R₂ = 2.5` of
# continuity `P + 1`, `P = 3`, and `hₑ = d/hdiv`, in a periodic box of
# side 5 that just holds the window. The particles sit on element
# corners, as the box's centre does; `--offgrid` moves both by
# `(0.31, 0.17, 0.43)hₑ`, where a particle generically is. Prints a header
# row, then one
# tab-separated row per quantity:
#
#     threads hdiv rnear quantity value
#
# Each time is the least of `reps` after a warm-up, in seconds. The rows:
#
# - `nodes`, `window_nodes`: the box's nodes and those in elements that
#   meet the window's ball;
# - `far_elements`, `graded_elements`, `near_elements`: the window's
#   elements by rule (the far rule beyond `rfar = 9`, the graded rule
#   from `rnear` to there, the nested rule within `rnear`, all in units of
#   `hₑ`), and `far_points`, `graded_points`, `near_points`, the source
#   evaluations each rule makes;
# - `source`: one evaluation of the source, both particles, in the
#   window's inner ball;
# - `projection`: one `project_source!`, and its parts `plan`, `far`,
#   `graded` and `near`;
# - `rhs`: one `wave_rhs!` over the box, and `rhs_window`, its share on the
#   window's nodes;
# - `pairing_rho`, `pairing_gradrho`: the pairing of `ρ` and `∇ρ` with a
#   state over the support's elements, and `fold_rho`, the same through a
#   projection over the mesh and a `mesh_fold`, which it replaces.
#
# "Quadrature" in `CODE.md` has the numbers.

using TreeWaveOLDG
using TreeWaveOLDG: SVector, ForwardDiff, statevector, statearray, mesh_fold, nblocks,
                    projection_plan, project_far!, project_graded!, project_near!,
                    element_class, element_box, node_weight, FAR_ELEMENT

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function measure(hdiv, rnear, order, reps, offgrid)
    P = 3
    E = 4
    L = 5.0
    hₑ = 1 / hdiv
    roots = round(Int, L / (E * hₑ))
    forest = wave_forest(Float64, Val(3), P, E; roots=roots, L=L)
    p = WaveProblem(forest, Element(Float64, P))
    t = TransferTables(Float64, P)

    q = sqrt(0.005)
    Ω = 0.4
    c = SVector(L / 2, L / 2, L / 2)
    m₁ = ParticleModel(; Q=q, ℓ=1e-3, M=1.0)
    m₂ = ParticleModel(; Q=-q, ℓ=1e-3, M=1.0)
    r = SVector(0.5, 0.0, 0.0)
    v = SVector(0.0, Ω / 2, 0.0)
    a = -Ω^2 * r
    off = offgrid ? SVector(0.31, 0.17, 0.43) * hₑ : zero(c)
    tr₁ = Trajectory(; z=c + r + off, v=v, a=a, ȧ=-Ω^2 * v, ä=-Ω^2 * a, a⃛=Ω^4 * v)
    tr₂ = Trajectory(; z=c - r + off, v=-v, a=-a, ȧ=Ω^2 * v, ä=Ω^2 * a, a⃛=-Ω^4 * v)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    S = ParticleSource((m₁, m₂), w, (tr₁, tr₂); order=order)
    particles = (ParticleSupport(m₁, tr₁), ParticleSupport(m₂, tr₂))
    ball = (c, w.R₂)

    plan = projection_plan(p, t, particles, ball, rnear)
    nfar = 0
    nwindow = 0
    for b in 1:nblocks(p.fs), J in CartesianIndices((E, E, E))
        lo, hi = element_box(plan.origins[b], plan.spacings[b], Tuple(J) .- 1, Val(P + 1),
                             Val(3))
        class = element_class(lo, hi, plan.region, particles, plan.rf)
        nfar += class == FAR_ELEMENT
        nwindow += class != 0
    end
    count = Threads.Atomic{Int}(0)
    counted(x) = (Threads.atomic_add!(count, 1); S(x))
    du = statevector(p.fs)
    dua = statearray(du, p.fs)
    project_near!(dua, plan, t, counted, -1.0, 2)
    near_points = count[]
    count[] = 0
    project_graded!(dua, plan, t, counted, -1.0, 2)
    graded_points = count[]

    u = initial_state(p, wave_exact(3, L, 1, 0.0))
    scratch = projection_scratch(p, t)
    xs = [c + SVector(0.3 * cos(k), 0.4 * sin(2k), 0.2 * cos(3k)) for k in 1:1000]
    ρ(x) = particle_density(m₁, x, tr₁.z, tr₁.v)
    ∇ρ(x) = ForwardDiff.gradient(ρ, x)
    fold() = (du .= 0;
              project_source!(du, p, t, ρ, particles[1:1]; ball=(tr₁.z, 2m₁.ℓ),
                              factor=1.0, var=1, rnear=rnear, scratch=scratch);
              mesh_fold((acc, pt) -> acc + node_weight(pt.I, pt.h, p.element, Val(3)) *
                                           pt.u[1], +, 0.0, p.fs, du .* u; vars=1))
    nodes = length(u) ÷ 2
    rhs = best(() -> wave_rhs!(du, u, p, 0.0), reps)
    return Pair{String,Any}[
        "nodes" => nodes,
        "window_nodes" => nwindow * (P + 1)^3,
        "far_elements" => nfar,
        "graded_elements" => sum(length, plan.graded),
        "near_elements" => length(plan.near),
        "far_points" => nfar * length(t.gauss)^3,
        "graded_points" => graded_points,
        "near_points" => near_points,
        "source" => best(() -> sum(S, xs), reps) / length(xs),
        "projection" => best(() -> project_source!(du, p, t, S, particles; ball=ball,
                                                   rnear=rnear, scratch=scratch), reps),
        "plan" => best(() -> projection_plan(p, t, particles, ball, rnear), reps),
        "far" => best(() -> project_far!(dua, plan, t, S, scratch, -1.0, 2), reps),
        "graded" => best(() -> project_graded!(dua, plan, t, S, -1.0, 2), reps),
        "near" => best(() -> project_near!(dua, plan, t, S, -1.0, 2), reps),
        "rhs" => rhs,
        "rhs_window" => rhs * nwindow * (P + 1)^3 / nodes,
        "pairing_rho" => best(() -> pairing(p, t, u, ρ, particles[1]), reps),
        "pairing_gradrho" => best(() -> pairing(p, t, u, ∇ρ, particles[1]), reps),
        "fold_rho" => best(fold, reps),
    ]
end

function main()
    hdiv = parse(Int, option("hdiv", "8"))
    rnear = parse(Float64, option("rnear", "1"))
    order = parse(Int, option("order", "3"))
    reps = parse(Int, option("reps", "1"))
    offgrid = "--offgrid" in ARGS
    println(join(("threads", "hdiv", "rnear", "quantity", "value"), '\t'))
    for (name, value) in measure(hdiv, rnear, order, reps, offgrid)
        println(join((Threads.nthreads(), hdiv, rnear, name, value), '\t'))
        flush(stdout)
    end
    return nothing
end

main()

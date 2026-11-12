# The far rule of the source projection on a device, the gate for moving
# the particles there (`PLAN.md`, "Particles on a device"):
#
#     julia -t N,0 --project=<env> bin/far_rule.jl --backend=cuda [--hdiv=8] \
#         [--order=3] [--reps=3] [--offgrid]
#
# The source of `bin/projection.jl`'s first run, two opposite charges on a
# circle of diameter `d = 1` at `v = 0.2`, in the periodic box of side 5
# that holds the window, `hₑ = d/hdiv`, `P = 3`, in `Float64`, which the
# source needs and Metal does not have. `project_far!`'s two kernels, the
# source at the far elements' Gauss points and their contraction to the
# nodes, are already KernelAbstractions kernels; the other passes run on the
# host. This runs the far rule on `--backend` and on the host, in the same
# process, and compares them. Prints a header row, then one tab-separated
# row per quantity:
#
#     threads backend hdiv order quantity value
#
# - `nodes`, `far_points`: the box's nodes and the far rule's source
#   evaluations;
# - `far_device`, `far_host`: one far rule, seconds, the least of `reps`
#   after a warm-up, and `far_ns_per_point` on the device; and its two
#   kernels alone on the device, `far_source_device`, the source at the
#   Gauss points, and `far_contract_device`, their contraction;
# - `far_transition_points`: the far rule's points in the window's
#   transition, `R₁ < |x - centre| < R₂`, where the source carries `∇W`;
# - `rhs_device`: one `wave_rhs!` over the box on the device, for scale;
# - `far_relative_difference`: the device's far rule against the host's,
#   `‖dev − host‖/‖host‖` over the box;
# - `far_error`: instead of the device's rows, the error if the device
#   refuses the kernels (the source, about 26 KB, is a kernel argument).

include("backend.jl")
using TreeWaveOLDG: SVector, statevector, statearray, projection_plan, project_far!,
                    element_class, element_box, nblocks, FAR_ELEMENT, device_source,
                    source_points_kernel!, contract_kernel!, TreeAMR
using LinearAlgebra: norm

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

function far_setup(backend, hdiv, order, offgrid)
    P, E, L = 3, 4, 5.0
    hₑ = 1 / hdiv
    roots = round(Int, L / (E * hₑ))
    forest = wave_forest(Float64, Val(3), P, E; roots=roots, L=L)
    p = WaveProblem(forest, Element(Float64, P); backend=backend)
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
    tr₁ = Trajectory(; z=c + r + off, v=v, a=a, ȧ=-Ω^2 * v, ä=-Ω^2 * a, a⃛=Ω^4 * v,
                     a⃜=Ω^4 * a)
    tr₂ = Trajectory(; z=c - r + off, v=-v, a=-a, ȧ=Ω^2 * v, ä=Ω^2 * a, a⃛=-Ω^4 * v,
                     a⃜=-Ω^4 * a)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    S = ParticleSource((m₁, m₂), w, (tr₁, tr₂); order=order)
    particles = (ParticleSupport(m₁, tr₁), ParticleSupport(m₂, tr₂))
    plan = projection_plan(p, t, particles, (c, w.R₂), 1.0)
    du = statevector(p.fs)
    fill!(du, 0)
    return (; p, t, S, plan, du, dua=statearray(du, p.fs),
            scratch=projection_scratch(p, t))
end

function measure(backend, hdiv, order, reps, offgrid)
    host = far_setup(CPU(), hdiv, order, offgrid)
    dev = far_setup(backend, hdiv, order, offgrid)
    far!(s) = project_far!(s.dua, s.plan, s.t, s.S, s.scratch, -1.0, 2)
    P1 = 4
    E = host.p.fs.forest.N ÷ P1
    nfar = 0
    for b in 1:nblocks(host.p.fs), J in CartesianIndices((E, E, E))
        lo, hi = element_box(host.plan.origins[b], host.plan.spacings[b], Tuple(J) .- 1,
                             Val(P1), Val(3))
        nfar += element_class(lo, hi, host.plan.region, host.plan.particles,
                              host.plan.rf) == FAR_ELEMENT
    end
    Q = length(host.t.gauss)
    points = nfar * Q^3
    ntransition = 0
    w = host.S.window
    for b in 1:nblocks(host.p.fs), J in CartesianIndices((E, E, E))
        lo, hi = element_box(host.plan.origins[b], host.plan.spacings[b], Tuple(J) .- 1,
                             Val(P1), Val(3))
        element_class(lo, hi, host.plan.region, host.plan.particles,
                      host.plan.rf) == FAR_ELEMENT || continue
        for K in CartesianIndices((Q, Q, Q))
            x = SVector(ntuple(d -> lo[d] + (hi[d] - lo[d]) *
                                       (1 + host.t.gauss[K[d]]) / 2, 3))
            r² = sum(abs2, x - w.centre)
            ntransition += w.R₁^2 < r² < w.R₂^2
        end
    end
    rows = Pair{String,Any}["nodes" => length(host.du) ÷ 2, "far_points" => points,
                            "far_transition_points" => ntransition]
    far!(host)
    reference = copy(host.du)
    push!(rows, "far_host" => best(() -> far!(host), CPU(), reps))
    try
        far!(dev)
        difference = norm(Array(dev.du) .- reference) / norm(reference)
        tdev = best(() -> far!(dev), backend, reps)
        u = statevector(dev.p.fs)
        fill!(u, 1)
        du = similar(u)
        plan, n = dev.plan, length(dev.plan.active)
        Sd = device_source(dev.S, backend)
        source!() = TreeAMR.launch_by_owner!(
            source_points_kernel!, backend, dev.scratch, Sd, plan.dev_origins,
            plan.dev_spacings, plan.active, dev.t, plan.region, plan.particles, plan.rf,
            Val(3), Val(P1); ndrange=(E * Q, E * Q, E * Q, n))
        contract!() = TreeAMR.launch_by_owner!(
            contract_kernel!, backend, dev.dua, dev.scratch, plan.dev_origins,
            plan.dev_spacings, plan.active, dev.t, plan.region, plan.particles, plan.rf,
            -1.0, 2, Val(3), Val(P1); ndrange=(E, E, E, n))
        append!(rows, ["far_device" => tdev, "far_ns_per_point" => 1e9 * tdev / points,
                       "far_source_device" => best(source!, backend, reps),
                       "far_contract_device" => best(contract!, backend, reps),
                       "rhs_device" => best(() -> wave_rhs!(du, u, dev.p, 0.0), backend,
                                            reps),
                       "far_relative_difference" => difference])
    catch err
        msg = replace(sprint(showerror, err), r"\s+" => " ")
        push!(rows, "far_error" => first(msg, 400))
    end
    return rows
end

function main()
    hdiv = parse(Int, option("hdiv", "8"))
    order = parse(Int, option("order", "3"))
    reps = parse(Int, option("reps", "3"))
    offgrid = "--offgrid" in ARGS
    bname = option("backend", "cpu")
    println(join(("threads", "backend", "hdiv", "order", "quantity", "value"), '\t'))
    withbackend(bname, "f64") do backend, _
        for (name, value) in measure(backend, hdiv, order, reps, offgrid)
            println(join((Threads.nthreads(), bname, hdiv, order, name, value), '\t'))
            flush(stdout)
        end
    end
    return nothing
end

main()

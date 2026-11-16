# One problem on several devices over MPI, against the same problem on
# one device ("Running distributed" in `CODE.md`, "Not done"): each rank
# takes its own device (`rank` modulo the devices it sees), runs the
# cases distributed, and rank 0 then runs them again alone. Prints, per
# case, whether the states are equal bit for bit, their largest
# difference relative to the state's largest value, the step, and for
# the inspiral the forces.
#
#     mpiexec -n 2 julia -t 4,0 --project=<env> bin/distributed_gpu.jl \
#         --backend=cuda [--type=f64]
#
# The environment `Pkg.develop`s the checkout and adds MPI and the device
# package (`bin/symmetry/distributed_gpu.sh` builds it). The inspiral needs
# `Float64`, so it runs on CUDA only; Metal runs the field in `Float32`.

using MPI
include("backend.jl")
using TreeAMR
using IMEXRungeKutta: solve!
using StaticArrays

const TW = TreeWaveOLDG

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, '=')[2]
    end
    return default
end

# The state over every rank, in curve order, on the host.
everything(comm, u) = TreeAMR.allgatherv(comm, Array(u))

function field_case(case, ::Type{T}, backend, comm) where {T}
    if case == "box"
        forest = wave_forest(T, Val(3), 3, 2; roots=4, comm=comm)
        p = WaveProblem(forest, Element(T, 3); backend)
        u0 = initial_state(p, wave_exact(3, one(T), 1, zero(T)))
    else
        forest, geo = sevenpatch_forest(T, 3, 2; roots=(4, 2), Lc=1, R1=3, comm=comm)
        p = WaveProblem(forest, Element(T, 3); geometry=geo,
                        boundary=RadiativeBoundary(), backend)
        u0 = initial_state(p, WithPatch(TW.PulseState(geo, OutgoingPulse(T(1 // 2)),
                                                      zero(T))))
    end
    dt = oldg_timestep(p; cfl=1 // 2)
    integ = solve!(wave_integrator(p, u0, zero(T), 4dt; dt=dt))
    return (; u=everything(forest.comm, integ.u), dt, nleaves=nleaves(forest),
            F=nothing)
end

# A particle in uniform motion on three nested cubes, as the MPI test's.
function inspiral_case(backend, comm)
    forest = wave_forest(Float64, Val(3), 3, 2; roots=4, L=16.0,
                         refined=nested_cubes(Val(3), 2.0, 3), comm=comm)
    wave = WaveProblem(forest, Element(Float64, 3); backend)
    c = SVector(8.0, 8.0, 8.0)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(c, 1.0, 1.75; continuity=4)
    motion = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(0.2, 0.0, 0.0),))
    p = InspiralProblem(wave, (m,), w, motion)
    u0 = inspiral_initial_state(p)
    dt = oldg_timestep(wave; cfl=1 // 2)
    integ = solve!(inspiral_integrator(p, u0, 0.0, 4dt; dt=dt))
    F = particle_forces(p, integ.u, integ.t)
    field = everything(forest.comm, field_state(integ.u, p))
    return (; u=vcat(field, Array(integ.u)[(TW.fieldlength(p) + 1):end]), dt,
            nleaves=nleaves(forest), F, remote=TW.rank_reduce(+, forest,
                                                               length(p.shared.remote)))
end

function main()
    MPI.Init()
    world = MPI.COMM_WORLD
    rank, nranks = MPI.Comm_rank(world), MPI.Comm_size(world)
    name = option("backend", "cuda")
    T = resolvetype(option("type", name == "metal" ? "f32" : "f64"))
    backend = resolvebackend(name)
    if name == "cuda"
        n = Core.eval(Main, :(length(CUDA.devices())))
        Core.eval(Main, :(CUDA.device!($(rank % n))))
    end
    device = name == "cuda" ? Core.eval(Main, :(string(CUDA.device()))) : name
    for r in 0:(nranks - 1)
        r == rank && println("# rank $rank of $nranks on $(gethostname()), $device")
        flush(stdout)
        MPI.Barrier(world)
    end
    cases = ["box", "ball"]
    T == Float64 && push!(cases, "inspiral")
    rank == 0 && println("case\tranks\tleaves\tidentical\trel_diff\tdt_equal\tforce_diff")
    for case in cases
        run(comm) = case == "inspiral" ? inspiral_case(backend, comm) :
                    field_case(case, T, backend, comm)
        dist = Base.invokelatest(run, world)
        if rank == 0
            alone = Base.invokelatest(run, nothing)
            scale = maximum(abs, alone.u)
            rel = maximum(abs.(dist.u .- alone.u)) / scale
            fd = dist.F === nothing ? "-" :
                 string(maximum(k -> maximum(abs.(dist.F[k] .- alone.F[k])),
                                eachindex(alone.F)))
            println(case, '\t', nranks, '\t', dist.nleaves, '\t', dist.u == alone.u, '\t',
                    rel, '\t', dist.dt == alone.dt, '\t', fd)
            case == "inspiral" && println("# inspiral: $(dist.remote) remote pairs " *
                                          "of the averaging, force $(alone.F)")
            flush(stdout)
        end
        MPI.Barrier(world)
    end
    return nothing
end

main()

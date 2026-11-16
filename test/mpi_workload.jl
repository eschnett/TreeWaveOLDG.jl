# The workload behind the test that a run distributed over MPI is the
# serial run (added 2026-11-16, on TreeAMR's M7).
#
# Run under MPI, three ranks, the first two on several threads and the
# third on one:
#
#     mpiexec -n 2 julia --threads=3 --project=<env> test/mpi_workload.jl OUTDIR : \
#             -n 1 julia --threads=1 --project=<env> test/mpi_workload.jl OUTDIR
#
# — it runs every case below three times in one launch, so that one
# compilation serves three rank counts: over all three ranks; then over a
# communicator of ranks 0 and 1, while rank 2 runs the same cases over a
# communicator of itself alone, and then once more with no communicator at
# all. Each group's first rank writes the group's lines to `OUTDIR/n<k>.txt`
# (`n0.txt` being the serial run, on one thread), and `mpi_tests.jl`
# compares them. TreeHydro's `test/mpi_workload.jl` is the pattern.
#
# Three kinds of line. A line that starts with neither `~` nor `#` is a
# claim of bit-identity: the state vector in curve order, the leaves, the
# step, every maximum, the particles' forces and a reconstructed value,
# none of which is a floating-point sum over blocks. A line that starts
# with `~` holds sums (energies, L2 norms), which a partition
# reassociates, and is compared to roundoff. A line that starts with `#`
# depends on the rank count and is not compared; `mpi_tests.jl` asserts
# what it must say.
#
# The cases are small, so that the launch is compilation-bound, and each
# is one that a distributed forest could get wrong where a serial one
# cannot: a rank with no blocks (the line of two leaves at three ranks);
# a refined reflecting box; the seven-patch ball's radiative sphere, whose
# time step and boundary energy are host reductions over blocks; the
# thirteen-patch ball's factored truncation and Dirichlet data; the
# adaptive blast, whose regrids repartition the blocks; and particles on
# nested boxes and in the ball's cube, whose source is averaged across
# faces that ranks share and whose forces read blocks of other ranks.

using MPI
using TreeAMR
using TreeWaveOLDG
using IMEXRungeKutta: solve!
using StaticArrays

const TW = TreeWaveOLDG

# Every element, in curve order: over a distributed forest the ranks'
# state vectors concatenated in rank order are the serial state vector.
# A fold of `hash` over every element, not `hash` of the vector, which
# samples a long one.
function mpi_digest(comm, u::AbstractVector)
    full = TreeAMR.allgatherv(comm, Array(u))
    return string(foldl((h, x) -> hash(x, h), full; init=UInt(0x5eed)); base=16,
                  pad=16)
end

mpi_leaves(forest) =
    string(foldl((h, k) -> hash(k, h), forest.leaves; init=UInt(0x1eaf)); base=16,
           pad=16)

# The integrator's `n` steps of `dt` from 0, as the drivers take them.
function mpi_steps(p, u0, n; cfl=1 // 2)
    dt = oldg_timestep(p; cfl=cfl)
    integ = wave_integrator(p, u0, 0.0, n * dt; dt=dt)
    solve!(integ)
    return integ.u, dt
end

function mpi_field(lines, tag, p, u0, n)
    comm = p.fs.forest.comm
    u, dt = mpi_steps(p, u0, n)
    push!(lines, string(tag, " leaves=", mpi_leaves(p.fs.forest), " n=",
                        nleaves(p.fs.forest), " dt=", repr(dt), " u0=",
                        mpi_digest(comm, u0), " u=", mpi_digest(comm, u), " max=",
                        repr(wave_norm(p, u; p=Inf))))
    push!(lines, string("~", tag, " energy0=", repr(wave_energy(p, u0; t=0.0)),
                        " energy=", repr(wave_energy(p, u; t=n * dt)), " l2=",
                        repr(wave_norm(p, u))))
    return nothing
end

# The particles' state is the same on every rank; the field's is the
# rank's blocks. The digest takes the field over the ranks and the
# particles once.
function inspiral_digest(p, u)
    comm = p.wave.fs.forest.comm
    return mpi_digest(comm, Array(field_state(u, p))) * "/" *
           string(hash(Array(u)[(TW.fieldlength(p) + 1):end]); base=16, pad=16)
end

function mpi_inspiral(lines, tag, p, n)
    comm = p.wave.fs.forest.comm
    u0 = inspiral_initial_state(p)
    dt = oldg_timestep(p.wave; cfl=1 // 2)
    integ = solve!(inspiral_integrator(p, u0, 0.0, n * dt; dt=dt))
    u = integ.u
    F0 = particle_forces(p, u0, 0.0)
    F = particle_forces(p, u, integ.t)
    z = p.motion.path(integ.t)[1].z
    x = z + SVector(0.3, -0.2, 0.25)
    g = reconstruct(p.wave, p.tables, field_state(u, p), x)
    ∇g = reconstruct_gradient(p.wave, p.tables, field_state(u, p), x)
    push!(lines, string(tag, " leaves=", mpi_leaves(p.wave.fs.forest), " dt=", repr(dt),
                        " u0=", inspiral_digest(p, u0), " u=", inspiral_digest(p, u),
                        " F0=", repr(F0), " F=", repr(F), " g=", repr(g), " ∇g=",
                        repr(∇g)))
    c = p.window.centre
    push!(lines, string("~", tag, " ball=", repr(ball_norm(p, u, c, 1.0)),
                        " max=", repr(ball_norm(p, u, c, 1.0; p=Inf))))
    remote = TW.rank_reduce(+, p.wave.fs.forest, length(p.shared.remote))
    push!(lines, string("# ", tag, " remote=", remote))
    return nothing
end

# One particle of `ℓ = 0.01` off the element corners, moving uniformly at
# `v = 0.2`, in the window `R₁ = 1`, `R₂ = 1.75` about `c`.
function mpi_particle(wave, c)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    w = Window(c, 1.0, 1.75; continuity=4)
    motion = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(0.2, 0.0, 0.0),))
    return InspiralProblem(wave, (m,), w, motion)
end

function mpi_cases(comm)
    lines = String[]

    # Two leaves: at three ranks one rank holds no block.
    forest = wave_forest(Float64, Val(1), 3, 2; roots=2, comm=comm)
    p = WaveProblem(forest, Element(Float64, 3))
    mpi_field(lines, "line", p, initial_state(p, wave_exact(1, 1.0, 1, 0.0)), 4)

    forest = wave_forest(Float64, Val(2), 3, 2; roots=4, reflecting=true, refined=true,
                         comm=comm)
    p = WaveProblem(forest, Element(Float64, 3))
    mpi_field(lines, "refined", p, initial_state(p, reflect_exact(2, 1.0, 1, 0.0)), 3)

    forest, geo = sevenpatch_forest(Float64, 3, 2; roots=(2, 1), Lc=1, R1=3, comm=comm)
    p = WaveProblem(forest, Element(Float64, 3); geometry=geo,
                    boundary=RadiativeBoundary())
    pulse = WithPatch(TW.PulseState(geo, OutgoingPulse(0.5), 0.0))
    mpi_field(lines, "ball7", p, initial_state(p, pulse), 3)

    forest, geo = thirteenpatch_forest(Float64, 3, 2; roots=(2, 1, 1), Lc=1, Rm=2, R1=3,
                                       comm=comm)
    wave = PlaneWave((1.1, -0.7, 0.5), 0.3)
    p = WaveProblem(forest, Element(Float64, 3); geometry=geo,
                    boundary=wave)
    mpi_field(lines, "ball13", p, initial_state(p, sevenpatch_exact(geo, wave, 0.0)), 2)

    r = adaptive_blast(; P=3, E=2, roots=4, maxlevel_cap=2, refine_tol=1e-2,
                       coarsen_tol=1e-5, chunk=0.05, t_end=0.1, comm=comm)
    push!(lines, string("adaptive leaves=", mpi_leaves(r.forest), " n=",
                        nleaves(r.forest), " nsteps=", r.nsteps, " regrids=",
                        r.nregrids, " passes=", r.passes, " err=", repr(r.err)))
    push!(lines, string("~adaptive l2=", repr(r.l2), " energy=", repr(r.energy)))

    # Three nested cubes about the centre of a box of side 16, the finest
    # of half-width 2 at `hₑ = 1/2`.
    forest = wave_forest(Float64, Val(3), 3, 2; roots=4, L=16.0,
                         refined=nested_cubes(Val(3), 2.0, 3), comm=comm)
    p = mpi_particle(WaveProblem(forest, Element(Float64, 3)), SVector(8.0, 8.0, 8.0))
    mpi_inspiral(lines, "nested", p, 2)

    # The same cubes in the seven-patch ball's cube, with shells to a
    # radiative sphere.
    forest, geo = sevenpatch_forest(Float64, 3, 2; roots=(4, 1), Lc=4.0, R1=12.0,
                                    refined=nested_cubes(Val(3), 1.0, 2), comm=comm)
    wave = WaveProblem(forest, Element(Float64, 3); geometry=geo,
                       boundary=RadiativeBoundary())
    mpi_inspiral(lines, "ballparticle", mpi_particle(wave, SVector(0.0, 0.0, 0.0)), 2)
    return lines
end

function write_lines(path, lines)
    open(path, "w") do io
        foreach(l -> println(io, l), lines)
    end
    return nothing
end

function mpi_main(out)
    MPI.Init()
    world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(world)
    MPI.Comm_size(world) == 3 || error("the workload runs at three ranks")

    lines = mpi_cases(world)
    rank == 0 && write_lines(joinpath(out, "n3.txt"), lines)

    sub = MPI.Comm_split(world, rank < 2 ? 0 : 1, rank)
    n = MPI.Comm_size(sub)
    lines = mpi_cases(sub)
    MPI.Comm_rank(sub) == 0 && write_lines(joinpath(out, "n$n.txt"), lines)
    rank == 2 && write_lines(joinpath(out, "n0.txt"), mpi_cases(nothing))
    MPI.Barrier(world)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        mpi_main(only(ARGS))
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
end

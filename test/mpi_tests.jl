# Rank-count independence (added 2026-11-16, on TreeAMR's M7).
#
# The claim extends the thread one: a run distributed over MPI is the
# serial run, its state, its step, every maximum, the particles' forces
# and a reconstructed value bit for bit at any rank count and thread
# count, and its sums (energies, L2 norms) agree to roundoff, exactly at
# one rank. TreeAMR makes the exchange, the regrid and its own reductions
# so; what can break it here is this package's, and each failure mode is
# one a serial suite cannot see:
#
#   * a number combined on the host from per-block values, the time
#     step's speeds or the radiative sphere's boundary energy, that is a
#     rank's own: the ranks then take different steps and hang in the
#     next exchange;
#   * a global leaf index from `locate_point` read as a local block: the
#     particles' supports, their neighbours, the faces of the source's
#     averaging and a reconstructed point then read another block, or
#     past the array;
#   * a face the source's averaging shares between two ranks, left
#     unaveraged on both;
#   * a refusal found in one rank's blocks alone, which leaves the others
#     waiting;
#   * a rank with no blocks, on which a reduction or a launch reads a
#     block that is not there;
#   * a sphere's point read in another rank's block, or not gathered;
#   * a dynamic particle's solve whose course differs between ranks: a
#     rank that takes another Newton step calls the pairing's collectives
#     a different number of times, and the run hangs.

isdefined(@__MODULE__, :take_mpi_job!) || include("mpi_jobs.jl")

# The numbers in a line, and the line with each of them replaced by `#`,
# so that two lines of sums are the same line exactly when their skeletons
# are equal and their numbers agree.
const MPI_NUMBER = r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+|NaN|Inf)"

mpi_skeleton(line) = replace(line, MPI_NUMBER => "#")
mpi_numbers(line) = [parse(Float64, m.match) for m in eachmatch(MPI_NUMBER, line)]

# Roundoff: the sums are energies and norms of order one to a hundred, and
# a partition reassociates each once per rank boundary.
mpi_agree(a, b) = isapprox(a, b; rtol=1e-12, atol=1e-14)

@testset "A distributed run is the serial run, at every rank and thread count" begin
    files = finish_mpi_workload(take_mpi_job!())
    serial = files[0]
    # Every case reached its end, and the adaptive run regridded.
    @test length(serial) == 19
    @test !occursin("regrids=0", only(filter(startswith("adaptive "), serial)))
    # At three ranks the particles' source is averaged across faces that
    # two ranks share, which is what the case is for.
    for tag in ("nested", "ballparticle")
        line(n) = only(filter(startswith("# $tag remote="), files[n]))
        @test line(0) == "# $tag remote=0"
        @test parse(Int, split(line(3), '=')[end]) > 0
    end

    for n in 1:3
        lines = files[n]
        @test length(lines) == length(serial)
        for (a, b) in zip(serial, lines)
            startswith(a, "#") && (@test startswith(b, "#"); continue)
            if startswith(a, "~")
                ok = mpi_skeleton(a) == mpi_skeleton(b) &&
                     all(map(mpi_agree, mpi_numbers(a), mpi_numbers(b)))
                # At one rank nothing is reassociated.
                n == 1 && (ok &= a == b)
            else
                ok = a == b
            end
            @test ok
            ok || @info "a line differs at $n rank(s)" serial = a distributed = b
        end
    end
end

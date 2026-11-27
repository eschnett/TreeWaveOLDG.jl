# Some of the suite's files, named on the command line, in the test
# environment, for checking one part on a cluster without the whole suite
# (`jobs/tests_by_part.jl`):
#
#     julia --project=test test/some_tests.jl predictor_tests.jl dynamic_tests.jl
#
# A file that needs `reference.jl` or the MPI job of `runtests.jl` is not
# for this; the suite runs those.

using Test
using TreeAMR
using TreeWaveOLDG

isempty(ARGS) && error("name the test files to run, such as dynamic_tests.jl")

@info "Running $(join(ARGS, ", ")) on $(Threads.nthreads()) thread(s)"

const START = time()

@testset "TreeWaveOLDG.jl: $(join(ARGS, ", "))" begin
    for name in ARGS
        @info "$name, at $(round(Int, time() - START)) s"
        include(joinpath(@__DIR__, name))
    end
end

@info "done after $(round(Int, time() - START)) s"

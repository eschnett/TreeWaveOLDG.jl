using Test
using TreeAMR
using TreeWaveOLDG

@info "Running the tests on $(Threads.nthreads()) thread(s)"

include("reference.jl")

# Test prints a testset's results only when the outermost one ends, so
# without these lines a run is silent from its first file to its summary,
# which at one thread on CI takes most of an hour; with them a slow or
# stuck run shows which file it is in. Each file's name as it starts, with
# the time since the suite began.
const SUITE_START = time()
function testfile(name)
    @info "$name, at $(round(Int, time() - SUITE_START)) s"
    return include(name)
end

@testset "TreeWaveOLDG.jl" begin
    testfile("element_tests.jl")
    testfile("layout_tests.jl")
    testfile("evolution_tests.jl")
    testfile("sinewave_tests.jl")
    testfile("reflect_tests.jl")
    testfile("refined_tests.jl")
    testfile("annulus_tests.jl")
    testfile("sevenpatch_tests.jl")
    testfile("adaptive_tests.jl")
    testfile("blast_tests.jl")
    testfile("particle_tests.jl")
    testfile("transfer_tests.jl")
    testfile("inspiral_tests.jl")
    testfile("nested_tests.jl")
    testfile("prescribed_tests.jl")
end

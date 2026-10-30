using Test
using TreeAMR
using TreeWaveOLDG

@info "Running the tests on $(Threads.nthreads()) thread(s)"

include("reference.jl")

@testset "TreeWaveOLDG.jl" begin
    include("element_tests.jl")
    include("layout_tests.jl")
    include("evolution_tests.jl")
    include("sinewave_tests.jl")
    include("reflect_tests.jl")
    include("blast_tests.jl")
    include("particle_tests.jl")
    include("transfer_tests.jl")
    include("inspiral_tests.jl")
end

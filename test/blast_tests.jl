# The blast wave: the one case whose solution is not aligned with the
# elements, measured against an exact solution that is a quadrature
# rather than a formula. A wrong reference would make every error
# measured against it meaningless, so the reference is checked first.

@testset "The exact blast solution at t = 0 is the initial data" begin
    # Guards the Hankel table, its periodic image sum and its radial
    # interpolation: at t = 0 the quadrature must give back the Gaussian
    # it transformed, at every node, to its own accuracy.
    forest = wave_forest(Float64, Val(2), 3, 2; roots=4)
    el = Element(Float64, 3)
    L, x₀, σ = 1.0, (0.5, 0.5), 0.08
    ref = blast_reference(L, x₀, σ; rmax=6σ + 0.4)
    a = FieldSet(forest, 2; G=1)
    b = FieldSet(forest, 2; G=1)
    fill_at_nodes!(blast_initial(L, x₀, σ), a, el)
    fill_at_nodes!(blast_exact(Float64, ref, 0.0), b, el)
    ua, ub = statevector(a), statevector(b)
    gather!(ua, a)
    gather!(ub, b)
    @test maximum(abs, ua .- ub) < 1e-5
end

@testset "The blast wave converges and keeps its energy" begin
    # The ring crosses element and block faces at every angle, so this
    # is where a face term that only works for aligned data would show.
    # From 8 to 16 blocks per dimension at P = 3 the error falls by more
    # than 2³ (measured: 28×), and the energy lost is RK4's, small and
    # shrinking.
    coarse = uniform_blast(; P=3, E=2, roots=8)
    fine = uniform_blast(; P=3, E=2, roots=16)
    @test coarse.l2 / fine.l2 > 2^3
    @test fine.err < 0.05
    for r in (coarse, fine)
        @test 0 <= (r.energy0 - r.energy) / r.energy0 < 1e-6
    end
end

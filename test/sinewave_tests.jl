# Convergence: the claim the prototype exists to test. Each rate is
# measured over three resolutions, `roots = 2, 4, 8` blocks per dimension
# of `E = 2` elements each, on the standing sine mode at the default
# `cfl = 1/2`. The numbers asserted are the ones in "Measured results" in
# `CODE.md`; if one moves, re-measure and update both rather than widen a
# tolerance.

function sine_rates(D, P; truncate=true, T=Float64, roots=(2, 4, 8))
    rs = [wave_errors(T, Val(D); P=P, E=2, roots=r, truncate=truncate) for r in roots]
    hs = [r.h for r in rs]
    return (l2=convergence_rate(hs, [r.l2 for r in rs]),
            linf=convergence_rate(hs, [r.linf for r in rs]),
            errs=[r.l2 for r in rs], runs=rs)
end

@testset "With truncation, the error converges at order P" begin
    # The paper's pointwise bound (eq. 66) is O(hₑ^P), and with `v` held
    # at degree P - 1 that is what comes out, cleanly, for `u` and `v`
    # alike. A face term with the wrong weight, or a truncation that left
    # the top mode in, would show here as a rate off by one.
    for D in (1, 2), P in 2:4
        r = sine_rates(D, P)
        @test P - 0.25 < r.l2 < P + 0.2
        @test P - 0.8 < r.linf < P + 0.4
    end
end

@testset "Without truncation, u is more accurate but the rate is irregular" begin
    # The plain scheme keeps `v` at degree P. In 1D `u` then converges
    # faster than with truncation and its error is smaller, while `v`
    # converges at less than P + 1 — which is why the combined rate is
    # not a clean number, and why the paper truncates.
    for P in 3:4
        r = sine_rates(1, P; truncate=false)
        t = sine_rates(1, P)
        @test r.l2 > P - 1
        @test r.errs[end] < t.errs[end]
    end
end

@testset "At P = 1 the scheme is first order, as the paper's bound says" begin
    # With truncation `v` is piecewise constant; this is the bound with
    # no room to spare, and a reminder that `P = 1` is not the
    # second-order finite-difference scheme it superficially resembles.
    r = sine_rates(1, 1)
    @test 0.8 < r.l2 < 1.1
end

@testset "Float32 runs and agrees with Float64 at coarse resolution" begin
    # Guards the precision plumbing: the element is rounded from
    # BigFloat, the step count and time are `T`, and the kernels promote
    # nothing. At a coarse mesh the discretization error dominates
    # Float32's roundoff, so the two agree closely.
    r32 = wave_errors(Float32, Val(2); P=3, E=2, roots=2)
    r64 = wave_errors(Float64, Val(2); P=3, E=2, roots=2)
    @test r32.l2 isa Float32
    @test r32.nsteps == r64.nsteps
    @test isapprox(r32.l2, r64.l2; rtol=1e-3)
end

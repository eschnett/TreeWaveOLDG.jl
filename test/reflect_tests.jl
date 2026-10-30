# Reflecting box (Neumann, ∂ₙu = 0). The wall is a TreeAMR reflection
# symmetry: every face of the forest is `reflecting`, and `WaveProblem`
# derives the per-variable parities the wall imposes (`u`, `v` even;
# `∂_d u` odd in `d`) for the field sets' ghosts. The exact state is the
# standing Neumann mode `u = cos(ωt) ∏ cos(mπx_d/L)` of `reflect_exact`,
# which vanishes in no normal derivative and so is the reflection of the
# periodic box's sine mode.

using IMEXRungeKutta: solve!

# Convergence of the reflected standing mode, over the same root counts as
# `sine_rates`.
function reflect_rates(D, P; truncate=true, T=Float64, roots=(2, 4, 8))
    rs = [reflect_errors(T, Val(D); P=P, E=2, roots=r, truncate=truncate) for r in roots]
    hs = [r.h for r in rs]
    return (l2=convergence_rate(hs, [r.l2 for r in rs]),
            linf=convergence_rate(hs, [r.linf for r in rs]),
            errs=[r.l2 for r in rs], runs=rs)
end

@testset "On a reflecting box the Neumann mode converges at order P" begin
    # The wall enters only through the parities the field sets carry, so the
    # same kernels as on the periodic box must reproduce the reflected mode to
    # the paper's bound O(hₑ^P). A wrong parity (even gradient, say) leaves
    # an O(1) flux on the wall and the rate falls apart.
    for D in (1, 2), P in 2:3
        r = reflect_rates(D, P)
        @test P - 0.35 < r.l2 < P + 0.4
        @test P - 1.0 < r.linf < P + 0.6
    end
end

@testset "Reflection is an exact discrete symmetry: energy is conserved" begin
    # On the reflecting box the discrete energy — with the same quadrature
    # weights and the gradient field that now carries the wall's parity — is
    # an exact invariant of the semi-discrete system, just as on the periodic
    # box. The change over a run is RK4's own loss, never a gain.
    for (D, P) in ((1, 2), (2, 2))
        p = WaveProblem(wave_forest(Float64, Val(D), P, 2; roots=2, reflecting=true),
                        Element(Float64, P))
        u0 = initial_state(p, reflect_exact(D, 1.0, 1, 0.0))
        dt = oldg_timestep(p; cfl=1 // 2)
        integ = solve!(wave_integrator(p, u0, 0.0, 50dt; dt=dt))
        E0, E1 = wave_energy(p, u0), wave_energy(p, integ.u)
        @test E1 <= E0
        @test isapprox(E1, E0; rtol=1e-6)
    end
end

@testset "The wall really is Neumann: the forest declares reflecting faces" begin
    # `wave_forest(…; reflecting=true)` asks TreeAMR for a reflection on
    # every face, and `WaveProblem` accepts it (periodic staying the default),
    # deriving the solid wall's parities for the field sets' ghosts.
    periodic = wave_forest(Float64, Val(2), 2, 2; roots=2, reflecting=false)
    @test !TreeAMR.hasreflecting(periodic)
    @test all(periodic.periodic)
    forest = wave_forest(Float64, Val(2), 2, 2; roots=2, reflecting=true)
    @test TreeAMR.hasreflecting(forest)
    @test !any(forest.periodic)
    p = WaveProblem(forest, Element(Float64, 2))
    # u and v are even scalars; the gradient's d-th component is odd in d.
    @test all(all(q === EvenParity for q in qs) for qs in p.fs.parity)
    @test all(p.ufs.parity[1][d] === EvenParity for d in 1:2)
    @test all(p.grads[j].parity[1][j] === OddParity for j in 1:2)
    @test all(p.grads[j].parity[1][mod1(j + 1, 2)] === EvenParity for j in 1:2)
end

@testset "Float32 reflecting runs and agrees at coarse resolution" begin
    r32 = reflect_errors(Float32, Val(1); P=3, E=2, roots=2)
    r64 = reflect_errors(Float64, Val(1); P=3, E=2, roots=2)
    @test r32.l2 isa Float32
    @test isapprox(r32.l2, r64.l2; rtol=1e-2)
end

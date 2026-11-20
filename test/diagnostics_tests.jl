# The binary's diagnostics (INSPIRAL-PLAN.md, M6b): its orbit, energy and
# centre of mass from the particles' state, the leading-order inspiral,
# and the eccentricity fit, each held to an orbit known in closed form.

using LinearAlgebra: norm
using StaticArrays: SVector

@testset "On the binary's circle the diagnostics are the circle's" begin
    # Guards `orbit_diagnostics`: on `newtonian_binary`'s circle the
    # separation and frequency are the circle's, the separation does not
    # change, the centre of mass is the circle's centre and the momentum
    # vanishes, and the phase advances by `Ωt`; a frequency taken from one
    # particle's speed alone, a relative velocity of the wrong sign or a
    # centre without `γ` breaks one of them.
    q = sqrt(0.005)
    ms = (ParticleModel(; Q=q, ℓ=1e-3, M=1.0), ParticleModel(; Q=-q, ℓ=1e-3, M=1.0))
    c = SVector(1.0, 2.0, 3.0)
    for v in (0.05, 0.2)
        f, motion = newtonian_binary(ms, 1.0, v; centre=c, phase=0.3)
        Ω = 2v
        o(t) = (trs = motion.path(t);
                orbit_diagnostics(ms, f, map(tr -> tr.z, trs), map(tr -> tr.v, trs)))
        d0 = o(0.0)
        @test d0.d ≈ 1 rtol = 1e-15
        @test abs(d0.ddot) <= 1e-15
        @test d0.omega ≈ Ω rtol = 1e-14
        @test norm(d0.centre - c) <= 1e-15
        @test norm(d0.momentum) <= 1e-16
        @test mod(o(1.0).phase - d0.phase, 2π) ≈ Ω rtol = 1e-13
        γ = 1 / sqrt(1 - v^2)
        @test d0.energy ≈ 2(γ - 1) - binding_constant(ms, f) rtol = 1e-14
        # In the slow limit, the virial theorem: `E = -K/2d` to `O(v²)`.
        @test abs(d0.energy / (-binding_constant(ms, f) / 2) - 1) < 2v^2
    end
end

@testset "The leading-order inspiral is the plan's first run's" begin
    # Guards `dipole_ddot` and `binding_constant` against INSPIRAL.md's
    # "Parameters for a first run": opposite charges `q² = 0.005`, `M = 1`,
    # `d = 1`, `K = 0.08` give `ḋ = -8.5·10⁻⁵`; like charges with no
    # difference of charge over mass do not radiate a dipole.
    q = sqrt(0.005)
    ms = (ParticleModel(; Q=q, ℓ=1e-3, M=1.0), ParticleModel(; Q=-q, ℓ=1e-3, M=1.0))
    f = NewtonianForce(0.08 + q^2 / 4π)
    @test binding_constant(ms, f) ≈ 0.08 rtol = 1e-15
    @test dipole_ddot(ms, f, 1.0) ≈ -(2q)^2 * 0.08 / 6π rtol = 1e-15
    @test round(dipole_ddot(ms, f, 1.0); sigdigits=2) == -8.5e-5
    @test dipole_ddot((ms[1], ms[1]), f, 1.0) == 0
    @test binding_constant((ms[1], ms[1]), NoInteraction()) ≈ q^2 / 4π rtol = 1e-15
end

@testset "The eccentricity fit finds the oscillation about a secular trend" begin
    # Guards `eccentricity_fit`: a separation that falls and curves while
    # it oscillates at the orbital frequency must give the oscillation's
    # amplitude over the trend's start, and the trend; a fit without the
    # trend's curvature or at the wrong phase misreads the amplitude.
    ω = 0.4
    t = collect(range(0, 3 * 2π / ω; length=600))
    d = @. 1 - 1e-4 * t + 3e-6 * t^2 + 2e-3 * cos(ω * t + 0.7)
    fit = eccentricity_fit(t, d, ω)
    @test fit.e ≈ 2e-3 rtol = 1e-10
    @test all(abs.(fit.trend .- (1.0, -1e-4, 3e-6)) .< 1e-11)
    @test_throws ArgumentError eccentricity_fit(t[1:4], d[1:4], ω)
end

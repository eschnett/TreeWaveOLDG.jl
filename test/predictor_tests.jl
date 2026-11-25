# The acceleration predictor (INSPIRAL-PLAN.md, M6a): the instantaneous
# acceleration under external forces and the other particles' boosted
# static fields, and its derivatives along the motion it predicts, which
# give a dynamic particle its `a⃛` and `a⃜`. Each claim is held against
# a motion known otherwise: the spring's exact circle, and a two-body
# orbit integrated numerically.

using LinearAlgebra: dot, norm
using StaticArrays: SVector

@testset "The spring's predicted jets are its circle's" begin
    # Guards the predictor's derivatives along its own motion and the
    # relativistic equation of motion: a central force `k r` holds a
    # particle on a circle of `Ω² = k/(γM)`, so a wrong `γ`, a lost
    # projection `(I - vvᵀ)F` or a derivative taken along the wrong
    # polynomial leaves the circle's exact jets.
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    for v in (0.05, 0.2)
        F, motion = spring_circle(m, (1.0, 2.0, 3.0), 0.5, v; phase=0.3,
                                  normal=(0.2, 0.1, 1.0))
        tr = motion.path(0.0)[1]
        j = predicted_jets((m,), (F,), (tr.z,), (tr.v,))[1]
        for f in (:a, :ȧ, :ä, :a⃛, :a⃜)
            @test norm(getfield(j, f) - getfield(tr, f)) <= 1e-14 * norm(getfield(tr, f))
        end
    end
    @test predicted_acceleration((m,), (NoForce(),),
                                 (Trajectory(; z=SVector(1.0, 2, 3),
                                             v=SVector(0.1, 0, 0)),))[1] == zeros(3)
end

@testset "Newton's attraction is equal and opposite, and acts as a force from outside" begin
    # Guards the binary's binding force (M6b): Newton's third law and the
    # inverse square, and that the predictor adds it to the external force
    # before the projection `(I - vvᵀ)`, with the physical mass and the
    # other particle's potential as the inertia; without an interaction
    # the predictor is M6a's bit for bit.
    f = NewtonianForce(0.08)
    @test isbits(f) && isbits(NoInteraction())
    z1, z2 = SVector(0.3, -0.2, 0.1), SVector(-0.5, 0.4, 0.2)
    F12, F21 = f(1.0, z1, 1.5, z2), f(1.5, z2, 1.0, z1)
    @test F12 == -F21
    @test norm(F12) ≈ 0.08 * 1.5 / norm(z2 - z1)^2 rtol = 1e-15
    @test dot(F12, z2 - z1) > 0
    @test NoInteraction()(1.0, z1, 1.5, z2) == zeros(3)
    ms = (ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0),
          ParticleModel(; Q=-sqrt(0.005), ℓ=1e-3, M=1.5))
    ext = (SpringForce((0.0, 0.0, 0.0), 0.1), NoForce())
    trs = (Trajectory(; z=z1, v=SVector(0.0, 0.2, 0.05)),
           Trajectory(; z=z2, v=SVector(-0.1, 0.0, 0.0)))
    a0 = predicted_acceleration(ms, ext, trs)
    @test a0 == predicted_acceleration(ms, ext, trs; interaction=NoInteraction())
    a1 = predicted_acceleration(ms, ext, trs; interaction=f)
    for (k, F) in ((1, F12), (2, F21))
        v = trs[k].v
        γ = 1 / sqrt(1 - dot(v, v))
        b = 3 - k
        Φ = TreeWaveOLDG.boosted_static(ms[b], trs[k].z, trs[b].z, trs[b].v)
        @test norm(a1[k] - a0[k] -
                   (F - dot(v, F) * v) / ((ms[k].M + ms[k].Q * Φ) * γ)) <= 1e-15
    end
end

@testset "The predictor's inertia holds the other charge's potential, as the solve's does" begin
    # Guards the predictor's inertia (`CODE.md`, "The release"): the
    # equation of motion's `m = M₀ + QΦ̄ₑ` holds the other particle's
    # potential `γ∫ρΨ_b`, `q²/4πd` for a binary, and a predictor that
    # leaves it out, or takes it with a `γ` or with the wrong sign,
    # predicts a binary's acceleration off by that share of the mass, which
    # makes the released orbit eccentric and biases `ä` by about `0.07q²`.
    # In uniform motion the other particle's field is its boosted static
    # one, so `other_force`'s potential is the predictor's to the
    # support's size.
    q = sqrt(0.2)
    ms = (ParticleModel(; Q=q, ℓ=0.04, M=1.0), ParticleModel(; Q=-q, ℓ=0.04, M=1.0))
    trs = (Trajectory(; z=SVector(0.5, 0.0, 0.0), v=SVector(0.0, 0.2, 0.0)),
           Trajectory(; z=SVector(-0.5, 0.0, 0.0), v=SVector(0.0, -0.2, 0.0)))
    γ = 1 / sqrt(1 - 0.2^2)
    QΦ = q * TreeWaveOLDG.boosted_static(ms[2], trs[1].z, trs[2].z, trs[2].v)
    _, Qb = TreeWaveOLDG.other_force(ms[1], trs[1], ms[2], trs[2],
                                     TreeWaveOLDG.SupportRule(Float64))
    @test QΦ ≈ Qb rtol = 1e-12
    @test QΦ ≈ q^2 / 4π rtol = 1e-12
    # The predictor's force over its acceleration is that inertia times `γ`.
    a = predicted_acceleration(ms, (NoForce(), NoForce()), trs)[1]
    ∇Φ = TreeWaveOLDG.ForwardDiff.gradient(
        x -> TreeWaveOLDG.boosted_static(ms[2], x, trs[2].z, trs[2].v), trs[1].z)
    Π = -dot(trs[2].v, ∇Φ)
    F = -(q / γ) * (∇Φ + trs[1].v * Π)
    @test norm(a - F / ((1 + QΦ) * γ)) <= 1e-15 * norm(a)
end

@testset "The two-body predictor's jets are its integrated orbit's" begin
    # Guards the other particle's field in the predictor and its rate:
    # two opposite charges repel through their boosted static fields,
    # and attract through Newton's force when there is one, and the Taylor
    # polynomial of the predicted jets must follow the orbit integrated
    # from the predicted acceleration to its seventh order in time,
    # which a wrong gradient, a wrong rate `-v_b·∇Φ_C`, an interaction
    # left out of the derivatives or a jet missing a term breaks.
    T = BigFloat
    setprecision(T, 128) do
        ms = (ParticleModel(; Q=T(1) / 2, ℓ=T(1) / 20, M=T(1)),
              ParticleModel(; Q=-T(1) / 2, ℓ=T(1) / 20, M=T(3) / 2))
        ext = (NoForce(), NoForce())
        zs = (SVector{3,T}(1 // 2, 0, 1 // 10), SVector{3,T}(-1 // 2, 0, 0))
        vs = (SVector{3,T}(0, 3 // 10, 1 // 20), SVector{3,T}(0, -1 // 5, 0))
        @testset "$(nameof(typeof(interaction)))" for interaction in
                                                     (NoInteraction(),
                                                      NewtonianForce(T(1) / 5))
            jets = predicted_jets(ms, ext, zs, vs; interaction)
            accel(z, v) = predicted_acceleration(ms, ext,
                                                 map((z, v) -> Trajectory(; z, v), z, v);
                                                 interaction)
            # RK4 on (z, v), steps fine enough that its error is far below the
            # Taylor polynomial's.
            function orbit(t, n)
                z, v, h = zs, vs, t / n
                for _ in 1:n
                    k1z, k1v = v, accel(z, v)
                    z2 = map((z, k) -> z + h / 2 * k, z, k1z)
                    v2 = map((v, k) -> v + h / 2 * k, v, k1v)
                    k2z, k2v = v2, accel(z2, v2)
                    z3 = map((z, k) -> z + h / 2 * k, z, k2z)
                    v3 = map((v, k) -> v + h / 2 * k, v, k2v)
                    k3z, k3v = v3, accel(z3, v3)
                    z4 = map((z, k) -> z + h * k, z, k3z)
                    v4 = map((v, k) -> v + h * k, v, k3v)
                    k4z, k4v = v4, accel(z4, v4)
                    z = map((z, a, b, c, d) -> z + h / 6 * (a + 2b + 2c + d), z, k1z, k2z,
                            k3z, k4z)
                    v = map((v, a, b, c, d) -> v + h / 6 * (a + 2b + 2c + d), v, k1v, k2v,
                            k3v, k4v)
                end
                return z
            end
            errs = map((T(2) / 5, T(1) / 5)) do t
                z = orbit(t, 2000)
                maximum(k -> norm(trajectory_at(jets[k], t).z - z[k]), 1:2)
            end
            # The polynomial is exact through `t⁶`, so its error halves seven
            # times with `t`.
            @test 6.5 < log2(errs[1] / errs[2]) < 7.5
        end
    end
end

@testset "Newton's force holds the binary on its circle, with the predictor's jets" begin
    # Guards the binary's initial orbit (M6b, INSPIRAL.md's "Initial data"
    # step 1): `G` must make the predictor's acceleration, Newton's
    # attraction and the other charge's boosted static field, the circle's
    # on both particles, so that the circle is the predictor's exact
    # orbit and its jets through `a⃜` the circle's; the centre of mass and
    # the momentum must vanish. A `G` solved on one particle only, a lost
    # `γ` or a boosted field read at the wrong point leaves the circle.
    q = sqrt(0.005)
    ms = (ParticleModel(; Q=q, ℓ=1e-3, M=1.0), ParticleModel(; Q=-q, ℓ=1e-3, M=1.0))
    ext = (NoForce(), NoForce())
    for v in (0.05, 0.2)
        f, motion = newtonian_binary(ms, 1.0, v; centre=(1.0, 2.0, 3.0), phase=0.3,
                                     normal=(0.2, 0.1, 1.0))
        # The Newtonian circle's `K = G M² + Q₁Q₂/4π = 2Mv²d`, to `O(v²)`.
        @test abs(f.G - (2v^2 + q^2 / 4π)) <= 2v^2 * (2v^2 + q^2 / 4π)
        trs = motion.path(0.0)
        jets = predicted_jets(ms, ext, map(tr -> tr.z, trs), map(tr -> tr.v, trs);
                              interaction=f)
        for k in 1:2, s in (:a, :ȧ, :ä, :a⃛, :a⃜)
            @test norm(getfield(jets[k], s) - getfield(trs[k], s)) <=
                  1e-13 * norm(getfield(trs[k], s))
        end
        γ = 1 / sqrt(1 - v^2)
        @test norm(sum(tr -> γ * tr.v, trs)) <= 1e-16
        @test norm(sum(tr -> tr.z, trs) / 2 - SVector(1.0, 2.0, 3.0)) <= 1e-15
    end
    @test_throws ArgumentError newtonian_binary((ms[1], ParticleModel(; Q=-q, ℓ=1e-3,
                                                                     M=1.5)), 1.0, 0.2)
    # Like charges attract (`K = G M² + Q₁Q₂/4π`), and at a low speed more
    # than the circle needs.
    @test_throws ArgumentError newtonian_binary((ms[1], ms[1]), 1.0, 1e-3)
end

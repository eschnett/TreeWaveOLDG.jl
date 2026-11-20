# The acceleration predictor (INSPIRAL-PLAN.md, M6a): the instantaneous
# acceleration under external forces and the other particles' boosted
# static fields, and its derivatives along the motion it predicts, which
# give a dynamic particle its `a⃛` and `a⃜`. Each claim is held against
# a motion known otherwise: the spring's exact circle, and a two-body
# orbit integrated numerically.

using LinearAlgebra: norm
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

@testset "The two-body predictor's jets are its integrated orbit's" begin
    # Guards the other particle's field in the predictor and its rate:
    # two opposite charges attract through their boosted static fields,
    # and the Taylor polynomial of the predicted jets must follow the
    # orbit integrated from the predicted acceleration to its seventh
    # order in time, which a wrong gradient, a wrong rate `-v_b·∇Φ_C` or
    # a jet missing a term breaks.
    T = BigFloat
    setprecision(T, 128) do
        ms = (ParticleModel(; Q=T(1) / 2, ℓ=T(1) / 20, M=T(1)),
              ParticleModel(; Q=-T(1) / 2, ℓ=T(1) / 20, M=T(3) / 2))
        ext = (NoForce(), NoForce())
        zs = (SVector{3,T}(1 // 2, 0, 1 // 10), SVector{3,T}(-1 // 2, 0, 0))
        vs = (SVector{3,T}(0, 3 // 10, 1 // 20), SVector{3,T}(0, -1 // 5, 0))
        jets = predicted_jets(ms, ext, zs, vs)
        accel(z, v) = predicted_acceleration(ms, ext, map((z, v) -> Trajectory(; z, v),
                                                          z, v))
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

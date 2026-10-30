# The analytic particle of the inspiral: its profile and radial
# potentials, its near field, the field's time derivatives along the
# motion, the bracket of the effective source, and the window. Every
# closed form here was generated from `derivations/`, so these tests
# check the generated code against the definitions, independently of the
# Wolfram Language derivations that produced it.

using ForwardDiff
using LinearAlgebra: dot, tr
using StaticArrays

const TW = TreeWaveOLDG

# ∫_lo^hi f by Gauss–Lobatto quadrature with n + 1 points in BigFloat,
# exact for polynomials of degree 2n - 1.
function gl_integral(f, lo, hi; n=24)
    ξ, w = gll_nodes_weights(n; precision=precision(BigFloat))
    half = (hi - lo) / 2
    return half * sum(w[i] * f(lo + half * (ξ[i] + 1)) for i in eachindex(ξ))
end

# The profile per unit charge, as INSPIRAL.md defines it.
rho0_definition(s, ℓ) = s < ℓ ? 3465 / (512 * big(π) * ℓ^3) * (1 - s^2 / ℓ^2)^4 : zero(s)

# B_k(r) = ∫ ρ₀(x′) |x - x′|^k d³x′ in the coordinates (s, R), s = |x′| and
# R = |x - x′|, where the angular integral is ∫ R^{k+1} dR · 2π/(r s): a
# polynomial integrand on each side of s = r, so the quadrature is exact.
# The 1/s cancels against the s² of the volume element.
function bk_definition(k, r, ℓ)
    iszero(r) && return gl_integral(s -> 4big(π) * s^(2 + k) * rho0_definition(s, ℓ), 0, ℓ)
    f(s) = rho0_definition(s, ℓ) * s * 2big(π) / r *
           gl_integral(R -> R^(k + 1), abs(r - s), r + s)
    return r < ℓ ? gl_integral(f, zero(r), r) + gl_integral(f, r, ℓ) :
           gl_integral(f, zero(r), ℓ)
end

# n-th derivative of a scalar function, by nested dual numbers.
nth_derivative(f, x, n) =
    n == 0 ? f(x) : nth_derivative(y -> ForwardDiff.derivative(f, y), x, n - 1)

laplacian(f, x) = tr(ForwardDiff.hessian(f, x))

@testset "The closed forms are their definitions" begin
    # Guards the generated radial functions against a wrong derivation, a
    # wrong branch at σ = 1, or coefficients that lose accuracy when
    # rounded: each equals BigFloat quadrature of its definition to
    # 10⁻¹⁴ in Float64, inside and outside the particle.
    ℓ = 1e-3
    ℓbig = big(ℓ)
    radial = Dict(-1 => TW._Bm1, 1 => TW._B1, 3 => TW._B3, 5 => TW._B5)
    for σ in (0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.5, 3.0, 10.0)
        r = σ * ℓ
        rbig = big(r)
        for k in (-1, 1, 3, 5)
            @test radial[k](r^2, ℓ) ≈ bk_definition(k, rbig, ℓbig) rtol = 1e-14
        end
        # Φ₀ = -B₋₁/4π, the Poisson solution ∇²Φ₀ = ρ₀.
        @test TW._phi0(r^2, ℓ) ≈ -bk_definition(-1, rbig, ℓbig) / 4big(π) rtol = 1e-14
        # Near its edge ρ₀ ∝ (1 - σ²)⁴ is itself ill-conditioned in q:
        # rounding q at σ = 0.999 moves it by 10⁻¹³.
        σ == 0.999 && continue
        @test TW._rho0(r^2, ℓ) ≈ rho0_definition(rbig, ℓbig) rtol = 1e-14
    end
    # δm = (1/12π) ∬ ρρ′/|x - x′| and ∬ ρρ′|x - x′|, as integrals of ρ₀ B_k.
    ball(k) = gl_integral(r -> 4big(π) * r^2 * rho0_definition(r, ℓbig) *
                               bk_definition(k, r, ℓbig), 0, ℓbig; n=16)
    @test TW.self_inertia(1.0, ℓ) ≈ ball(-1) / 12big(π) rtol = 1e-14
    @test TW.rho_rho_r(1.0, ℓ) ≈ ball(1) rtol = 1e-14
    @test TW.self_force_addot(1.0, ℓ) ≈ -TW.rho_rho_r(1.0, ℓ) / 24π rtol = 1e-14
end

@testset "The particle is refused below its classical radius" begin
    # Guards the bare mass: a particle with M₀ = M - δm ≤ 0 has runaway
    # solutions, and must not be constructed.
    m = ParticleModel(; Q=0.07, ℓ=1e-3, M=1.0)
    @test m.M₀ ≈ 1 - 693 * 0.07^2 / (4199π * 1e-3)
    @test m.M₀ ≈ m.M - self_inertia(m)
    @test_throws ArgumentError ParticleModel(; Q=1.0, ℓ=1e-3, M=1.0)
    @test_throws ArgumentError ParticleModel(; Q=1.0, ℓ=0.0, M=1.0)
    traj = Trajectory(; z=SA[0, 0, 0], v=SA[0.1, 0, 0])
    @test isbits(traj)
    @test eltype(traj.z) == Float64
end

# A BigFloat particle at ℓ = 10⁻³ and a moving trajectory, for the tests
# that differentiate.
big_model() = ParticleModel{BigFloat}(big"0.07", big"1e-3", big"1.0")
function big_trajectory(; v=SA[big"0.12", big"-0.1", big"0.1"])
    return Trajectory(; z=SA[big"0.1", big"0.2", big"0.3"], v,
                      a=SA[big"0.03", big"0.1", big"-0.02"],
                      ȧ=SA[big"0.01", big"0.0", big"0.02"],
                      ä=SA[big"-0.01", big"0.02", big"0.03"],
                      a⃛=SA[big"0.004", big"-0.003", big"0.001"])
end
# Points at σ = |x - z|/ℓ inside and outside the particle, in several
# directions.
sample_points(z, ℓ) = [z + σ * ℓ * n / sqrt(dot(n, n))
                       for σ in (big"0.3", big"0.9", big"1.5", big"4.0")
                       for n in (SA[1, 0, 0], SA[0, 1, 0], SA[1, 2, -2], SA[-3, 1, 1])]

@testset "The boosted static field solves its own wave equation" begin
    # Guards the boost and the profile's field together: □_vΦ_C = ρ holds
    # exactly, so that ρ can cancel it analytically in the bracket.
    m = big_model()
    z = SA[big"0.1", big"0.2", big"0.3"]
    for v in (SA[big"0.2", 0, 0], SA[0, 0, big"-0.2"],
              big"0.2" * SA[1, 2, 2] / 3)
        for x in sample_points(z, m.ℓ)
            H = ForwardDiff.hessian(ξ -> boosted_static(m, ξ, z, v), x)
            □ = tr(H) - dot(v, H * v)
            @test □ ≈ particle_density(m, x, z, v) atol = 1e-60 * m.Q / m.ℓ^3
        end
    end
end

@testset "Each near-field order's Laplacian is its generated closed form" begin
    # Guards the generated Laplacians, which the bracket uses in place of
    # differentiating twice, against the terms they belong to: nested
    # dual numbers in BigFloat, inside and outside the particle.
    m = big_model()
    a = SA[big"0.3", big"-0.5", big"0.2"]
    ä = SA[big"-0.4", big"0.1", big"0.6"]
    for N in 1:3, y in sample_points(zero(a), m.ℓ)
        ∇² = laplacian(ξ -> TW.near_field_term(Val(N), m.Q, m.ℓ, ξ, a, ä), y)
        @test ∇² ≈ TW.near_field_laplacian(Val(N), m.Q, m.ℓ, y, a, ä) rtol = 1e-60
    end
end

@testset "Along a motion through rest, the bracket is the first order not carried" begin
    # Guards the hierarchy ∇²Φₙ = ∂ₜ²Φₙ₋₂ as the bracket sees it. With the
    # derivatives of the velocity scaled as a → εa, ȧ → ε²ȧ, ä → ε³ä, the
    # bracket with orders up to N carried is O(ε^{N+1}), and for N = 0, 1
    # its leading part is ∇²Φ⁽ᴺ⁺¹⁾. At order 3 the rest-frame terms, without
    # the lab-slice re-expansion, leave a part that does not vanish at rest:
    # (a·r)²(a·∇Φ₀) for a point, plus finite-size terms in (ℓ/r)² (CODE.md,
    # "The near field"). So carrying order 3 changes the bracket by
    # -∇²Φ⁽³⁾ only.
    ε = big"1e-20"
    scale(traj) = Trajectory(; traj.z, traj.v, a=ε * traj.a, ȧ=ε^2 * traj.ȧ,
                             ä=ε^3 * traj.ä, a⃛=ε^4 * traj.a⃛)
    traj = big_trajectory(; v=SA[big(0), big(0), big(0)])
    (; z, a, ä) = traj
    m = big_model()
    for x in sample_points(z, m.ℓ)
        bracket(order) = source_bracket(m, x, scale(traj); order)
        lap(N) = TW.near_field_laplacian(Val(N), m.Q, m.ℓ, x - z, a, ä)
        @test bracket(0) / ε ≈ lap(1) rtol = 1e-18
        @test bracket(1) / ε^2 ≈ lap(2) rtol = 1e-18
        @test (bracket(2) - bracket(3)) / ε^3 ≈ lap(3) rtol = 1e-18
    end
    # A particle of size 10⁻¹² seen from 10⁻² and farther is a point to
    # (ℓ/r)² = 10⁻²⁰.
    point = ParticleModel{BigFloat}(big"0.07", big"1e-12", big"1e10")
    for x in sample_points(z, big"0.01")[9:end]
        ∇Φ₀ = ForwardDiff.gradient(ξ -> boosted_static(point, ξ, z, zero(z)), x)
        missing3 = dot(a, x - z)^2 * dot(a, ∇Φ₀)
        @test source_bracket(point, x, scale(traj); order=3) / ε^3 ≈ missing3 rtol = 1e-15
    end
end

@testset "The point limits are the field of uniform acceleration" begin
    # Guards the near field against INSPIRAL.md's third method: at t = 0
    # the field of a point charge in hyperbolic motion (proper acceleration
    # g, so ȧ = 0 and ä = -3g²a) is -(Q/4πr) Σₖ (-gr/2)^k P_k(cos θ). Orders
    # 1–3 are its terms k = 1–3, and what is left after them falls as r³.
    ℓ = big"1e-40"                       # a point, to 10⁻⁷⁰ at these radii
    g = big"0.7"
    a = SA[0, 0, g]
    ä = -3g^2 * a
    legendre = (c -> c, c -> (3c^2 - 1) / 2, c -> (5c^3 - 3c) / 2)
    for θ in (big"0.3", big"1.2", big"2.5"), r in (big"0.01", big"0.3")
        y = r * SA[sin(θ), 0, cos(θ)]
        for k in 1:3
            exact = -(-g * r / 2)^k * legendre[k](cos(θ)) / (4big(π) * r)
            @test TW.near_field_term(Val(k), 1, ℓ, y, a, ä) ≈ exact rtol = 1e-60
        end
    end
    θ = big"0.3"
    rs = [big"0.002" * 10^(j / big(4)) for j in 0:4]
    residual = map(rs) do r
        y = r * SA[sin(θ), 0, cos(θ)]
        exact = -1 / (4big(π) * r * sqrt(1 + g * r * cos(θ) + g^2 * r^2 / 4))
        Ψ = TW._phi0(r^2, ℓ) + sum(TW.near_field_term(Val(k), 1, ℓ, y, a, ä) for k in 1:3)
        return abs(exact - Ψ)
    end
    @test convergence_rate(Float64.(rs), Float64.(residual)) ≈ 3 atol = 0.1
end

@testset "The time derivatives are those of finite differences in t" begin
    # Guards 𝒟Ψ and the bracket against a wrong chain rule through the
    # boost, α′ or the Taylor polynomial: both agree with BigFloat central
    # differences along the trajectory, and the bracket with finite
    # differences in x for ∇²Ψ, for each order carried.
    m = big_model()
    traj = big_trajectory()
    h = big"1e-20"
    e = (SA[1, 0, 0], SA[0, 1, 0], SA[0, 0, 1])
    for x in sample_points(traj.z, m.ℓ)[1:4:end], order in 0:3
        Ψ(t, ξ=x) = particle_field(m, ξ, trajectory_at(traj, t); order)
        rate = (Ψ(h) - Ψ(-h)) / 2h
        @test particle_rate(m, x, traj; order) ≈ rate rtol = 1e-30
        ∂ₜ² = (Ψ(h) - 2Ψ(0) + Ψ(-h)) / h^2
        ∇² = sum((Ψ(0, x + h * ed) - 2Ψ(0) + Ψ(0, x - h * ed)) / h^2 for ed in e)
        bracket = ∂ₜ² - ∇² + particle_density(m, x, traj.z, traj.v)
        @test source_bracket(m, x, traj; order) ≈ bracket rtol = 1e-15
    end
end

# The bracket the naive way: two second derivatives and ρ, subtracted.
function naive_bracket(m, x, traj; order)
    Ψ(t) = particle_field(m, x, trajectory_at(traj, t); order)
    ∂ₜ² = TW.second_derivative(Ψ, zero(eltype(x)))
    ∇² = laplacian(ξ -> particle_field(m, ξ, traj; order), x)
    return ∂ₜ² - ∇² + particle_density(m, x, traj.z, traj.v)
end

@testset "The bracket is assembled without cancellation" begin
    # Guards the assembly of the bracket: near the particle ∇²Ψ, (v·∇)²Ψ
    # and ρ are each of size Q/ℓ³, and subtracting them in Float64 would
    # lose digits. The cancellation-free Float64 bracket equals the naive
    # difference evaluated in BigFloat to 10⁻¹² at v = 0.2.
    m = ParticleModel(; Q=0.07, ℓ=1e-3, M=1.0)
    mbig = big_model()
    for v in (SA[0.2, 0.0, 0.0], 0.2 * SA[1.0, 2.0, 2.0] / 3)
        traj = Trajectory(; z=SA[0.1, 0.2, 0.3], v, a=SA[0.03, 0.1, -0.02],
                          ȧ=SA[0.01, 0.0, 0.02], ä=SA[-0.01, 0.02, 0.03])
        trajbig = Trajectory(; z=big.(traj.z), v=big.(traj.v), a=big.(traj.a),
                             ȧ=big.(traj.ȧ), ä=big.(traj.ä))
        for xbig in sample_points(trajbig.z, mbig.ℓ), order in (0, 3)
            x = Float64.(xbig)
            exact = naive_bracket(mbig, big.(x), trajbig; order)
            @test source_bracket(m, x, traj; order) ≈ exact rtol = 1e-12
        end
    end
end

@testset "Without acceleration the bracket vanishes" begin
    # Guards the analytic cancellation of ρ against □_vΦ_C: for uniform
    # motion the particle field is exact and the bracket is zero, to
    # roundoff of the size Q/ℓ³ of the terms it does not form.
    m = ParticleModel(; Q=0.07, ℓ=1e-3, M=1.0)
    for v in (SA[0.0, 0.0, 0.0], SA[0.2, 0.0, 0.0], 0.2 * SA[1.0, 2.0, 2.0] / 3)
        traj = Trajectory(; z=SA[0.1, 0.2, 0.3], v)
        for xbig in sample_points(big.(traj.z), big(m.ℓ)), order in 0:3
            x = Float64.(xbig)
            @test abs(source_bracket(m, x, traj; order)) <= 1e-15 * m.Q / m.ℓ^3
        end
    end
end

@testset "The window is one inside, zero outside, and C^m between" begin
    # Guards the window's step: W must be exactly 1 where the particles
    # are (so that the split is exact there), exactly 0 beyond R₂, and as
    # smooth as asked, and its closed-form gradient and Laplacian must be
    # those of W.
    for m in (2, 4, 5)
        S(ξ) = TW.smoothstep(m, ξ)
        @test S(big(0)) == 0 && S(big(1)) == 1
        for k in 1:m
            @test nth_derivative(S, big(0), k) == 0
            @test nth_derivative(S, big(1), k) == 0
        end
        @test nth_derivative(S, big(0), m + 1) != 0
        @test nth_derivative(S, big(1), m + 1) != 0
    end
    w = Window(SA[0.1, -0.2, 0.05], 1.25, 2.5; continuity=4)
    for r in (0.0, 0.5, 1.25), n in (SA[1.0, 0, 0], SA[0.6, 0, 0.8])
        @test w(w.centre + r * n) == 1
        @test window_terms(w, w.centre + r * n) == (1, zero(n), 0)
    end
    for r in (2.5, 3.0)
        @test w(w.centre + r * SA[0.6, 0, 0.8]) == 0
    end
    for r in (1.3, 1.9, 2.4), n in (SA[1.0, 0, 0], SA[0.6, 0, 0.8], SA[-2, 1, 2] / 3)
        x = w.centre + r * n
        W, ∇W, ∇²W = window_terms(w, x)
        @test W ≈ w(x)
        @test ∇W ≈ ForwardDiff.gradient(w, x) rtol = 1e-13
        @test ∇²W ≈ laplacian(w, x) rtol = 1e-12
    end
end

@testset "The source is the residual of the windowed particle field" begin
    # Guards the window terms of the source, -2∇W·∇Ψ̂ - Ψ̂∇²W, and the sum
    # over particles: S = ∂ₜ²(WΨ̂) - ∇²(WΨ̂) + Σρ, evaluated naively in
    # BigFloat, inside the window, in its transition and beyond it.
    m1 = ParticleModel(; Q=0.07, ℓ=1e-3, M=1.0)
    m2 = ParticleModel(; Q=-0.07, ℓ=1e-3, M=1.0)
    t1 = Trajectory(; z=SA[0.5, 0.0, 0.0], v=SA[0.0, 0.05, 0.0],
                    a=SA[-0.0025, 0.0, 0.0], ä=SA[0.0, 0.0, 0.001])
    t2 = Trajectory(; z=-t1.z, v=-t1.v, a=-t1.a, ä=-t1.ä)
    w = Window(SA[0.0, 0.0, 0.0], 1.25, 2.5; continuity=4)
    bigm(m) = ParticleModel{BigFloat}(m.Q, m.ℓ, m.M)
    bigt(t) = Trajectory(; z=big.(t.z), v=big.(t.v), a=big.(t.a), ȧ=big.(t.ȧ),
                         ä=big.(t.ä))
    wbig = Window(big.(w.centre), big(w.R₁), big(w.R₂); continuity=w.m)
    models, trajs = (bigm(m1), bigm(m2)), (bigt(t1), bigt(t2))
    WΨ(ξ, t) = wbig(ξ) * sum(particle_field(m, ξ, trajectory_at(tr, t))
                             for (m, tr) in zip(models, trajs))
    for x in (SA[0.5, 0.002, 0.0], SA[0.9, 0.3, -0.2], SA[1.3, 0.4, 0.2],
              SA[-1.0, 1.4, 0.5], SA[2.0, 1.0, 1.0])
        xb = big.(x)
        exact = TW.second_derivative(t -> WΨ(xb, t), big(0)) - laplacian(ξ -> WΨ(ξ, 0), xb) +
                sum(particle_density(m, xb, tr.z, tr.v) for (m, tr) in zip(models, trajs))
        S = particle_source((m1, m2), w, x, (t1, t2))
        @test S ≈ exact rtol = 1e-10 atol = 1e-14
    end
end

@testset "The source can be a kernel argument and does not allocate" begin
    # Guards ParticleSource as what the projection evaluates some 10⁷
    # times: it must be isbits to reach a kernel, and an allocation or a
    # type instability per call would cost more than the source itself.
    m1 = ParticleModel(; Q=0.07, ℓ=1e-3, M=1.0)
    m2 = ParticleModel(; Q=-0.07, ℓ=1e-3, M=1.0)
    t1 = Trajectory(; z=SA[0.5, 0.0, 0.0], v=SA[0.0, 0.2, 0.0],
                    a=SA[-0.08, 0.0, 0.0], ȧ=SA[0.0, -0.032, 0.0],
                    ä=SA[0.0128, 0.0, 0.0])
    t2 = Trajectory(; z=-t1.z, v=-t1.v, a=-t1.a, ȧ=-t1.ȧ, ä=-t1.ä)
    w = Window(SA[0.0, 0.0, 0.0], 1.25, 2.5; continuity=4)
    S = ParticleSource((m1, m2), w, (t1, t2))
    @test isbits(S)
    for x in (SA[0.5, 0.002, 0.0], SA[1.3, 0.4, 0.2], SA[2.0, 1.0, 1.0])
        @test S(x) == particle_source((m1, m2), w, x, (t1, t2))
        S(x)
        @test (@allocated S(x)) == 0
    end
end

@testset "The checked-in closed forms are what the derivations generate" begin
    # Guards src/particle_forms.jl against a hand edit or a derivation
    # changed without regenerating: where wolframscript is available, the
    # generator reproduces the file byte for byte.
    wolframscript = get(ENV, "TREEWAVEOLDG_WOLFRAMSCRIPT", "")
    if isempty(wolframscript)
        @info "TREEWAVEOLDG_WOLFRAMSCRIPT is not set: not regenerating particle_forms.jl"
    else
        root = pkgdir(TreeWaveOLDG)
        mktempdir() do dir
            out = joinpath(dir, "particle_forms.jl")
            run(pipeline(`$wolframscript -file $(joinpath(root, "derivations", "generate.wl"))
                              $out`; stdout=devnull))
            @test read(out) == read(joinpath(root, "src", "particle_forms.jl"))
        end
    end
end

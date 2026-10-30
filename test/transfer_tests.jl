# The transfer between particles and grid: the projection of a source onto
# the nodes, its adjoint pairing, the reconstruction of a grid function at
# a point, and the quadrature rules under them. A source the projection
# integrates wrongly near a particle drives the remainder with the wrong
# force, and a pairing that is not the projection's adjoint makes the
# particle's energy exchange with the field inconsistent, so the
# references here are independent of the rules they check.

using ForwardDiff
using LinearAlgebra: dot
using StaticArrays

const TWT = TreeWaveOLDG

# A deterministic, irregular grid function, standing in for a random one.
irregular(n) = [sinpi(0.1 + 0.7548776662 * i^2) for i in 1:n]

# `⟨a, b⟩_w` over variable `var`, through TreeAMR's fold.
function weighted_inner(p::WaveProblem{T,D}, a, b, var) where {T,D}
    el = p.element
    return mesh_fold((acc, q) -> acc + TWT.node_weight(q.I, q.h, el, Val(D)) * q.u[1], +,
                     zero(T), p.fs, a .* b; vars=var)
end

# The elements of a problem as `(block, zero-based element, lo, hi)`.
function all_elements(p::WaveProblem{T,D}) where {T,D}
    P1 = TWT.order(p.element) + 1
    E = p.fs.forest.N ÷ P1
    list = []
    for b in 1:nblocks(p.fs), J in CartesianIndices(ntuple(_ -> E, D))
        key = blockkey(p.fs, b)
        e = Tuple(J) .- 1
        lo, hi = TWT.element_box(block_origin(p.fs.forest, key),
                                 spacing(T, p.fs.forest, key), e, Val(P1), Val(D))
        push!(list, (b, e, lo, hi))
    end
    return list
end

# An independent reference for `∫ f` over a box: composite Gauss–Legendre
# one dimension at a time, with breakpoints graded geometrically toward
# the peak `z` on the length `scale` and, where the line crosses a bump of
# radius `ℓ` (`nothing` for none), at the crossing and graded toward it.
# No sinh rule, no substitution.
const REF_X, REF_W = let (x, w) = gauss_nodes_weights(16)
    Float64.(x), Float64.(w)
end

function composite(g, a, b, breaks)
    pts = sort!(unique!([a; filter(x -> a < x < b, breaks); b]))
    acc = 0.0
    for k in 1:(length(pts) - 1)
        half = (pts[k + 1] - pts[k]) / 2
        for (x, w) in zip(REF_X, REF_W)
            acc += w * half * g(pts[k] + half * (1 + x))
        end
    end
    return acc
end

function reference_integral(f, lo, hi, z, scale, grades, ℓ, prefix=())
    d = length(prefix) + 1
    D = length(lo)
    breaks = [z[d] + s * scale * 2.0^k for k in grades for s in (-1, 1)]
    r² = sum((prefix[j] - z[j])^2 for j in 1:(d - 1); init=0.0)
    if ℓ !== nothing && r² < ℓ^2
        c = sqrt(ℓ^2 - r²)
        append!(breaks, [z[d] + s * c * (1 - 2.0^-k) for k in 0:12 for s in (-1, 1)])
    end
    g = d == D ? (x -> f(SVector((prefix..., x)...))) :
        (x -> reference_integral(f, lo, hi, z, scale, grades, ℓ, (prefix..., x)))
    return composite(g, lo[d], hi[d], breaks)
end

# A compact bump of radius ℓ in D dimensions, `C³` at its edge like ρ₀.
bump(r², ℓ) = r² < ℓ^2 ? (1 - r² / ℓ^2)^4 : zero(r²)

@testset "The tables are the BigFloat ones, rounded once" begin
    # Guards against building the transfer tables in the run's type, which
    # would lose digits in the irrational Gauss nodes and Lagrange values,
    # and against a Gauss rule that is not exact to degree 2n - 1.
    for n in (1, 2, 5, 12)
        x, w = gauss_nodes_weights(n)
        for k in 0:(2n - 1)
            @test sum(w .* x .^ k) ≈ (iseven(k) ? 2 / big(k + 1) : 0) atol = 1e-60
        end
        @test sum(w .* x .^ 2n) - 2 / big(2n + 1) < -1e-10
        @test Float64.(x) == -reverse(Float64.(x))
    end
    tb = TransferTables(BigFloat, 3)
    t32 = TransferTables(Float32, 3)
    rounded(v::Number) = Float32(v)
    rounded(v::Tuple) = map(rounded, v)
    fields(v) = map(f -> getfield(v, f), fieldnames(typeof(v)))
    for f in fieldnames(typeof(tb))
        a, b = getfield(t32, f), getfield(tb, f)
        @test (b isa Tuple{Vararg{TWT.GaussRule}} ? map(fields, a) == map(rounded ∘ fields, b) :
               a == rounded(b))
    end
    # The contraction table is the Lagrange table weighted, so the far rule
    # projects a polynomial of degree P - 1 onto its nodal values.
    t = TransferTables(Float64, 3)
    for i in 1:4, k in 1:6
        @test t.contract[i][k] ≈ t.gaussweights[k] * t.lagrange[k][i] / t.weights[i]
    end
end

@testset "The projection is consistent at order P" begin
    # Guards the far rule's points, its contraction one dimension at a
    # time, and the division by the node weight: for a smooth source,
    # `(P S)ₙ - S(xₙ)` falls at the scheme's order, as the lumped mass does.
    function projection_error(D, P, roots)
        forest = wave_forest(Float64, Val(D), P, 2; roots=roots)
        el = Element(Float64, P)
        p = WaveProblem(forest, el)
        t = TransferTables(Float64, P)
        S(x) = sinpi(2x[1]) * prod(d -> cospi(2x[d]), 2:D; init=1.0) + 0.3
        du = statevector(p.fs)
        du .= 0
        project_source!(du, p, t, S, (); ball=(SVector(ntuple(_ -> 0.5, D)), 2.0),
                        factor=1.0)
        fill_at_nodes!((x, v) -> v == 2 ? S(SVector(x)) : 0.0, p.fs, el)
        ref = statevector(p.fs)
        gather!(ref, p.fs)
        return quadrature_norm(p.fs, el, du .- ref; vars=2:2)
    end
    for D in (1, 2), P in (2, 3, 4)
        errs = [projection_error(D, P, roots) for roots in (2, 4, 8)]
        @test convergence_rate([1 / 2, 1 / 4, 1 / 8], errs) > P - 0.1
    end
    errs = [projection_error(3, 2, roots) for roots in (2, 4)]
    @test log2(errs[1] / errs[2]) > 2 - 0.2
end

@testset "The projection is adjoint to the reconstruction" begin
    # Guards the identity the particle's energy exchange rests on,
    # `⟨P S, g⟩_w = ∫ S g_h`, for a source with a compact bump straddling
    # an element face (the nested rule split at its support), a
    # Lorentzian peak (the sinh rule), and a smooth part (the far rule
    # where the elements are far), against a graded composite rule.
    for (D, roots, ε, grades, tol) in ((1, 8, 0.1, -12:6, 1e-13),
                                       (2, 4, 0.1, -10:4, 1e-12),
                                       (3, 1, 0.25, -2:1, 1e-12))
        P = 3
        forest = wave_forest(Float64, Val(D), P, 2; roots=roots)
        el = Element(Float64, P)
        p = WaveProblem(forest, el)
        t = TransferTables(Float64, P)
        hₑ = minimum_spacing(forest) * (P + 1)
        ℓ = 0.05hₑ
        z = SVector(ntuple(d -> 0.5 + (d == 1 ? 0.3ℓ : 0.11d * hₑ), D))
        bumpD = D < 3                    # in 3D the reference is too slow for the bump
        S(x) = (r² = sum(abs2, x - z);
                cos(x[1]) + (bumpD ? bump(r², ℓ) / ℓ^D : 0.0) +
                ε * hₑ / (r² + (ε * hₑ)^2)^(D == 3 ? 1 : 1 / 2))
        du = statevector(p.fs)
        du .= 0
        s = ParticleSupport(z, zero(z), ℓ)
        project_source!(du, p, t, S, (s,); ball=(z, 10.0), factor=1.0)
        g = irregular(length(du))
        lhs = weighted_inner(p, du, g, 2)
        rhs = sum(all_elements(p)) do (b, e, lo, hi)
            gv = TWT.element_values(statearray(g, p.fs), b, e, 2, Val(P + 1))
            reference_integral(x -> S(x) * dot(gv, TWT.element_basis(t, lo, hi, x)), lo,
                               hi, z, bumpD ? ℓ : ε * hₑ, grades,
                               bumpD ? ℓ : nothing)
        end
        @test lhs ≈ rhs rtol = tol
    end
end

@testset "The near rule integrates the particle to 1e-12, at any size" begin
    # Guards the split at the support: the profile is only C³ at its edge,
    # where Gauss and sinh rules alike converge algebraically (5e-5 at
    # 128 points for ℓ/hₑ = 1e-3). Split there, `∫ ρ φₙ` is exact up to
    # the inner rule's order whatever ℓ/hₑ, for a particle at rest and a
    # moving one whose support straddles an element corner. The reference
    # is the ball rule in the rest frame, exact for `ρ` times a polynomial
    # of degree `P` per dimension, which is what the pairing of `P ρ` with
    # a polynomial `g` integrates; `g = 1` gives `Q/γ`.
    P = 3
    forest = wave_forest(Float64, Val(3), P, 1; roots=2)
    el = Element(Float64, P)
    p = WaveProblem(forest, el)
    t = TransferTables(Float64, P)
    hₑ = minimum_spacing(forest) * (P + 1)
    ys, ws = ball_rule(Float64, 12, 12)
    poly(x) = 1 + x[1] - 2x[2]^2 * x[3] + x[1]^3 * x[2]^3 * x[3]^3
    g = (fill_at_nodes!((x, v) -> poly(SVector(x)), p.fs, el);
         u = statevector(p.fs);
         gather!(u, p.fs);
         u)
    for ratio in (1e-1, 1e-2, 1e-3, 1e-4),
        v in (SVector(0.0, 0.0, 0.0), SVector(0.1, -0.15, 0.05))
        ℓ = ratio * hₑ
        m = ParticleModel(; Q=1e-3, ℓ=ℓ, M=1.0)
        z = SVector(0.5, 0.5, 0.5) + ℓ * SVector(0.3, -0.2, 0.1)
        γ = 1 / sqrt(1 - dot(v, v))
        Λinv(y) = y - (γ / (γ + 1) * dot(v, y)) * v       # Λ(v)⁻¹ y
        exact(f) = sum(w * m.Q * TWT._rho0(ℓ^2 * dot(y, y), ℓ) *
                       f(z + Λinv(ℓ * y)) * ℓ^3 for (y, w) in zip(ys, ws)) / γ
        s = ParticleSupport(m, Trajectory(; z=z, v=v))
        ρ(x) = particle_density(m, x, z, v)
        du = statevector(p.fs)
        du .= 0
        project_source!(du, p, t, ρ, (s,); ball=(z, 2ℓ), factor=1.0, var=1)
        unit = statevector(p.fs)
        unit .= 1
        @test weighted_inner(p, du, unit, 1) ≈ m.Q / γ rtol = 1e-12
        @test weighted_inner(p, du, g, 1) ≈ exact(poly) rtol = 1e-12
        @test pairing(p, t, g, ρ, s) ≈ exact(poly) rtol = 1e-12
        @test length(support_elements(p, s)) == 8
    end
end

@testset "Near a particle the sinh rule's points grow like log(hₑ/ℓ)" begin
    # Guards the sinh rule's scale and its projection onto each line: for
    # the field of a particle of size ℓ just outside an element face, a
    # rule of 16 + 8·log₁₀(hₑ/ℓ) points per line reaches 1e-12 from ℓ/hₑ =
    # 1e-1 to 1e-4, where a Gauss rule on the element would need ever
    # more.
    P = 3
    lo = SVector(0.0, 0.0, 0.0)
    hi = SVector(1.0, 1.0, 1.0)
    tref = TransferTables(Float64, P; nsinh=80)
    for ratio in (1e-1, 1e-2, 1e-3, 1e-4)
        ℓ = ratio
        z = SVector(1 + 1.01ℓ, 0.3, 0.6)
        s = ParticleSupport(z, zero(z), ℓ)
        f(t) = x -> (r² = sum(abs2, x - z);
                     (r² < ℓ^2 ? (3 - r² / ℓ^2) / 2ℓ : 1 / sqrt(r²)) *
                     TWT.element_basis(t, lo, hi, x))
        rule(t) = TWT.nested_rule(f(t), zero(SVector{64,Float64}), s, t, lo, hi,
                                  SVector{0,Float64}(), Val(3), false)
        ref = rule(tref)
        n = 16 + 8 * round(Int, log10(1 / ratio))
        @test maximum(abs, rule(TransferTables(Float64, P; nsinh=n)) - ref) <
              1e-12 * maximum(abs, ref)
    end
end

@testset "The transfer is smooth in z across faces and r_near" begin
    # Guards against a jump as the particle moves: where the support
    # crosses an element face, an edge or a corner, the pairing with a
    # polynomial follows the exact value to 1e-12, so its second
    # differences in z have no jump above that; and where an element
    # switches between the far and the near rule, or the support's split
    # appears, the projection of a source that does not move stays put to
    # 1e-12, so the switch is invisible.
    P = 3
    forest = wave_forest(Float64, Val(3), P, 1; roots=2)
    el = Element(Float64, P)
    p = WaveProblem(forest, el)
    t = TransferTables(Float64, P)
    hₑ = minimum_spacing(forest) * (P + 1)
    ℓ = 1e-3
    m = ParticleModel(; Q=0.07, ℓ=ℓ, M=1.0)
    ys, ws = ball_rule(Float64, 12, 12)
    poly(x) = 1 + x[1] - 2x[2]^2 * x[3] + x[1]^3 * x[2]^3 * x[3]^3
    g = (fill_at_nodes!((x, v) -> poly(SVector(x)), p.fs, el);
         u = statevector(p.fs);
         gather!(u, p.fs);
         u)
    # Across a face, an edge and a corner.
    for (start, dir) in ((SVector(0.5, 0.3, 0.2), SVector(1.0, 0.0, 0.0)),
                         (SVector(0.5, 0.5, 0.2), SVector(1.0, 1.0, 0.0) / sqrt(2)),
                         (SVector(0.5, 0.5, 0.5), SVector(1.0, 1.0, 1.0) / sqrt(3)))
        zs = [start + (k * ℓ / 4) * dir for k in -6:6]
        num = [pairing(p, t, g, x -> particle_density(m, x, z, zero(z)),
                       ParticleSupport(z, zero(z), ℓ)) for z in zs]
        exact = [sum(w * m.Q * TWT._rho0(ℓ^2 * dot(y, y), ℓ) * poly(z + ℓ * y) * ℓ^3
                     for (y, w) in zip(ys, ws)) for z in zs]
        Δ²(f) = [f[k - 1] - 2f[k] + f[k + 1] for k in 2:(length(f) - 1)]
        @test maximum(abs, Δ²(num) - Δ²(exact)) < 4e-12 * maximum(abs, exact)
    end
    # A source that does not move, projected with a particle that does:
    # across r_near (here 1.5 hₑ) and across the face the support meets.
    # The source is entire, so that the far rule integrates it to roundoff
    # too.
    S(x) = cos(x[1] - x[2] / 2) * exp(x[3] / 2)
    project(z) = (du = statevector(p.fs);
                  du .= 0;
                  project_source!(du, p, t, S, (ParticleSupport(z, zero(z), ℓ),);
                                  ball=(SVector(0.5, 0.5, 0.5), 2.0), rnear=1.5,
                                  factor=1.0);
                  du)
    base = project(SVector(5.0, 5.0, 5.0))    # every element far
    for z in (SVector(0.5 + 1.5hₑ + 1e-6, 0.3, 0.2),
              SVector(0.5 + 1.5hₑ - 1e-6, 0.3, 0.2),
              SVector(0.5 - 0.3ℓ, 0.3, 0.2), SVector(0.5 + 0.3ℓ, 0.5 - 0.2ℓ, 0.2))
        @test maximum(abs, project(z) - base) < 1e-12 * maximum(abs, base)
    end
end

@testset "The graded rule is the tensor Gauss rule, contracted" begin
    # Guards the graded rule's contraction one dimension at a time as the
    # points are visited, in the state's node order, against the tensor
    # Gauss rule applied point by point with the element's basis.
    for (D, P) in ((1, 4), (2, 3), (3, 3))
        t = TransferTables(Float64, P)
        g = t.graded[2]
        lo = SVector(ntuple(d -> 0.1d, D))
        hi = lo .+ 0.3
        S(x) = exp(x[1]) * prod(d -> cos(3x[d] + d), 1:D)
        gt = TransferTables(Float64, P; q=length(g.nodes))
        ref = TWT.gauss_rule(x -> S(x) * TWT.element_basis(gt, lo, hi, x),
                             zero(SVector{(P + 1)^D,Float64}), gt, lo, hi) ./
              TWT.element_weights(gt, 0.3, Val(D))
        @test TWT.tensor_rule(S, g, lo, hi, lo, Val(D)) ≈ ref rtol = 1e-14
    end
end

@testset "Each rule meets 1e-12 from where it applies, on the first run's source" begin
    # Guards the graded rule's bands and r_far, which are measurements: on
    # the source of INSPIRAL.md's first run (hₑ = d/8, orders 1–3), an
    # element whose nearest point is at the inner edge of a band, or at
    # r_far for the far rule, integrates `S φₙ` to 1e-12 of `∫|S|` against
    # a 32-point tensor Gauss rule, in 24 directions spread over the sphere
    # (elements within hₑ of the window's spheres left out, where the far
    # rule is limited by the window). Where one fails, a band's points or
    # its distance must change, and "Quadrature" in CODE.md with them.
    P = 3
    hₑ = 1 / 8
    q = sqrt(0.005)
    Ω = 0.4
    m₁ = ParticleModel(; Q=q, ℓ=1e-3, M=1.0)
    m₂ = ParticleModel(; Q=-q, ℓ=1e-3, M=1.0)
    r = SVector(0.5, 0.0, 0.0)
    v = SVector(0.0, Ω / 2, 0.0)
    tr₁ = Trajectory(; z=r, v=v, a=-Ω^2 * r, ȧ=-Ω^2 * v, ä=Ω^4 * r, a⃛=Ω^4 * v)
    tr₂ = Trajectory(; z=-r, v=-v, a=Ω^2 * r, ȧ=Ω^2 * v, ä=-Ω^4 * r, a⃛=-Ω^4 * v)
    R₁, R₂ = 1.25, 2.5
    S = ParticleSource((m₁, m₂), Window(zero(r), R₁, R₂; continuity=P + 1), (tr₁, tr₂))
    t = TransferTables(Float64, P)
    ref = TransferTables(Float64, P; q=32, graded=((0, 32),))
    far = TransferTables(Float64, P; graded=((0, P + 3),))
    @test far.graded[1].nodes == t.gauss
    w = TWT.element_weights(t, hₑ, Val(3))
    distance(lo, hi, z) = sqrt(TWT.box_distance2(z, lo, hi, 1:3))
    rules = (zip(t.gradedfrom, t.graded)..., (t.rfar, far.graded[1]))
    for (δ, g) in rules, k in 1:24
        # A direction on the golden spiral, the point δ hₑ from z₁ along it,
        # and the element on the far side of that point, offset across.
        c = 1 - (2k - 1) / 24
        φ = 2π * 0.618034 * k
        n = SVector(sqrt(1 - c^2) * cos(φ), sqrt(1 - c^2) * sin(φ), c)
        pt = tr₁.z + δ * hₑ * n
        lo = SVector(ntuple(d -> abs(n[d]) < 0.2 ? pt[d] - mod(0.618034 * (k + d), 1) * hₑ :
                                 n[d] > 0 ? pt[d] : pt[d] - hₑ, 3))
        hi = lo .+ hₑ
        min(distance(lo, hi, tr₁.z), distance(lo, hi, tr₂.z)) >= (δ - 0.05) * hₑ || continue
        rmin, rmax = distance(lo, hi, zero(r)), sqrt(sum(max.(abs.(lo), abs.(hi)) .^ 2))
        any(R -> rmin - hₑ < R < rmax + hₑ, (R₁, R₂)) && continue
        exact = TWT.tensor_rule(S, ref.graded[1], lo, hi, lo, Val(3))
        absS = TWT.gauss_rule(x -> abs(S(x)), 0.0, ref, lo, hi)
        @test maximum(abs, w .* (TWT.tensor_rule(S, g, lo, hi, lo, Val(3)) - exact)) <
              1e-12 * absS
    end
end

@testset "The pairing over the support equals the fold over the mesh" begin
    # Guards the host-side pairing, which finds the support's elements
    # from z alone and reads the state directly, against the reference it
    # replaces: a projection of the kernel over the whole mesh, folded
    # with the grid function, for `ρ` and for `∇ρ` (a vector), at rest
    # and moving.
    P = 3
    forest = wave_forest(Float64, Val(3), P, 1; roots=2)
    el = Element(Float64, P)
    p = WaveProblem(forest, el)
    t = TransferTables(Float64, P)
    hₑ = minimum_spacing(forest) * (P + 1)
    m = ParticleModel(; Q=0.07, ℓ=0.01hₑ, M=1.0)
    g = irregular(length(statevector(p.fs)))
    for v in (SVector(0.0, 0.0, 0.0), SVector(0.2, 0.1, 0.0))
        z = SVector(0.5, 0.5 + 0.004hₑ, 0.3)
        s = ParticleSupport(m, Trajectory(; z=z, v=v))
        ρ(x) = particle_density(m, x, z, v)
        ∇ρ(x) = ForwardDiff.gradient(ρ, x)
        fold(K) = (du = statevector(p.fs);
                   du .= 0;
                   project_source!(du, p, t, K, (s,); ball=(z, 2m.ℓ), factor=1.0, var=1);
                   weighted_inner(p, du, g, 1))
        @test pairing(p, t, g, ρ, s) ≈ fold(ρ) rtol = 1e-13
        ∇fold = SVector(ntuple(d -> fold(x -> ∇ρ(x)[d]), 3))
        @test pairing(p, t, g, ∇ρ, s) ≈ ∇fold rtol = 1e-12
    end
end

@testset "The reconstruction is the element's polynomial" begin
    # Guards `reconstruct`'s element lookup and Lagrange evaluation away
    # from the nodes: exact on polynomials of degree P per dimension, with
    # its gradient, and equal to the stored value at a node.
    for (D, P) in ((1, 4), (2, 3), (3, 2))
        forest = wave_forest(Float64, Val(D), P, 2; roots=2)
        el = Element(Float64, P)
        p = WaveProblem(forest, el)
        t = TransferTables(Float64, P)
        poly(x) = prod(d -> 1 + d * x[d] - x[d]^P, 1:D)
        fill_at_nodes!((x, v) -> poly(x), p.fs, el)
        u = statevector(p.fs)
        gather!(u, p.fs)
        for k in 1:20
            x = SVector(ntuple(d -> mod(0.3 + 0.618k * d, 1.0), D))
            @test reconstruct(p, t, u, x) ≈ poly(x) rtol = 1e-13
            @test reconstruct_gradient(p, t, u, x) ≈ ForwardDiff.gradient(poly, x) rtol =
                1e-12
        end
        w = irregular(length(u))
        ua = statearray(w, p.fs)
        for b in (1, nblocks(p.fs)), I in CartesianIndices(ntuple(_ -> forest.N, D))
            x = SVector(node_coordinates(p.fs, el, b, Tuple(I)))
            # A face node's position is also the next element's first node,
            # and a point on a face reconstructs the upper element.
            any(i -> (i - 1) % (P + 1) == P, Tuple(I)) && continue
            @test reconstruct(p, t, w, x; var=2) ≈ ua[Tuple(I)..., 2, b] atol = 1e-14
        end
    end
end

@testset "The ball rule is exact on what it should be" begin
    # Guards the ball rule's radial weight and its product rule on the
    # sphere: the volume and the moments of the unit ball, and the
    # charge of the particle profile.
    ys, ws = ball_rule(Float64, 8, 8)
    @test sum(ws) ≈ 4π / 3 rtol = 1e-14
    @test sum(w * y[1]^2 for (y, w) in zip(ys, ws)) ≈ 4π / 15 rtol = 1e-14
    @test sum(w * y[1]^2 * y[2]^2 * y[3]^2 for (y, w) in zip(ys, ws)) ≈ 4π / 945 rtol =
        1e-13
    @test abs(sum(w * y[1]^3 * y[2] for (y, w) in zip(ys, ws))) < 1e-15
    ℓ = 1e-3
    charge = sum(w * TWT._rho0(ℓ^2 * dot(y, y), ℓ) * ℓ^3 for (y, w) in zip(ys, ws))
    @test charge ≈ 1 rtol = 1e-14
end

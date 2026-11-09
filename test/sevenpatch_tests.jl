# The seven-patch ball: TreeAMR's forest of a cube and six shells, glued
# by the identity and by quarter turns (its M17), mapped to `|x| ≤ R1` by
# `SevenPatchBall`, a full geometry. The gradient crosses the glues in
# Cartesian components, one set with ghosts on every face, and the logical
# flux is formed from it where the Laplacian reads it. The scheme's
# promises are the annulus's, in a general metric: the discrete energy
# `½ Σ w √g (v² + |∇u|²)` is invariant across every glue, with the
# truncation orthogonal in the `√g`-weighted inner product per element;
# and a plane wave, with itself as the Dirichlet data on the outer sphere,
# converges at order `P`. Uses `scrambled`, `node_weights` and `state_of`
# from `refined_tests.jl`, and `metric_inner` and `statevar` from
# `annulus_tests.jl`.

using LinearAlgebra: det

# The identity map as a full geometry, for a brick in flat space: `√g = 1`
# and `S = I`, so the Cartesian path's sums are the flat path's terms
# plus exact zeros.
struct IdentityFrame <: FullGeometry end
TreeWaveOLDG.geometry_frame(::IdentityFrame, x, p) =
    (one(x[1]), ntuple(d -> ntuple(i -> d == i ? one(x[1]) : zero(x[1]), length(x)),
                       length(x)))
TreeWaveOLDG.physical_coordinates(::IdentityFrame, x, p) = Tuple(x)
TreeWaveOLDG.check_geometry(::IdentityFrame, forest) = nothing
TreeWaveOLDG.geometry_boundary(::IdentityFrame) = false

# Dirichlet data that are zero everywhere.
struct ZeroData end
(::ZeroData)(x, t) = (0.0, 0.0, 0.0, 0.0)

# The owned part of every variable of a field set's working array,
# `(N…, var, block)`.
ownedvars(fs::FieldSet{T,D}) where {T,D} =
    fs.work[ntuple(d -> (1:(fs.forest.N)) .+ fs.G[d], D)..., :, :]

@testset "With the identity map the Cartesian path is the flat path, bit for bit" begin
    # On a periodic brick with `√g = 1` and `S = I` the gradient pass writes
    # `∂_i u` into component `i`, and the Laplacian pass forms `F^d` as that
    # component plus exact zeros: the right-hand side must be the flat
    # one's to the bit. A component written to the wrong variable, a ghost
    # read with the wrong face's frame, or a window shifted by one plane
    # makes them differ. The per-element truncation is then the element's
    # tensor projector, built another way, so equal to rounding.
    for D in (2, 3), P in (2, 3)
        el = Element(Float64, P)
        forest = wave_forest(Float64, Val(D), P, 2; roots=2)
        pf = WaveProblem(forest, el)
        pc = WaveProblem(forest, el; geometry=IdentityFrame())
        @test pc.grads == () && pc.cartgrad.nvars == D
        u = statevector(pf.fs)
        u .= scrambled(length(u), 0.4)
        duf, duc = similar(u), similar(u)
        wave_rhs!(duf, u, pf, 0.0)
        wave_rhs!(duc, u, pc, 0.0)
        @test duf == duc
        @test wave_energy(pc, u) ≈ wave_energy(pf, u) rtol = 1e-14
        @test truncate_state!(copy(u), pc) ≈ truncate_state!(copy(u), pf) rtol = 1e-13
        @test pc.metric.speed == 1
    end
end

@testset "The seven-patch ball's frame is its map's, and it refuses what it does not map" begin
    # `S[d] = √g ∇ξ^d` is the cofactor of the map's Jacobian; a frame taken
    # from the transpose, from `∂x/∂ξ` instead of its inverse, or without
    # `√g`, differs from the finite differences of the map by `O(1)`. The
    # cube is affine, so there it is exact. Every shell's inner face is the
    # cube's face, and its outer face the sphere.
    forest, geo = sevenpatch_forest(Float64, 2, 2; roots=(2, 2), Lc=1, R1=3)
    x = (0.3, 1.7, 0.9)
    ε = 1e-6
    for q in 1:7
        J, S = geometry_frame(geo, x, q)
        Jm = hcat([collect((physical_coordinates(geo, ntuple(j -> x[j] + (j == e) * ε, 3),
                                                 q) .-
                            physical_coordinates(geo, ntuple(j -> x[j] - (j == e) * ε, 3),
                                                 q)) ./ (2ε)) for e in 1:3]...)
        Sfd = abs(det(Jm)) .* inv(Jm)
        @test all(abs(S[d][i] - Sfd[d, i]) < 1e-8 * max(1, J) for d in 1:3, i in 1:3)
        @test J > 0
    end
    @test geometry_frame(geo, x, 1) == (1.0, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                              (0.0, 0.0, 1.0)))
    for (k, s) in TreeWaveOLDG.SEVENPATCH_SHELLS
        q = TreeWaveOLDG.sevenpatch_index(k, s)
        inner = ntuple(j -> j == k ? (s == 1 ? 0.0 : 2.0) : 0.6, 3)
        outer = ntuple(j -> j == k ? (s == 1 ? 2.0 : 0.0) : 0.6, 3)
        @test physical_coordinates(geo, inner, q) ==
              ntuple(j -> j == k ? Float64(s) : -0.4, 3)
        @test sqrt(sum(abs2, physical_coordinates(geo, outer, q))) ≈ 3 rtol = 1e-15
    end
    el = Element(Float64, 2)
    wave = PlaneWave((1.0, 0.0, 0.0), 0.0)
    brick = wave_forest(Float64, Val(3), 2, 2; roots=2)
    @test_throws "sevenpatch_forest" WaveProblem(brick, el; geometry=geo, boundary=wave)
    @test_throws "sevenpatch_forest" WaveProblem(forest, el; boundary=wave,
                                                 geometry=SevenPatchBall(1.0, 3.0, (2, 4)))
    @test_throws "odd" sevenpatch_forest(Float64, 2, 3)
    @test_throws "√3 Lc" sevenpatch_forest(Float64, 2, 2; Lc=1, R1=1.5)
    @test_throws "no data were given" WaveProblem(forest, el; geometry=geo)
    # Without a geometry the gradient is a set per dimension, which TreeAMR
    # does not carry across a quarter turn.
    @test_throws "hold their points alike" WaveProblem(forest, el)
    @test_throws "full geometry only" WaveProblem(brick, el; boundary=wave)
    refined, geo2 = sevenpatch_forest(Float64, 2, 2; roots=(2, 2))
    refine!(refined, refined.leaves[1:1])
    balance!(refined)
    @test_throws "uniform forest" WaveProblem(refined, el; geometry=geo2, boundary=wave)
    walled = wave_forest(Float64, Val(3), 2, 2; roots=2, reflecting=true)
    @test_throws "Cartesian one" WaveProblem(walled, el; geometry=IdentityFrame())
end

@testset "On the ball the truncation is the √g-orthogonal projection onto Q_{P-1}" begin
    # `√g` does not factorize on the shells, so the problem builds one
    # projector per element; a table built from the wrong element, or in
    # the flat weights, fails the adjointness. The element's own projector
    # is the identity on `Q_{P-1}` and on nothing else, so it leaves the
    # result.
    for P in (2, 3)
        el = Element(Float64, P)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(2, 2))
        p = WaveProblem(forest, el; geometry=geo, boundary=ZeroData())
        function flat_truncate(x)
            TreeWaveOLDG.launch_truncation!(TreeWaveOLDG.CPU(), statearray(x, p.fs), el,
                                            p, forest.N ÷ (P + 1))
            return x
        end
        n = length(statevector(p.fs))
        a, b = statevector(p.fs), statevector(p.fs)
        a .= scrambled(n, 0.1)
        b .= scrambled(n, 0.6)
        Πa, Πb = truncate_state!(copy(a), p), truncate_state!(copy(b), p)
        va, vb = statevar(p, a, 2), statevar(p, b, 2)
        Πva, Πvb = statevar(p, Πa, 2), statevar(p, Πb, 2)
        scale = sqrt(metric_inner(p, va, va) * metric_inner(p, vb, vb))
        @test abs(metric_inner(p, Πva, vb) - metric_inner(p, va, Πvb)) < 1e-14 * scale
        @test truncate_state!(copy(Πa), p) ≈ Πa rtol = 1e-13
        @test flat_truncate(copy(Πa)) ≈ Πa rtol = 1e-13
        @test !(flat_truncate(copy(a)) ≈ a)
        @test statevar(p, Πa, 1) == statevar(p, a, 1)
        @test metric_inner(p, Πva, Πva) <= metric_inner(p, va, va)
    end
end

@testset "The energy is invariant across the ball's cube–shell and shell–shell glues" begin
    # `dE/dt = Σ w √g v v̇ + Σ w √g ∇u ⋅ ∇v` vanishes when the wide
    # derivative is summation by parts across every glue, which holds when
    # the ghost's flux `F^d = S[d] ⋅ ∇u` is the neighbour's own: the
    # Cartesian gradient carried by the glue unchanged, contracted with the
    # frame of the face node beside the ghost. A frame read at the wrong
    # node, a component permuted by a quarter turn, or a divergence not
    # divided by `√g` leaves a rate of the size of its terms. The state
    # vanishes in the shells' outermost elements and the Dirichlet data are
    # zero, so nothing crosses the outer sphere.
    for P in (2, 3), truncate in (false, true)
        el = Element(Float64, P)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(2, 2))
        p = WaveProblem(forest, el; geometry=geo, boundary=ZeroData(), truncate=truncate)
        u = statevector(p.fs)
        u .= scrambled(length(u), 0.2)
        ua = statearray(u, p.fs)
        N = forest.N
        he = (P + 1) / N                       # an element's logical width
        for b in 1:nblocks(p.fs)
            q = patch(forest, blockkey(p.fs, b))
            q == 1 && continue
            k, s = TreeWaveOLDG.SEVENPATCH_SHELLS[q - 1]
            for I in CartesianIndices((N, N, N))
                xk = coordinates(p.fs, b, Tuple(I))[k]
                (s == 1 ? xk >= 2 - he - 1e-12 : xk <= he + 1e-12) && (ua[I, :, b] .= 0)
            end
        end
        truncate_state!(u, p)
        du = similar(u)
        wave_rhs!(du, u, p, 0.0)
        Vu = ownedvars(p.cartgrad)
        v, vdot = statevar(p, u, 2), statevar(p, du, 2)
        TreeWaveOLDG.fill_gradient!(p, state_of(p, v), 0.0)
        Vv = ownedvars(p.cartgrad)
        dot3(A, B) = sum(selectdim(A, 4, i) .* selectdim(B, 4, i) for i in 1:3)
        J = selectdim(p.metric.coef, 4, 1)
        w = node_weights(p.fs, el)
        rate = metric_inner(p, v, vdot) + sum(w .* J .* dot3(Vu, Vv))
        scale = metric_inner(p, abs.(v), abs.(vdot)) + sum(w .* J .* abs.(dot3(Vu, Vv)))
        @test scale > 1
        @test abs(rate) < 1e-13 * scale
    end
end

@testset "TreeAMR fills the ball's sets across every glue, swapped axes included" begin
    # `u`'s set and the Cartesian gradient's have ghosts on every face and
    # the same layout along every axis, so TreeAMR's glue-layout check takes
    # them across the quarter turns; a set per dimension would be refused.
    forest, geo = sevenpatch_forest(Float64, 2, 2; roots=(2, 2))
    p = WaveProblem(forest, Element(Float64, 2); geometry=geo, boundary=ZeroData())
    @test occursin("turned across a glue", sprint(show, p.cartschedule))
    @test occursin("turned across a glue", sprint(show, p.uschedule))
    @test p.metric.speed > 1
    @test oldg_timestep(p; cfl=1) < minimum_spacing(forest) / 3
end

@testset "A plane wave converges at order P on the seven-patch ball" begin
    # The wave crosses every glue and enters and leaves through the outer
    # sphere, where it is the Dirichlet data. A glue, a frame or a ghost
    # contraction that is wrong to `O(1)` breaks the rate; the bounds are
    # the measured rates (CODE.md, "The seven-patch ball" under "Measured
    # results") with room either side.
    for P in (2, 3)
        rs = [sevenpatch_errors(; P=P, E=2, roots=(2s, 2s), tend=1/4) for s in (1, 2)]
        rate = convergence_rate([r.h for r in rs], [r.l2 for r in rs])
        @test P - 0.3 < rate < P + 0.4
    end
end

# The rate `Σ w √g v v̇ + Σ w √g ∇u ⋅ ∇v` of the energy of state `u`, and
# the same sum of absolute values, its scale.
function energy_rate(p::WaveProblem, u)
    du = similar(u)
    wave_rhs!(du, u, p, 0.0)
    Vu = ownedvars(p.cartgrad)
    v, vdot = statevar(p, u, 2), statevar(p, du, 2)
    TreeWaveOLDG.fill_gradient!(p, state_of(p, v), 0.0)
    Vv = ownedvars(p.cartgrad)
    dot3(A, B) = sum(selectdim(A, 4, i) .* selectdim(B, 4, i) for i in 1:3)
    J = selectdim(p.metric.coef, 4, 1)
    w = node_weights(p.fs, p.element)
    rate = metric_inner(p, v, vdot) + sum(w .* J .* dot3(Vu, Vv))
    scale = metric_inner(p, abs.(v), abs.(vdot)) + sum(w .* J .* abs.(dot3(Vu, Vv)))
    return rate, scale
end

# `-Σ w |S[d]| (κ v² + u v/R)` over the outer faces' nodes, `w` the face's
# quadrature weight, written out independently of the problem's own loop.
function radiative_rate(p::WaveProblem, u, κ)
    forest, el = p.fs.forest, p.element
    N, P1 = forest.N, length(el.weights)
    ua = statearray(u, p.fs)
    rate = 0.0
    for (b, d, σ) in p.boundary.faces
        h = spacing(Float64, forest, blockkey(p.fs, Int(b)))
        for I in CartesianIndices((N, N, N))
            I[d] == (σ > 0 ? N : 1) || continue
            w = prod(q -> q == d ? 1.0 : h * P1 / 2 * el.weights[(I[q] - 1) % P1 + 1], 1:3)
            S = [p.metric.coef[I, 1 + 3(d - 1) + i, b] for i in 1:3]
            v = ua[I, 2, b]
            rate -= w * sqrt(sum(abs2, S)) * (κ * v^2 + p.boundary.invR * ua[I, 1, b] * v)
        end
    end
    return rate
end

@testset "Through a radiative boundary the energy changes by its boundary term alone" begin
    # The ghosts set `u`'s face value to the face node's and the normal flux
    # to `-σ |S| (κ v + u/R)`, so the energy's rate is
    # `-Σ w |S| (κ v² + u v/R)` over the outer faces' nodes, to rounding. A
    # ghost written on the wrong side, a flux of the wrong sign, a frame
    # read at the ghost instead of the face node, or a `u` ghost that is not
    # the face node's (which leaves the time derivative of the ghost out of
    # the rate) leaves a difference of the size of the terms. `κ = 0`
    # without the `u/R` term holds the normal derivative at zero, and then
    # nothing crosses.
    for P in (2, 3), (κ, ir) in ((1.0, true), (1.0, false), (0.5, true), (0.0, false)),
        truncate in (false, true)

        el = Element(Float64, P)
        forest, geo = sevenpatch_forest(Float64, P, 2; roots=(2, 2))
        p = WaveProblem(forest, el; geometry=geo, truncate=truncate,
                        boundary=RadiativeBoundary(; κ=κ, inverse_r=ir))
        @test length(p.boundary.faces) == 6 * 2^2
        @test p.boundary.invR == (ir ? 1 / 3 : 0)
        u = statevector(p.fs)
        u .= scrambled(length(u), 0.3)
        truncate_state!(u, p)
        rate, scale = energy_rate(p, u)
        @test scale > 1
        @test abs(rate - radiative_rate(p, u, κ)) < 1e-13 * scale
        κ > 0 && @test radiative_rate(p, u, κ) < -1e-3 * scale
    end
end

@testset "A radiative boundary refuses what it cannot impose" begin
    # A negative `κ` feeds energy in; without a full geometry there is no
    # Cartesian gradient to form the normal flux from; and the `u/R` term
    # needs the geometry's outer radius.
    el = Element(Float64, 2)
    @test_throws "negative" RadiativeBoundary(; κ=-1)
    brick = wave_forest(Float64, Val(3), 2, 2; roots=2)
    @test_throws "full geometry" WaveProblem(brick, el; boundary=RadiativeBoundary())
    @test_throws "no outer radius" WaveProblem(brick, el; geometry=IdentityFrame(),
                                               boundary=RadiativeBoundary())
    p = WaveProblem(brick, el; geometry=IdentityFrame(),
                    boundary=RadiativeBoundary(; inverse_r=false))
    @test isempty(p.boundary.faces)
end

@testset "An outgoing pulse leaves through Sommerfeld's condition and stays behind a wall" begin
    # The pulse of width 1/2 starts at the origin and has crossed the sphere
    # `R1 = 3` by `t = 5`. Under Sommerfeld's condition what is left is the
    # reflection and the discretization error, 3.5e-3 of the energy at this
    # resolution (CODE.md, "The seven-patch ball" under "Measured
    # results"); a flux of the wrong sign or a ghost that is not written
    # keeps most of it. With `κ = 0` and no `u/R` term the face's flux is
    # zero, a Neumann wall, and the energy stays to RK4's loss, 2e-9.
    free = radiative_errors(; P=3, E=2, roots=(2, 2), tend=5)
    @test free.energy < 1e-2 * free.energy0
    @test free.l2 < 0.05
    wall = radiative_errors(; P=3, E=2, roots=(2, 2), tend=5, κ=0, inverse_r=false)
    @test abs(wall.energy / wall.energy0 - 1) < 1e-7
end

# Refined forests: the same kernels on a two-level forest, where TreeAMR's
# element-local operators fill the ghosts across coarse-fine faces. The
# scheme's two promises there are the uniform forest's: the wide
# derivative is summation by parts in the quadrature inner product, so the
# discrete energy is an exact invariant of the semi-discrete system, and it
# is exact on the polynomials its elements hold. A coarse ghost that is
# not the adjoint of the fine one breaks the first, and a ghost from the
# wrong side or the wrong nodes the second, and the convergence of a
# smooth mode hides both well. `test/reference.jl`'s global arrays are
# uniform-only, so the references here are inner products under the
# quadrature, with each block's weights scaled by its own spacing.

using LinearAlgebra: norm
using StaticArrays

const TWR = TreeWaveOLDG

# A deterministic, irregular sequence, standing in for a random one: the
# assertions follow from the setup and not from a draw.
scrambled(n, s) = [sinpi(s + 0.7548776662 * i^2 + 0.5698402910 * i) for i in 1:n]

# The owned part of the working array of a one-variable set, `(N…, block)`.
owned(fs::FieldSet{T,D}) where {T,D} =
    fs.work[ntuple(d -> (1:(fs.forest.N)) .+ fs.G[d], D)..., 1, :]

# The quadrature weight of every owned node, `(N…, block)`, from the
# element's weights and each block's own spacing.
function node_weights(fs::FieldSet{T,D}, el::Element{T,P1}) where {T,D,P1}
    N = fs.forest.N
    w = zeros(T, ntuple(_ -> N, D)..., nblocks(fs))
    for b in 1:nblocks(fs)
        h = spacing(T, fs.forest, blockkey(fs, b))
        for I in CartesianIndices(ntuple(_ -> N, D))
            w[I, b] = prod(d -> h * P1 / 2 * el.weights[(I[d] - 1) % P1 + 1], 1:D)
        end
    end
    return w
end

# A state vector of `p` with `a` in variable 1 and `c` in variable 2.
function state_of(p::WaveProblem{T,D}, a, c=zero(a)) where {T,D}
    s = statevector(p.fs)
    sa = statearray(s, p.fs)
    selectdim(sa, D + 1, 1) .= a
    selectdim(sa, D + 1, 2) .= c
    return s
end

# `D_d a` for every `d`, as the gradient pass computes it: `a` is
# exchanged as `u` is, even under a mirror.
function gradients(p::WaveProblem{T,D}, a) where {T,D}
    TWR.fill_gradient!(p, state_of(p, a))
    return map(owned, p.grads)
end

# `D_d g` for a field `g` laid out as `∂_d u`, as the Laplacian pass
# computes it: `g` is exchanged as `∂_d u` is, odd under a mirror in `d`.
function divergence(p::WaveProblem{T,D}, g, d) where {T,D}
    foreach(s -> s.work .= 0, p.grads)
    gs = p.grads[d]
    gs.work[ntuple(k -> (1:(gs.forest.N)) .+ gs.G[k], D)..., 1, :] .= g
    fill_ghosts!(gs, p.gradschedules[d])
    du = statevector(p.fs)
    map_blocks!(TWR.laplacian_kernel!, p.fs, statearray(du, p.fs),
                statearray(state_of(p, zero(g)), p.fs), map(s -> s.work, p.grads),
                p.spacings, p.element, p.valD)
    return copy(selectdim(statearray(du, p.fs), D + 1, 2))
end

# The forests the tests run on: the middle refined in a periodic box, and
# a corner touching two walls refined in a reflecting one, so that both
# kinds of box have fine blocks on the low and the high side of a
# coarse-fine face.
function refined_forest(D, P; reflecting=false)
    roots = D == 3 ? 3 : 4
    region = reflecting ? ntuple(_ -> (0.0, 0.5), D) : true
    return wave_forest(Float64, Val(D), P, 2; roots=roots, reflecting=reflecting,
                       refined=region)
end

@testset "The wide derivative stays skew-adjoint across coarse-fine faces" begin
    # Guards the coupling the energy argument needs: the coarse ghost must
    # be the adjoint of the fine one in the quadrature, `M_c R = Iᵀ M_f`,
    # and the per-block weights must scale with each block's spacing.
    # Then `⟨g, D_d u⟩_w + ⟨D_d g, u⟩_w = 0` for any `u` and `g`, with `u`
    # exchanged as the state and `g` as its gradient (a mirror makes the
    # first even and the second odd). Any other coarse ghost, a weight on
    # the wrong level, or a ghost from the wrong side leaves a residual of
    # the size of the terms.
    for (D, P) in ((1, 3), (2, 2), (2, 3), (3, 2)), reflecting in (false, true)
        forest = refined_forest(D, P; reflecting=reflecting)
        @test maxlevel(forest) == 1
        el = Element(Float64, P)
        p = WaveProblem(forest, el)
        w = node_weights(p.fs, el)
        n = length(w)
        u = reshape(scrambled(n, 0.1), size(w))
        Du = gradients(p, u)
        for d in 1:D
            g = reshape(scrambled(n, 0.3 + d / 7), size(w))
            Dg = divergence(p, g, d)
            ip(a, b) = sum(w .* a .* b)
            lhs = ip(g, Du[d]) + ip(Dg, u)
            scale = sqrt(ip(g, g) * ip(Du[d], Du[d])) + sqrt(ip(Dg, Dg) * ip(u, u))
            @test abs(lhs) < 1e-13 * scale
        end
    end
end

@testset "The discrete energy is an invariant of the semi-discrete system on a refined forest" begin
    # Guards the energy argument end to end through `wave_rhs!`: the rate
    # `dE/dt = ⟨v, ∂ₜv⟩_w + Σ_d ⟨D_d u, D_d ∂ₜu⟩_w` vanishes for any state,
    # and with truncation for any truncated state under the projected
    # rate `Π ∂ₜv`, since `Π` is self-adjoint in the same inner product.
    # A coupling across a coarse-fine face that is not summation by
    # parts leaves a rate of the size of its terms.
    for (D, P, reflecting) in ((1, 3, false), (2, 2, false), (2, 3, true), (3, 2, false)),
        truncate in (true, false)

        forest = refined_forest(D, P; reflecting=reflecting)
        el = Element(Float64, P)
        p = WaveProblem(forest, el; truncate=truncate)
        w = node_weights(p.fs, el)
        n = length(w)
        u = state_of(p, reshape(scrambled(n, 0.2), size(w)),
                     reshape(scrambled(n, 0.7), size(w)))
        truncate_state!(u, p)
        du = similar(u)
        wave_rhs!(du, u, p, 0.0)
        truncate_state!(du, p)
        V = selectdim(statearray(u, p.fs), D + 1, 2)
        Vt = selectdim(statearray(du, p.fs), D + 1, 2)
        Du = gradients(p, selectdim(statearray(u, p.fs), D + 1, 1))
        Dv = gradients(p, selectdim(statearray(du, p.fs), D + 1, 1))
        terms = [sum(w .* V .* Vt); [sum(w .* Du[d] .* Dv[d]) for d in 1:D]]
        @test abs(sum(terms)) < 1e-12 * sum(abs, terms)
        # And it is the rate of `wave_energy` itself, which is quadratic,
        # so that `E(u + du) - E(u - du)` is twice its rate along `du`.
        E₊, E₋ = wave_energy(p, u .+ du), wave_energy(p, u .- du)
        @test abs(E₊ - E₋) < 1e-12 * (E₊ + E₋)
    end
end

@testset "The right-hand side reads no edge or corner ghost" begin
    # Guards the kernels against TreeAMR's faces-only fill of element
    # sets: an edge or corner ghost is never written, so a window that
    # reached one would read whatever was there before. With every
    # working array poisoned with NaN first, a single such read shows.
    for (D, P, reflecting) in ((2, 2, false), (2, 3, true), (3, 2, false))
        forest = refined_forest(D, P; reflecting=reflecting)
        p = WaveProblem(forest, Element(Float64, P))
        u = initial_state(p, (x, v) -> sum(x)^2 + v)
        p.ufs.work .= NaN
        foreach(s -> s.work .= NaN, p.grads)
        du = similar(u)
        wave_rhs!(du, u, p, 0.0)
        @test all(isfinite, du)
        @test isfinite(wave_energy(p, u))
    end
end

@testset "Across a coarse-fine face the wide derivative is exact to the operators' degrees" begin
    # Guards what each side of a coarse-fine face reads: a fine ghost is
    # the coarse element's polynomial, exact to degree `P` tangentially,
    # and a coarse ghost the adjoint restriction, exact to degree `P - 1`
    # and not `P`. So `D_d` is exact on data of degree `P` along `d` and
    # `P - 1` across it, and on degree `P` across it at the fine nodes
    # only. A ghost taken from the wrong nodes or the wrong side is not
    # exact at all. The elements at the box's faces along `d` are left
    # out: the polynomial is not periodic.
    for (D, P) in ((2, 2), (2, 3), (3, 2))
        forest = refined_forest(D, P)
        el = Element(Float64, P)
        p = WaveProblem(forest, el)
        N = forest.N
        shifts = ntuple(k -> 0.25 * k, D)
        for d in 1:D, k in (P - 1, P)
            deg = ntuple(j -> j == d ? P : k, D)
            f(x) = prod(j -> (x[j] + shifts[j])^deg[j], 1:D)
            ∂f(x) = P * (x[d] + shifts[d])^(P - 1) *
                    prod(j -> j == d ? 1.0 : (x[j] + shifts[j])^deg[j], 1:D)
            fill_at_nodes!((x, v) -> v == 1 ? f(x) : 0.0, p.fs, el)
            s = statevector(p.fs)
            gather!(s, p.fs)
            Du = gradients(p, selectdim(statearray(s, p.fs), D + 1, 1))[d]
            errs = (zeros(0), zeros(0))                       # coarse, fine
            scale = 0.0
            for b in 1:nblocks(p.fs)
                key = blockkey(p.fs, b)
                origin = block_origin(forest, key)
                h = spacing(forest, key)
                for I in CartesianIndices(ntuple(_ -> N, D))
                    q = (I[d] - 1) ÷ (P + 1)
                    (origin[d] == 0 && q == 0) && continue
                    (origin[d] + N * h ≈ 1 && q == N ÷ (P + 1) - 1) && continue
                    x = node_coordinates(p.fs, el, b, Tuple(I))
                    push!(errs[level(key) + 1], abs(Du[I, b] - ∂f(x)))
                    scale = max(scale, abs(∂f(x)))
                end
            end
            @test maximum(errs[2]) < 1e-11 * scale
            if k == P - 1
                @test maximum(errs[1]) < 1e-11 * scale
            else
                @test maximum(errs[1]) > 1e-6 * scale
            end
        end
    end
end

@testset "Positions and norms on a refined forest are TreeAMR's, block by block" begin
    # Guards the per-block geometry: every node at its own block's origin
    # and spacing, bit for bit the element's offsets as this package
    # builds them and as TreeAMR's `coordinates` gives them; and a
    # quadrature that weights each block by its own spacing, so that it
    # integrates what the elements can represent and agrees with
    # TreeAMR's `volume_weighted_norm` of the element set.
    for (D, P) in ((2, 3), (3, 2))
        forest = refined_forest(D, P)
        el = Element(Float64, P)
        fs = FieldSet(forest, D; G=1, centering=elementcentered(D), P=P)
        fill_at_nodes!((x, v) -> x[v], fs, el)
        N = forest.N
        same = true
        for b in 1:nblocks(fs)
            key = blockkey(fs, b)
            origin = block_origin(forest, key)
            h = spacing(forest, key)
            for I in CartesianIndices(ntuple(_ -> N, D))
                x = node_coordinates(fs, el, b, Tuple(I))
                xr = ntuple(D) do d
                    q, i = divrem(I[d] - 1, P + 1)
                    origin[d] + h * (q * (P + 1) + el.offsets[i + 1])
                end
                same &= x === xr
                same &= x === coordinates(fs, b, Tuple(I) .+ 1)
                same &= all(d -> fs.work[(Tuple(I) .+ 1)..., d, b] === x[d], 1:D)
            end
        end
        @test same

        p = WaveProblem(forest, el)
        u = state_of(p, reshape(scrambled(nblocks(p.fs) * N^D, 0.4), ntuple(_ -> N, D)...,
                                nblocks(p.fs)),
                     reshape(scrambled(nblocks(p.fs) * N^D, 0.9), ntuple(_ -> N, D)...,
                             nblocks(p.fs)))
        for q in (1, 2)
            @test quadrature_norm(p.fs, el, u; p=q) ≈ volume_weighted_norm(p.fs, u; p=q) rtol =
                1e-14
        end
        @test quadrature_norm(p.fs, el, u; p=Inf) == volume_weighted_norm(p.fs, u; p=Inf)
        k = P - 1                        # |x^k|² has degree 2P-2 ≤ 2P-1
        fill_at_nodes!((x, v) -> prod(x)^k, p.fs, el)
        gather!(u, p.fs)
        @test quadrature_norm(p.fs, el, u) ≈ sqrt(1 / (2k + 1))^D rtol = 1e-13
    end
end

@testset "What a refined forest cannot run is refused, with the reason" begin
    # Guards the three ways a run could go silently wrong: an odd number
    # of elements per block, where a fine block would split a parent's
    # element; a field set whose order is not the element's, whose nodes
    # and weights would be another element's; and the particle transfer
    # on a refined forest, whose distances and rules are in the finest
    # element width.
    @test_throws "must be even" wave_forest(Float64, Val(2), 3, 3)
    @test_throws "E must be even" WaveProblem(
        Forest((2,); N=6, periodic=(true,), extents=((0.0, 1.0),)), Element(Float64, 1))

    forest = wave_forest(Float64, Val(2), 3, 2; roots=2)
    other = FieldSet(forest, 1; G=0, centering=elementcentered(2), P=3)
    cell = FieldSet(forest, 1; G=0)
    el = Element(Float64, 1)
    @test_throws "of order P = 3, and the element is of order P = 1" fill_at_nodes!(
        (x, v) -> 0.0, other, el)
    @test_throws "of order P = 3, and the element is of order P = 1" node_coordinates(
        other, el, 1, (1, 1))
    @test_throws "of order P = 3, and the element is of order P = 1" quadrature_norm(
        other, el, statevector(other))
    @test_throws "not element-centered" fill_at_nodes!((x, v) -> 0.0, cell,
                                                       Element(Float64, 3))

    @test_throws "no block centre" wave_forest(Float64, Val(2), 3, 2; roots=2,
                                               refined=true)

    L = 4.0
    P = 3
    wave = WaveProblem(wave_forest(Float64, Val(3), P, 2; roots=3, L=L, refined=true),
                       Element(Float64, P))
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=0.1, ℓ=0.01, M=1.0)
    window = Window(c, 1.0, 1.75; continuity=P + 1)
    motion = uniform_motion((c + SVector(0.11, 0.07, 0.13),), (SVector(0.0, 0.0, 0.0),))
    @test_throws "needs a uniform forest" InspiralProblem(wave, (m,), window, motion)
    t = TransferTables(Float64, P)
    s = ParticleSupport(m, Trajectory(; z=c, v=zero(c)))
    @test_throws "needs a uniform forest" project_source!(
        statevector(wave.fs), wave, t, x -> 0.0, (s,); ball=(c, 1.0))
end

@testset "The standing mode on a two-level forest lies between the uniform runs" begin
    # A smoke check of the whole run on a refined forest, periodic with
    # the middle refined and reflecting with a corner at two walls
    # refined: the error is between the uniform coarse forest's and the
    # uniform fine one's, and the energy lost is RK4's, small. The rates
    # and the RK4 limit on such forests are measured, not asserted here.
    for reflecting in (false, true)
        run(roots, refined) = wave_errors(Float64, Val(2); P=3, E=2, roots=roots,
                                          reflecting=reflecting, refined=refined)
        region = reflecting ? ((0.0, 0.5), (0.0, 0.5)) : true
        coarse, mixed, fine = run(4, false), run(4, region), run(8, false)
        @test fine.l2 < mixed.l2 < coarse.l2
        @test mixed.h == fine.h
        @test 0 <= (mixed.energy0 - mixed.energy) / mixed.energy0 < 1e-8
    end
end

# Adaptive refinement: the modal indicator, the regrid of a state, the
# problem rebuilt on the new mesh, and the adaptive blast wave that puts
# them together. An indicator that reads the wrong mode or the wrong
# nodes still refines somewhere, a regrid that loses mass or leaves `v`'s
# top mode in still runs, and a problem that keeps a stale schedule still
# steps, so each piece is checked on its own before the run is.
#
# `scrambled` and `node_weights` are `refined_tests.jl`'s.

const TWA = TreeWaveOLDG

# A smooth periodic function that is not a single mode: the indicator's
# rate is measured on it.
smooth_data(D) = (x, v) -> begin
    s = one(x[1])
    for d in 1:D
        s *= sinpi(2 * x[d] + 3 * d / 10 + v / 7)
    end
    return s
end

# Data of degree `n` in every coordinate, and not below it in any.
poly_data(D, n) = (x, v) -> begin
    s = one(x[1])
    for d in 1:D
        s *= (x[d] - 3 / 10)^n + x[d] / 2
    end
    return s
end

# A state vector of `p` holding the deterministic irregular `scrambled`
# values in both variables, with `v` truncated.
function scrambled_state(p::WaveProblem)
    u = statevector(p.fs)
    u .= scrambled(length(u), 0.2)
    return truncate_state!(u, p)
end

# The flags that refine the blocks whose centres lie in the middle half
# of the unit box, and keep the others.
middle_flags(forest::Forest{D}) where {D} = flag_blocks(forest) do b, k
    ext = block_extent(forest, k)
    all(d -> 1 / 4 < (ext[d][1] + ext[d][2]) / 2 < 3 / 4, 1:D) ? Refine : Keep
end

@testset "The modal indicator vanishes below its degree and falls as h^mode" begin
    # Guards the indicator's table and indexing: a dual of the wrong mode,
    # a line read across the wrong dimension, or owned nodes offset by the
    # wrong ghost width would score data the element represents exactly,
    # or fall at the wrong rate. One mode's amplitude must vanish on data
    # of degree below it, on a refined forest and with ghosts in the set,
    # and not on data of its degree; the indicator, which reads every
    # mode up to `mode`, must vanish on constants only. Both must score
    # smooth data as `hₑ^mode`, and there the indicator is the top mode's
    # amplitude, since the weights make every mode's estimate of it exact
    # on a single Fourier mode. Measured: 1D rates 2.00-4.00 for modes 2-4
    # from roots = 4 to 32, 2D 1.95-3.96 from 2 to 8; the indicator equals
    # the amplitude to 1e-14 there.
    for D in (1, 2), P in (3, 4), mode in (P - 1, P)
        el = Element(Float64, P)
        forest = wave_forest(Float64, Val(D), P, 2; roots=4, refined=true)
        for G in (0, 1)
            fs = FieldSet(forest, 2; G=G, centering=elementcentered(D), P=P)
            fill_at_nodes!(poly_data(D, mode - 1), fs, el)
            @test maximum(mode_amplitude(fs, el; mode=mode)) < 1e-13
            @test maximum(modal_indicator(fs, el; mode=mode)) > 1e-9
            # `(hₑ/2)^mode` small on the fine blocks, 5e-6 at P = 4, and
            # far above rounding.
            fill_at_nodes!(poly_data(D, mode), fs, el)
            @test maximum(mode_amplitude(fs, el; mode=mode)) > 1e-9
            fill_at_nodes!((x, v) -> 2.5 + v, fs, el)
            @test maximum(modal_indicator(fs, el; mode=mode)) < 1e-13
        end

        hs, τs, as = Float64[], Float64[], Float64[]
        for roots in (D == 1 ? (4, 8, 16, 32) : (2, 4, 8))
            forest = wave_forest(Float64, Val(D), P, 2; roots=roots)
            fs = FieldSet(forest, 2; G=0, centering=elementcentered(D), P=P)
            fill_at_nodes!(smooth_data(D), fs, el)
            τ = modal_indicator(fs, el; mode=mode)
            @test size(τ) == (ntuple(_ -> 2, D)..., roots^D)
            push!(hs, minimum_spacing(forest))
            push!(τs, maximum(τ))
            push!(as, maximum(mode_amplitude(fs, el; mode=mode)))
        end
        @test abs(convergence_rate(hs, τs) - mode) < 0.06
        @test abs(convergence_rate(hs, as) - mode) < 0.06
        @test τs ≈ as rtol = 1e-12
    end

    # The top mode's dual is the truncation's, bit for bit, so the
    # indicator of a truncated `v` at mode P is what truncation left.
    el = Element(Float64, 3)
    @test el.duals[end] === el.topdual
    # Relative to the global peak: scaling the data changes nothing, and
    # a field that vanishes scores zero rather than NaN.
    forest = wave_forest(Float64, Val(2), 3, 2; roots=2)
    fs = FieldSet(forest, 2; G=0, centering=elementcentered(2), P=3)
    fill_at_nodes!(smooth_data(2), fs, el)
    τ = modal_indicator(fs, el; mode=2)
    fs.work .*= 1000
    @test modal_indicator(fs, el; mode=2) ≈ τ rtol = 1e-14
    # The indicator is the largest weighted estimate over the modes.
    a = [mode_amplitude(fs, el; mode=m) for m in 1:3]
    @test modal_indicator(fs, el; mode=3) ≈ max.(a[3], (3a[2]) .^ 1.5 ./ 15,
                                                 a[1] .^ 3 ./ 15) rtol = 1e-13
    fs.work .= 0
    @test all(iszero, modal_indicator(fs, el; mode=2))
end

@testset "The indicator does not pass an under-resolved Gaussian as resolved" begin
    # Guards the indicator's lower modes. On an element wider than the
    # feature the Legendre coefficients no longer fall with the degree,
    # and one of them can cross zero: the blast wave's Gaussian, centred on
    # an element corner, scores 2.8e-2 in mode 2 at roots = 4 (hₑ = 1.56σ)
    # and 5.7e-2 at roots = 8, so a threshold between the two passed the
    # root mesh as resolved and the run's error reached 7.4. The indicator
    # must score every mesh whose elements are wider than σ above every
    # mesh whose elements are narrower, where the single mode does not, and
    # fall with the spacing from there on; and the initial adaptation must
    # bring the centre to the level the threshold asks for. Measured at
    # P = 3: 2.17e-1, 9.58e-2, 5.74e-2, 2.21e-2, 6.14e-3 at roots = 2 to
    # 32; at P = 4: 6.22e-2, 7.70e-2, 1.30e-2, 1.87e-3, 2.40e-4. From
    # roots = 4 at refine_tol = 3e-2, hₑ = 0.39σ at the centre, where the
    # single mode stops at 1.56σ.
    σ = 0.08
    for P in (3, 4)
        el = Element(Float64, P)
        τs, as = Float64[], Float64[]
        coarse = 1:2                               # roots = 2 and 4: hₑ > σ
        for roots in (2, 4, 8, 16, 32)
            forest = wave_forest(Float64, Val(2), P, 2; roots=roots)
            fs = FieldSet(forest, 2; G=0, centering=elementcentered(2), P=P)
            fill_at_nodes!(blast_initial(1.0, (0.5, 0.5), σ), fs, el)
            push!(τs, maximum(modal_indicator(fs, el; mode=P - 1)))
            push!(as, maximum(mode_amplitude(fs, el; mode=P - 1)))
        end
        @test minimum(τs[coarse]) > maximum(τs[3:end])
        @test issorted(τs[2:end]; rev=true)
        P == 3 && @test minimum(as[coarse]) < maximum(as[3:end])
        @test τs[end] == as[end]
        P == 3 && @test τs ≈ [2.17e-1, 9.58e-2, 5.74e-2, 2.21e-2, 6.14e-3] rtol = 5e-3
    end

    # The element width at the centre after the initial adaptation, in σ.
    function adapted(indicator)
        P, roots = 3, 4
        el = Element(Float64, P)
        forest = wave_forest(Float64, Val(2), P, 2; roots=roots)
        fs = TWA.wave_fieldset(Float64, forest, 2, (0, 0), TWA.state_statesparity(Val(2)),
                               P, TWA.CPU())
        # The indicator outside the callback, which TreeAMR runs
        # concurrently, as `refine_flags` does.
        function flags(fs)
            τ = indicator(fs, el; mode=P - 1)
            return flag_blocks(fs.forest) do b, k
                refine_mark(τ, el, b, k; refine_tol=3e-2, coarsen_tol=3e-5,
                            maxlevel_cap=3)
            end
        end
        adapt_to_initial_data!(fs, Operators(family=ElementLocal);
                               initial=blast_initial(1.0, (0.5, 0.5), σ), flags=flags,
                               buffer=2, maxpasses=8)
        centre = filter(forest.leaves) do k
            all(d -> block_extent(forest, k)[d][1] <= 0.5 <= block_extent(forest, k)[d][2], 1:2)
        end
        return 1 / (roots * 2^maximum(level, centre) * 2) / σ
    end
    @test adapted(modal_indicator) ≈ 0.39 atol = 0.01
    @test adapted(mode_amplitude) ≈ 1.56 atol = 0.01
end

@testset "The block verdict is TreeWave's, with the box in nodes" begin
    # Guards the reduction from elements to a block: the box must cover
    # the nodes of the elements above `coarsen_tol`, in owned indices,
    # and the four cases must be TreeWave's, so that a block at the cap
    # still asks for its own level around the feature and a quiet block
    # recruits nobody.
    el = Element(Float64, 3)                       # P + 1 = 4 nodes
    k0 = MortonKey(1, 0, (0, 0))
    k1 = childkeys(k0)[1]
    τ = zeros(2, 2, 2)
    τ[2, 1, 1] = 0.5
    τ[1, 2, 2] = 0.005
    tols = (refine_tol=0.01, coarsen_tol=0.001)
    @test refine_mark(τ, el, 1, k0; tols..., maxlevel_cap=1) == (Refine, (5:8, 1:4))
    @test refine_mark(τ, el, 1, k1; tols..., maxlevel_cap=1) == (Keep, (5:8, 1:4))
    @test refine_mark(τ, el, 2, k1; tols..., maxlevel_cap=2) == (Keep, (1:4, 5:8))
    τ[2, 2, 2] = 0.002
    @test refine_mark(τ, el, 2, k1; tols..., maxlevel_cap=2) == (Keep, (1:8, 5:8))
    τ[:, :, 2] .= 0.0005
    @test refine_mark(τ, el, 2, k1; tols..., maxlevel_cap=2) == Coarsen
    @test refine_mark(τ, el, 2, k0; tols..., maxlevel_cap=2) == Keep
    @test_throws "dead band" refine_mark(τ, el, 1, k0; refine_tol=0.01,
                                         coarsen_tol=0.01, maxlevel_cap=1)

    # The flag vector is the verdict of each block, whatever the thread
    # count, and the buffer is TreeWave's.
    forest = wave_forest(Float64, Val(2), 3, 2; roots=4)
    fs = FieldSet(forest, 2; G=0, centering=elementcentered(2), P=3)
    fill_at_nodes!(blast_initial(1.0, (0.5, 0.5), 0.08), fs, el)
    flags = refine_flags(fs, el; mode=2, tols..., maxlevel_cap=1)
    τ = modal_indicator(fs, el; mode=2)
    @test flags == [refine_mark(τ, el, b, blockkey(fs, b); tols..., maxlevel_cap=1)
                    for b in 1:nblocks(fs)]
    @test count(f -> f isa Tuple && f[1] == Refine, flags) == 4
    @test refinement_buffer(forest, 2, 0.05) == 8
    @test_throws "regrid more often" refinement_buffer(forest, 2, 0.1)
    @test_throws "Mode 0 is the element's mean" modal_indicator(fs, el; mode=0)
    @test_throws "not a Legendre mode" modal_indicator(fs, el; mode=4)
    @test_throws "P = 1 has no mode" adaptive_blast(; P=1, E=2, roots=4,
                                                    maxlevel_cap=1, refine_tol=0.01,
                                                    coarsen_tol=0.001, chunk=0.05)
end

@testset "A regrid conserves Σwu, and with the truncation after it keeps v truncated" begin
    # Guards the regrid of a state: TreeAMR refines an element set by the
    # coarse polynomial and coarsens it by the exact L² projection, both
    # of which keep the Gauss–Lobatto integral of any data, and neither
    # needs ghosts. Refining keeps `v` in `Q_{P-1}`; coarsening projects
    # two fine polynomials onto one of degree P and brings `v`'s top mode
    # back, which is why the driver truncates after every regrid. The
    # truncation keeps the integral too, since `L_P` has none.
    for (D, P) in ((1, 3), (2, 3), (3, 2))
        el = Element(Float64, P)
        forest = wave_forest(Float64, Val(D), P, 2; roots=4, refined=true)
        p = WaveProblem(forest, el)
        u = scrambled_state(p)
        scatter!(p.fs, u)
        mass = (total_mass(p.fs, 1), total_mass(p.fs, 2))
        conserved() = all(v -> isapprox(total_mass(p.fs, v), mass[v]; rtol=1e-13), 1:2)

        flags = flag_blocks(forest) do b, k
            level(k) > 0 ? Coarsen : Keep
        end
        @test regrid!(forest, p.fs => p.schedule; flags=flags)
        @test maxlevel(forest) == 0
        @test conserved()
        @test maximum(mode_amplitude(p.fs, el; mode=P, var=2)) > 1e-3

        p = WaveProblem(p.fs, el)
        u = statevector(p.fs)
        gather!(u, p.fs)
        truncate_state!(u, p)
        scatter!(p.fs, u)
        @test maximum(mode_amplitude(p.fs, el; mode=P, var=2)) < 1e-13
        @test conserved()

        @test regrid!(forest, p.fs => p.schedule; flags=middle_flags(forest))
        @test maxlevel(forest) == 1
        @test conserved()
        @test maximum(mode_amplitude(p.fs, el; mode=P, var=2)) < 1e-13
    end
end

@testset "A problem rebuilt from the regridded state set is a fresh problem on its mesh" begin
    # Guards `WaveProblem(fs, el)`: after `regrid!` it must rebuild every
    # set, schedule and spacing the right-hand side reads for the new
    # mesh, and keep the parities of a reflecting box. A stale schedule
    # or spacing gives a different right-hand side, or none.
    for reflecting in (false, true)
        el = Element(Float64, 3)
        forest = wave_forest(Float64, Val(2), 3, 2; roots=4, reflecting=reflecting)
        p = WaveProblem(forest, el)
        scatter!(p.fs, scrambled_state(p))
        @test regrid!(forest, p.fs => p.schedule; flags=middle_flags(forest))
        rebuilt = WaveProblem(p.fs, el)
        fresh = WaveProblem(forest, el)
        @test rebuilt.fs === p.fs
        u = statevector(rebuilt.fs)
        gather!(u, rebuilt.fs)
        du1, du2 = similar(u), similar(u)
        wave_rhs!(du1, u, rebuilt, 0.0)
        wave_rhs!(du2, u, fresh, 0.0)
        @test du1 == du2
        @test wave_energy(rebuilt, u) == wave_energy(fresh, u)
    end

    el = Element(Float64, 3)
    forest = wave_forest(Float64, Val(2), 3, 2; roots=2)
    other = FieldSet(forest, 2; G=0, centering=elementcentered(2), P=1)
    @test_throws "the element is of order P = 3" WaveProblem(other, el)
    single = FieldSet(forest, 1; G=0, centering=elementcentered(2), P=3)
    @test_throws "nvars = 2" WaveProblem(single, el)
end

# One adaptive run shared by the next three testsets, with the
# recommended thresholds ("Adaptive refinement" in `CODE.md`): P = 3 from
# 4 × 4 root blocks up to level 2, to t = 0.2 in chunks of 0.05, three
# regrids. About half a second after compilation.
const ADAPTIVE = (P=3, E=2, roots=4, maxlevel_cap=2, refine_tol=1e-2, coarsen_tol=1e-5,
                  chunk=0.05, t_end=0.2)
const adaptive_run = adaptive_blast(; ADAPTIVE...)

@testset "The adaptive blast refines the ring up to the cap" begin
    # Guards the driver's cycle: an indicator that does not move with the
    # wave (`u`'s top mode, which truncation keeps constant in time),
    # flags taken before the scatter, or a regrid whose result is not
    # stepped would leave the ring on coarse blocks. Every block the ring
    # crosses must be at the cap at the end, and blocks far outside it
    # must not be.
    r = adaptive_run
    σ, t = 0.08, ADAPTIVE.t_end
    @test r.converged
    @test r.maxlevel == ADAPTIVE.maxlevel_cap
    @test r.nregrids >= 2
    distance(x) = hypot(x[1] - 0.5, x[2] - 0.5)
    ring = filter(r.forest.leaves) do k
        ext = block_extent(r.forest, k)
        corners = [distance((a, b)) for a in ext[1], b in ext[2]]
        minimum(corners) <= t + σ && maximum(corners) >= t - σ
    end
    @test !isempty(ring)
    @test all(k -> level(k) == ADAPTIVE.maxlevel_cap, ring)
    coarse = filter(k -> level(k) < ADAPTIVE.maxlevel_cap, r.forest.leaves)
    @test !isempty(coarse)
    @test all(coarse) do k
        ext = block_extent(r.forest, k)
        distance(((ext[1][1] + ext[1][2]) / 2, (ext[2][1] + ext[2][2]) / 2)) > t + 2σ
    end
end

@testset "The adaptive blast's energy changes only at regrids" begin
    # Guards the chunking: each chunk is the semi-discrete system on one
    # mesh, whose energy is an exact invariant, so within a chunk only
    # RK4 loses any (measured: 7.1e-10 of it per chunk), and a regrid
    # that leaves the mesh alone must leave the state alone. A regrid that
    # changes the mesh changes the operator, and the energy with it, by
    # how far the data jump across the element faces whose width it
    # changes (measured: at most 2.7e-11 of it; 1e-5 and more where the
    # mesh refines evolved data, which costs accuracy).
    r = adaptive_run
    for c in r.chunks
        @test 0 <= (c.energy_start - c.energy_end) / c.energy_start < 2e-9
    end
    for (a, b) in zip(r.chunks[1:(end - 1)], r.chunks[2:end])
        if a.regridded
            @test b.energy_start != a.energy_end
            @test abs(b.energy_start - a.energy_end) < 1e-10 * r.energy0
        else
            @test b.energy_start == a.energy_end
        end
    end
    @test count(c -> c.regridded, r.chunks) == r.nregrids
    @test !r.chunks[end].regridded
    @test r.chunks[end].t == ADAPTIVE.t_end
end

@testset "The adaptive blast matches the uniform fine run at less work" begin
    # Guards the whole cycle end to end: a lossy regrid, a missed
    # truncation, refinement that trails the ring, or a mesh that refines
    # data already evolved on a coarser one shows as an error above the
    # uniform run's at the cap's spacing. At the recommended thresholds the
    # factor is at most 1.003 in L∞ and 1.002 in L2 for P = 3 and 4, caps
    # 2 and 3, to t = 0.2 and 0.4 (measured here: L∞ 1.000 and 1.003, L2
    # 1.000 and 1.000, at 79 and 62 per cent of the node-steps, at caps 2
    # and 3).
    r = adaptive_run
    u = uniform_blast(; P=ADAPTIVE.P, E=ADAPTIVE.E, t_end=ADAPTIVE.t_end,
                      roots=ADAPTIVE.roots * 2^ADAPTIVE.maxlevel_cap)
    @test r.err < 1.01 * u.err
    @test r.l2 < 1.01 * u.l2
    @test r.node_steps < 0.82 * u.nodes * u.nsteps
    @test r.h == u.h

    # A level deeper, in chunks short enough for the buffer to cover them.
    deep = (; ADAPTIVE..., maxlevel_cap=3, chunk=0.025)
    r = adaptive_blast(; deep...)
    u = uniform_blast(; P=deep.P, E=deep.E, t_end=deep.t_end, roots=deep.roots * 2^3)
    @test r.maxlevel == 3
    @test r.err < 1.01 * u.err
    @test r.l2 < 1.01 * u.l2
    @test r.node_steps < 0.65 * u.nodes * u.nsteps

    # The observer sees the snapshots `uniform_blast`'s does: the first at
    # t = 0 on the initial hierarchy, the last at t_end, each with the
    # exact state on the mesh of the moment.
    seen = []
    observer(p, t, s, se) = push!(seen, (t, length(s), length(se), statelength(p.fs)))
    adaptive_blast(; ADAPTIVE..., t_end=0.1, observer=observer, nsnapshots=5)
    @test first.(seen) ≈ [0.0, 0.025, 0.05, 0.075, 0.1] atol = 2e-3
    @test seen[end][1] == 0.1
    @test all(s -> s[2] == s[3] == s[4], seen)
end

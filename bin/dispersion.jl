# The dispersion relation of OLDG, INSPIRAL-PLAN.md's M8d: why error
# outruns light. On a periodic brick of elements of width `hₑ = 1`, the
# semi-discrete wave operator `A` (`wave_rhs!`, then `truncate_state!`
# when it truncates, applied to a state already truncated) is assembled
# column by column from the unit vectors of one element's nodes, and cut
# into the blocks `A_j` by which element `e + j` hears element `e`. The
# Bloch matrix
#
#     A(θ) = Σ_j A_j exp(-i θ·j)
#
# has eigenvalues `±iω_b(θ)`, one pair per branch `b`; the group velocity
# of branch `b` is `dω_b/dθ` along the line in `θ` (with `hₑ = 1`, `θ` is
# `k hₑ`). Branches are sorted by `ω` at each `θ` and differentiated
# centrally; where two cross, the sorted branches swap slopes, which
# leaves the largest speed unchanged.
#
#     julia --project=. bin/dispersion.jl [--P=3] [--D=1] [--dir=x|diag] \
#         [--truncate=both|yes|no] [--n=2001] [--rows]
#
# `--D=3` assembles the three-dimensional operator (`--dir` the line in
# `θ`: along `x`, or along the diagonal, where `θ` is `k hₑ/√3` in each
# dimension and the speed is `dω/dk`). On a brick the operator is the sum
# of the dimensions' one-dimensional Laplacians, so `ω² = Σ_d μ_d(θ_d)`
# with `μ_d` the one-dimensional branches' `ω²`, and no direction moves
# faster than the fastest one-dimensional branch; `--D=3` checks this
# against the one-dimensional relation at a few `θ` (`separable`).
#
# Prints, for each truncation, the largest group velocity and where it is
# (`vg_max`, `theta_at`, `omega_at`, `branch_at`), the largest `|Re λ|`
# (`decay_max`, zero for a scheme that keeps the energy), the largest `ω`
# (`omega_max`, the spectral radius that sets the time step), and the
# physical branch's phase and group velocity at a few points per
# wavelength (`phase_ppw<n>`, `group_ppw<n>`, `n` nodes per wavelength,
# `P + 1` per element). With `--rows`, every branch at every `θ`:
#
#     truncate theta branch omega vg
#
# "Why the error outruns light" in `CODE.md` has the numbers.

using TreeWaveOLDG
using TreeWaveOLDG: TreeAMR, statearray, statelength
using LinearAlgebra: eigvals
using Printf

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

# The blocks `A_j` of the operator on a periodic brick of `ne` elements of
# width 1 per dimension, as a `Dict` from the offset `j` to the matrix by
# which element `e + j`'s nodes hear element `e`'s, both in the order
# `(local node, variable)`.
function operator_blocks(P, D, truncate; ne=6)
    E = 2
    roots = ne ÷ E
    forest = wave_forest(Float64, Val(D), P, E; roots=roots, L=Float64(ne))
    el = Element(Float64, P)
    wave = WaveProblem(forest, el; truncate=truncate)
    fs = wave.fs
    N = forest.N
    P1 = P + 1
    n = statelength(fs)
    shape = size(statearray(zeros(n), fs))
    # Each state entry's element, local node and variable.
    elem = Vector{NTuple{D,Int}}(undef, n)
    loc = Vector{Int}(undef, n)
    for (i, I) in enumerate(CartesianIndices(shape))
        b, var = I[D + 2], I[D + 1]
        J = ntuple(d -> I[d], D)
        q = ntuple(d -> (J[d] - 1) ÷ P1, D)
        first = ntuple(d -> q[d] * P1 + 1, D)
        x = node_coordinates(fs, el, b, first)
        elem[i] = ntuple(d -> mod(round(Int, x[d]), ne), D)
        l = ntuple(d -> (J[d] - 1) % P1, D)
        loc[i] = 1 + sum(l[d] * P1^(d - 1) for d in 1:D) + (var - 1) * P1^D
    end
    e0 = ntuple(_ -> 0, D)
    cols = findall(==(e0), elem)
    m = 2 * P1^D
    @assert length(cols) == m
    blocks = Dict{NTuple{D,Int},Matrix{Float64}}()
    u, du = zeros(n), zeros(n)
    wrap(j) = mod(j + ne ÷ 2, ne) - ne ÷ 2
    for c in cols
        fill!(u, 0)
        u[c] = 1
        truncate_state!(u, wave)
        wave_rhs!(du, u, wave, 0.0)
        truncate_state!(du, wave)
        for r in 1:n
            iszero(du[r]) && continue
            j = map(wrap, elem[r])
            @assert all(abs.(j) .< ne ÷ 2) "the operator reaches $j, past the brick"
            B = get!(() -> zeros(m, m), blocks, j)
            B[loc[r], loc[c]] += du[r]
        end
    end
    return blocks
end

bloch(blocks, θ) = sum(B * cis(-sum(θ .* j)) for (j, B) in blocks)

# The branches' `ω ≥ 0` at `θ`, sorted, and the largest `|Re λ|`.
function branches(blocks, θ)
    λ = eigvals(bloch(blocks, θ))
    ω = sort(filter(>=(-1e-12), imag.(λ)))
    return max.(ω, 0.0), maximum(abs, real.(λ))
end

function scan(P, D, truncate, dir, nθ)
    blocks = operator_blocks(P, D, truncate)
    unit = D == 1 ? (1.0,) : dir == "x" ? (1.0, 0.0, 0.0) : ntuple(_ -> 1 / sqrt(3.0), 3)
    # `κ = k hₑ` along `unit`, over one period of the Bloch matrix along
    # that line, `2π` per dimension's `θ`.
    κmax = 2π / maximum(unit)
    κs = range(0, κmax; length=nθ)
    Δ = step(κs) / 100
    rows = []
    decay = 0.0
    for κ in κs
        ω, dec = branches(blocks, κ .* unit)
        ω₊, _ = branches(blocks, (κ + Δ) .* unit)
        ω₋, _ = branches(blocks, (κ - Δ) .* unit)
        decay = max(decay, dec)
        nb = min(length(ω), length(ω₊), length(ω₋))
        for b in 1:nb
            push!(rows, (κ, b, ω[b], (ω₊[b] - ω₋[b]) / 2Δ))
        end
    end
    return blocks, rows, decay
end

# The physical branch at `κ`: the `ω` closest to `κ`, and its slope.
function physical(blocks, unit, κ)
    ω, _ = branches(blocks, κ .* unit)
    Δ = 1e-5
    ω₊, _ = branches(blocks, (κ + Δ) .* unit)
    ω₋, _ = branches(blocks, (κ - Δ) .* unit)
    b = argmin(abs.(ω .- κ))
    return ω[b], (ω₊[b] - ω₋[b]) / 2Δ
end

# The nodes' positions in an element of width 1, from its left face.
function local_nodes(P)
    forest = wave_forest(Float64, Val(1), P, 2; roots=1, L=2.0)
    el = Element(Float64, P)
    fs = WaveProblem(forest, el).fs
    return [node_coordinates(fs, el, 1, (l,))[1] for l in 1:(P + 1)]
end

# `--front`: how far ahead of the light cone the semi-discrete solution
# reaches. On a periodic line of `ne` elements, `u` starts as the compact
# bump `(1 - (x/a)²)⁵` of half-width `a` about the origin, `v` as zero, so
# that in the continuum it splits into two halves, each moving at the
# speed of light, and is zero beyond `|x| = a + t`. The semi-discrete
# solution `exp(tA)` is taken exactly, Bloch wave by Bloch wave (the time
# step's own error is left out). For each threshold `ε` and time `t`, the
# farthest node at which `|u|` exceeds `ε` times the bump's height, as a
# speed: `(x_ε - a)/t`, and `x_ε - a - t`, how far ahead of light it is.
function front(P, truncate, a, ts, εs; ne=1024)
    blocks = operator_blocks(P, 1, truncate)
    xs = local_nodes(P)
    P1 = P + 1
    m = 2P1
    es = 0:(ne - 1)
    pos(e, l) = (e <= ne ÷ 2 ? e : e - ne) + xs[l]
    bump(x) = abs(x) < a ? (1 - (x / a)^2)^5 : 0.0
    x0 = [ComplexF64(l <= P1 ? bump(pos(e, l)) : 0.0) for l in 1:m, e in es]
    θs = 2π .* es ./ ne
    phase = cis.(-θs * es')                   # (θ, e)
    ĉ = x0 * transpose(phase)                 # (local, θ)
    rows = []
    for t in ts
        x̂ = similar(ĉ)
        for (k, θ) in enumerate(θs)
            x̂[:, k] = exp(t * bloch(blocks, (θ,))) * ĉ[:, k]
        end
        xt = real.(x̂ * conj.(phase)) ./ ne     # (local, e)
        for ε in εs
            far = 0.0
            for e in es, l in 1:P1
                abs(xt[l, e + 1]) > ε && (far = max(far, abs(pos(e, l))))
            end
            push!(rows, (t, ε, (far - a) / t, far - a - t))
        end
    end
    return rows
end

function main()
    P = parse(Int, option("P", "3"))
    D = parse(Int, option("D", "1"))
    dir = option("dir", "x")
    tr = option("truncate", "both")
    nθ = parse(Int, option("n", "2001"))
    truncs = tr == "both" ? (true, false) : (tr == "yes",)
    unit = D == 1 ? (1.0,) : dir == "x" ? (1.0, 0.0, 0.0) : ntuple(_ -> 1 / sqrt(3.0), 3)
    if "--front" in ARGS
        a = parse(Float64, option("a", "4"))
        ts = [parse(Float64, x) for x in split(option("t", "5,10,20,40"), ",")]
        εs = [parse(Float64, x) for x in split(option("eps", "1e-4,1e-6,1e-8,1e-10"), ",")]
        println("# P = $P, the bump of half-width a = $a, hₑ = 1")
        println("truncate\tt\teps\tspeed\tahead")
        for truncate in truncs, (t, ε, speed, ahead) in front(P, truncate, a, ts, εs)
            @printf("%s\t%g\t%g\t%.4f\t%.3f\n", truncate, t, ε, speed, ahead)
        end
        return nothing
    end
    println("# P = $P, D = $D", D == 1 ? "" : ", along $dir", ", hₑ = 1")
    for truncate in truncs
        blocks, rows, decay = scan(P, D, truncate, dir, nθ)
        if "--rows" in ARGS
            for (κ, b, ω, vg) in rows
                @printf("%s\t%.6f\t%d\t%.10g\t%.10g\n", truncate, κ, b, ω, vg)
            end
        end
        k = argmax(r -> abs(r[4]), rows)
        @printf("truncate=%s\tvg_max\t%.6f\n", truncate, abs(k[4]))
        @printf("truncate=%s\ttheta_at\t%.6f\n", truncate, k[1])
        @printf("truncate=%s\tomega_at\t%.6f\n", truncate, k[3])
        @printf("truncate=%s\tbranch_at\t%d\n", truncate, k[2])
        @printf("truncate=%s\tomega_max\t%.6f\n", truncate, maximum(r -> r[3], rows))
        @printf("truncate=%s\tdecay_max\t%.3e\n", truncate, decay)
        for ppw in (6, 10, 12, 16, 32)
            # `ppw` nodes per wavelength, `P + 1` per element: `κ = 2π(P+1)/ppw`.
            κ = 2π * (P + 1) / ppw
            ω, vg = physical(blocks, unit, κ)
            @printf("truncate=%s\tphase_ppw%d\t%.6f\n", truncate, ppw, ω / κ)
            @printf("truncate=%s\tgroup_ppw%d\t%.6f\n", truncate, ppw, vg)
        end
        D == 3 && separable(P, blocks, truncate)
    end
    return nothing
end

# On a brick `ω² = Σ_d μ_d(θ_d)`: the three-dimensional branches at a few
# `θ` against the sums of the one-dimensional ones' `ω²`.
function separable(P, blocks3, truncate)
    blocks1 = operator_blocks(P, 1, truncate)
    worst = 0.0
    for θ in ((0.3, 0.0, 0.0), (0.7, 1.9, 0.0), (2.1, 0.4, 2.9), (3.1, 3.1, 3.1))
        ω3, _ = branches(blocks3, θ)
        μs = [branches(blocks1, (θ[d],))[1] .^ 2 for d in 1:3]
        sums = sort(vec([a + b + c for a in μs[1], b in μs[2], c in μs[3]]))
        ω3² = ω3 .^ 2
        # Every three-dimensional branch is a sum of one-dimensional ones.
        worst = max(worst, maximum(w -> minimum(abs.(sums .- w)), ω3²))
    end
    @printf("truncate=%s\tseparable\t%.3e\n", truncate, worst)
    return nothing
end

main()

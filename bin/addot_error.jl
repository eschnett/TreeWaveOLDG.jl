# The error of a dynamic binary's source estimated from its run's rows,
# without a reference run (`CODE.md`, "The breathing instability" under
# "Measured results"), in minutes on a few threads:
#
#     julia -t N,0 --project=. bin/addot_error.jl LOG [--every=0.25] \
#         [--from=T] [--to=T] [--nR=48] [--nRl=64] [--nth=12] [--nph=24]
#
# `LOG` is `bin/binary.jl`'s output. The source `S = □(WΨ̂)` is written
# with `𝒟`, the derivative along the jets' polynomial, and is the field's
# only where the jets the source takes are each other's rates. Where they
# are not, the total field `Φ_r + WΨ̂` obeys `□Φ = −ρ + σ` with
#
#     σ = W (∂ₜ² − 𝒟²) Ψ̂,
#
# and the force the particle reads is off by the force of `σ`'s retarded
# field, `F_σ = −Q (∇Φ_σ + v ∂ₜΦ_σ)`. With `--source-addot=predicted` the
# source takes the predictor's `ä` while the equation of motion's own and
# other fields read the solved one, which adds the force inconsistency
# `ΔF`: `own_force` and `other_force` with the solved `ä` less with the
# source's. Both come from the rows alone:
#
# - `∂ₜ²Ψ̂` along the actual jets, `Ψ̂` at `J + τJ̇ + τ²J̈/2`, with the
#   jets' rates by central differences across the rows (`ż = v`,
#   `z̈ = v̇ = a` exactly), against `𝒟²Ψ̂` along `trajectory_at(J, τ)`;
#   the two agree in `z` to second order, so the subtraction holds near
#   the particle;
# - `Φ_σ` at particle `k` by quadrature in spherical coordinates, Gauss in
#   `ln R` from `ℓ/100` to 0.5 (`nR` nodes) and in `R` beyond (`nRl` per
#   unit length, for σ's oscillation in the retarded time): each particle's part of `σ`, sharp near it, on a
#   grid centred on it, the other's handed over by a smooth partition
#   between 0.3 and 0.6 from it; `σ` at the retarded time `t − D` from the
#   rows interpolated linearly, `∂ₜσ` by differences across a row.
#
# It prints, every `every` from the first time the retarded times reach
# into the rows, per particle the run's remainder force along the
# velocity and the inward radius (`Ft`, `Fr`), `F_σ` and `ΔF` along the
# same, and `|δä|/|ä|`, the solved `ä` against the predictor's; then their
# means over the times printed. A log without the positions, velocities
# and the predictor's `ä` (`bin/binary.jl` before they were recorded)
# takes the positions from the orbit's `d`, `phase`, `ddot` and `omega`
# with the centre of mass at `o`, and the predictor's `ä` from
# `predicted_jets`, from `onset` on only.

using TreeWaveOLDG, StaticArrays, ForwardDiff, LinearAlgebra
using TreeWaveOLDG: own_force, other_force, SupportRule, gauss_nodes_weights,
                    trajectory_at, lorentz_factor

const OFFSET = SVector(1.23, 0.71, 0.93) * 1e-2   # bin/prescribed.jl's

function arg(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, '=', limit=2)[2]
    end
    return default
end

# The rows: time → quantity → value, and the tag's order.
function read_rows(path)
    rows = Dict{Float64,Dict{String,Float64}}()
    order = 3
    for line in eachline(path)
        f = split(line, '\t')
        length(f) == 9 || continue
        t = tryparse(Float64, f[7])
        v = tryparse(Float64, f[9])
        (t === nothing || v === nothing) && continue
        order = parse(Int, f[6])
        get!(rows, t, Dict{String,Float64}())[f[8]] = v
    end
    return rows, order
end

vec3(r, name) = SVector(r[name * "x"], r[name * "y"], r[name * "z"])

function addot_error()
    path = ARGS[findfirst(a -> !startswith(a, "--"), ARGS)]
    rows, order = read_rows(path)
    r0 = rows[0.0]
    q2, ℓ = r0["q2"], r0["ell"]
    R₁, R₂ = r0["windowR1"], r0["windowR2"]
    onset = r0["onset"]
    predicted_source = get(r0, "source_addot_predicted", 0.0) == 1
    o = SVector(OFFSET[1], OFFSET[2], r0["oz"])
    q = sqrt(q2)
    ms = (ParticleModel(; Q=q, ℓ, M=1.0), ParticleModel(; Q=-q, ℓ, M=1.0))
    f, _ = newtonian_binary(ms, 1.0, 0.2; centre=o)
    f.G ≈ r0["G"] || error("the log's G is not newtonian_binary's at v = 0.2")
    w = Window(o, R₁, R₂; continuity=4)
    rule = SupportRule(Float64)
    ts = sort!([t for (t, r) in rows if haskey(r, "a1x")])
    recorded = haskey(rows[ts[end]], "addp1x")
    # The jets at each row: position, velocity, the solved `a`, `ȧ`, `ä`,
    # the predictor's `ä` (blended), and the `a⃛`, `a⃜` the source took.
    n = length(ts)
    Z = [zeros(SVector{3,Float64}, n) for _ in 1:2]
    V, A, Ȧ, Ä, Äp, A3, A4 = ([zeros(SVector{3,Float64}, n) for _ in 1:2] for _ in 1:7)
    for (i, t) in enumerate(ts)
        r = rows[t]
        if recorded
            for k in 1:2
                Z[k][i], V[k][i] = vec3(r, "z$k"), vec3(r, "v$k")
            end
        else
            # The centre of mass at `o`, the masses equal.
            d, φ = r["d"], r["phase"]
            r̂ = SVector(cos(φ), sin(φ), 0.0)
            t̂ = SVector(-sin(φ), cos(φ), 0.0)
            rel = d * r̂
            wrel = r["ddot"] * r̂ + r["omega"] * d * t̂
            Z[1][i], Z[2][i] = o + rel / 2, o - rel / 2
            V[1][i], V[2][i] = wrel / 2, -wrel / 2
        end
        for k in 1:2
            A[k][i], Ȧ[k][i], Ä[k][i] = vec3(r, "a$k"), vec3(r, "ad$k"), vec3(r, "add$k")
            A3[k][i], A4[k][i] = vec3(r, "a3$k"), vec3(r, "a4$k")
        end
        if recorded
            for k in 1:2
                Äp[k][i] = vec3(r, "addp$k")
            end
        else
            pj = predicted_jets(ms, (NoForce(), NoForce()), (Z[1][i], Z[2][i]),
                                (V[1][i], V[2][i]); interaction=f)
            for k in 1:2
                Äp[k][i] = pj[k].ä
            end
        end
    end
    Äu = predicted_source ? Äp : Ä        # the `ä` the source took
    # Rates by central differences across the rows.
    function rates(X)
        D1, D2 = similar(X), similar(X)
        for i in 2:(n - 1)
            h₋, h₊ = ts[i] - ts[i - 1], ts[i + 1] - ts[i]
            D1[i] = (X[i + 1] - X[i - 1]) / (h₋ + h₊)
            D2[i] = 2 * ((X[i + 1] - X[i]) / h₊ - (X[i] - X[i - 1]) / h₋) / (h₋ + h₊)
        end
        D1[1], D2[1], D1[n], D2[n] = D1[2], D2[2], D1[n - 1], D2[n - 1]
        return D1, D2
    end
    dA = [rates(A[k]) for k in 1:2]
    dȦ = [rates(Ȧ[k]) for k in 1:2]
    dÄ = [rates(Äu[k]) for k in 1:2]
    # Linear interpolation of a row series at time `t`.
    function at(X, t)
        i = clamp(searchsortedlast(ts, t), 1, n - 1)
        θ = (t - ts[i]) / (ts[i + 1] - ts[i])
        return (1 - θ) * X[i] + θ * X[i + 1]
    end
    # σ/W of particle `b` at `x` and time `t`.
    function σb(b, x, t)
        z, v, a, ȧ, ä = at(Z[b], t), at(V[b], t), at(A[b], t), at(Ȧ[b], t), at(Äu[b], t)
        a3, a4 = at(A3[b], t), at(A4[b], t)
        a1, a2 = at(dA[b][1], t), at(dA[b][2], t)
        ȧ1, ȧ2 = at(dȦ[b][1], t), at(dȦ[b][2], t)
        ä1, ä2 = at(dÄ[b][1], t), at(dÄ[b][2], t)
        used = Trajectory(; z, v, a, ȧ, ä, a⃛=a3, a⃜=a4)
        function actual(τ)
            one_ = one(τ)
            return Trajectory(; z=z + v * τ + a * τ^2 / 2, v=v + a * τ + a1 * τ^2 / 2,
                              a=a + a1 * τ + a2 * τ^2 / 2, ȧ=ȧ + ȧ1 * τ + ȧ2 * τ^2 / 2,
                              ä=ä + ä1 * τ + ä2 * τ^2 / 2, a⃛=a3 * one_, a⃜=a4 * one_)
        end
        d2(g) = ForwardDiff.derivative(τ -> ForwardDiff.derivative(g, τ), 0.0)
        return d2(τ -> particle_field(ms[b], x, actual(τ); order)) -
               d2(τ -> particle_field(ms[b], x, trajectory_at(used, τ); order))
    end
    nR = parse(Int, arg("nR", "48"))
    nRl = parse(Float64, arg("nRl", "64"))
    nθ, nφ = parse(Int, arg("nth", "12")), parse(Int, arg("nph", "24"))
    gauss(m) = map(x -> Float64.(x), gauss_nodes_weights(m; precision=128))
    xr, wr = gauss(nR)
    xc, wc = gauss(nθ)
    h = ts[end] - ts[end - 1]
    # Each particle's part of σ, sharp near it, is integrated on a grid
    # centred on it; the other particle's part on the grid of the particle
    # whose force is wanted, except within `rcut` of the other, where a
    # smooth partition `χ` hands it to a grid centred there.
    rcut = (0.3, 0.6)
    χ(r) = r <= rcut[1] ? 0.0 : r >= rcut[2] ? 1.0 :
           smoothstep(4, (r - rcut[1]) / (rcut[2] - rcut[1]))
    # ∇Φ and ∂ₜΦ at `zk`, time `t`, of the part `part(x, t′)` of σ, over a
    # grid centred at `c` out to `Rmax`: ∇Φ = ∫ ê (σ/D² + σ̇/D) d³x/4π,
    # ∂ₜΦ = ∫ σ̇/D d³x/4π, `D = |x − zk|`, `ê = (x − zk)/D`, σ at `t − D`.
    # Shells: Gauss in `ln R` from `ℓ/100` to `Rs`, then Gauss in `R`, with
    # `nRl` nodes per unit length so that σ's oscillation in the retarded
    # time is resolved; each with its weight times `R²`.
    function shells(Rmax)
        Rs = min(Rmax, 0.5)
        ulo, uhi = log(ℓ / 100), log(Rs)
        Rw = [(exp(ulo + (xr[i] + 1) * (uhi - ulo) / 2),
               wr[i] * (uhi - ulo) / 2 * exp(ulo + (xr[i] + 1) * (uhi - ulo) / 2)^3)
              for i in 1:nR]
        if Rmax > Rs
            m = max(8, ceil(Int, nRl * (Rmax - Rs)))
            xg, wg = gauss(m)
            append!(Rw, ((Rs + (x + 1) * (Rmax - Rs) / 2,
                          w * (Rmax - Rs) / 2 * (Rs + (x + 1) * (Rmax - Rs) / 2)^2)
                         for (x, w) in zip(xg, wg)))
        end
        return Rw
    end
    function retarded(part, c, zk, t, Rmax)
        Rw = shells(Rmax)
        acc = zeros(SVector{4,Float64}, length(Rw))    # per shell, summed in order
        Threads.@threads for i in eachindex(Rw)
            R, wRi = Rw[i]
            s = zero(SVector{4,Float64})
            for j in 1:nθ, kφ in 1:nφ
                cθ = xc[j]
                sθ = sqrt(1 - cθ^2)
                φ = 2π * (kφ - 0.5) / nφ
                x = c + R * SVector(sθ * cos(φ), sθ * sin(φ), cθ)
                W = window_terms(w, x)[1]
                iszero(W) && continue
                D = norm(x - zk)
                ê = (x - zk) / D
                tr = t - D
                s0 = part(x, tr)
                ṡ = (part(x, tr + h) - part(x, tr - h)) / 2h
                wgt = W * wRi * wc[j] * 2π / nφ / 4π
                g = wgt * (s0 / D^2 + ṡ / D) * ê
                s += SVector(g[1], g[2], g[3], wgt * ṡ / D)
            end
            acc[i] = s
        end
        tot = sum(acc)
        return SVector(tot[1], tot[2], tot[3]), tot[4]
    end
    # The force of σ's retarded field on particle `k` at `t`.
    function Fσ(k, t)
        b = 3 - k
        zk, zb, v = at(Z[k], t), at(Z[b], t), at(V[k], t)
        near(x, tr) = (r = norm(x - zb); r >= rcut[2] ? 0.0 : (1 - χ(r)) * σb(b, x, tr))
        far(x, tr) = σb(k, x, tr) + χ(norm(x - zb)) * σb(b, x, tr)
        g1, p1 = retarded(far, zk, zk, t, norm(zk - o) + R₂)
        g2, p2 = retarded(near, zb, zk, t, rcut[2])
        return -ms[k].Q * (g1 + g2 + v * (p1 + p2))
    end
    # The force inconsistency: the equation's own and other fields with the
    # solved `ä` against the source's.
    function ΔF(k, t)
        predicted_source || return zero(SVector{3,Float64})
        b = 3 - k
        tr(j, ä) = Trajectory(; z=at(Z[j], t), v=at(V[j], t), a=at(A[j], t), ȧ=at(Ȧ[j], t),
                              ä, a⃛=at(A3[j], t), a⃜=at(A4[j], t))
        ks, ku = tr(k, at(Ä[k], t)), tr(k, at(Äp[k], t))
        bs, bu = tr(b, at(Ä[b], t)), tr(b, at(Äp[b], t))
        return own_force(ms[k], ks, rule; order)[1] - own_force(ms[k], ku, rule; order)[1] +
               other_force(ms[k], ks, ms[b], bs, rule; order)[1] -
               other_force(ms[k], ks, ms[b], bu, rule; order)[1]
    end
    every = parse(Float64, arg("every", "0.25"))
    Rreach = maximum(norm(Z[k][1] - o) for k in 1:2) + R₂ + 0.1
    first_t = recorded ? ts[1] + Rreach + 2h : max(ts[1], onset) + Rreach + 2h
    from = max(first_t, parse(Float64, arg("from", "0")))
    to = min(ts[end] - 2h, parse(Float64, arg("to", "1e9")))
    println("# $(basename(path)): q² = $q2, ℓ = $ℓ, order $order, window $(R₁)–$(R₂), ",
            "source ä ", predicted_source ? "predicted" : "solved",
            recorded ? "" : " (positions and predictor reconstructed)")
    println(join(("t", "k", "Ft", "Fr", "Fσ_t", "Fσ_r", "ΔF_t", "ΔF_r", "dä/ä"), '\t'))
    sums = zeros(2, 7)
    count = 0
    for t in from:every:to
        i = searchsortedfirst(ts, t)
        r = rows[ts[i]]
        for k in 1:2
            z, v = at(Z[k], t), at(V[k], t)
            ρ̂ = (o - z) / norm(o - z)
            t̂ = v / norm(v)
            Fs, Fd = Fσ(k, t), ΔF(k, t)
            dä = norm(at(Ä[k], t) - at(Äp[k], t)) / norm(at(Ä[k], t))
            vals = (r["Ft$k"], r["Fr$k"], dot(Fs, t̂), dot(Fs, ρ̂), dot(Fd, t̂), dot(Fd, ρ̂), dä)
            sums[k, :] .+= vals
            println(join((round(t, digits=4), k, (round(x, sigdigits=4) for x in vals)...),
                         '\t'))
        end
        count += 1
        flush(stdout)
    end
    for k in 1:2
        println("# mean, particle $k: ",
                join(("$n = $(round(x / count, sigdigits=4))"
                      for (n, x) in zip(("Ft", "Fr", "Fσ_t", "Fσ_r", "ΔF_t", "ΔF_r", "dä/ä"),
                                        sums[k, :])), ", "))
    end
    return nothing
end

addot_error()

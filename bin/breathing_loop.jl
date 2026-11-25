# The breathing instability of the dynamic binary without a grid
# (`CODE.md`, "The breathing instability" under "Measured results"), in
# seconds on a few threads:
#
#     julia -t N,0 --project=. bin/breathing_loop.jl [--q2=0.2] [--ell=0.04] \
#         [--window=1.25,2.5] [--order=3] [--v=0.2] [--mode=breathing] \
#         [--source-addot=solved] [--tau=1] [--split]
#
# `Dynamic` solves `a`, `ȧ` and `ä` at each stage and takes `a⃛` and `a⃜`
# from the predictor's point particles. The source `□(WΨ̂)` it writes is
# the field's only if the jets' actual rates are those its derivative
# `𝒟` along their polynomial reads; where the solved jets oscillate at a
# frequency the predictor does not know, the total field `Φ_r + WΨ̂`
# instead obeys
#
#     □Φ = −ρ + σ,    σ = (∂ₜ² − 𝒟²)(WΨ̂) = W (∂ₜ²ȧ − a⃛)·∂_ȧΨ̂ + W (∂ₜ²ä − a⃜)·∂_äΨ̂,
#
# with `□ = ∂ₜ² − ∇²`, the solved `a` and its first two rates taken as
# consistent. For a perturbation `δa e^{st}` of both particles, with the
# predictor's jets unmoved, `σ = W (s³ ∂_ȧΨ̂ + s⁴ ∂_äΨ̂)·δa`, leaving out
# the terms of `𝒟∂_äΨ̂`, smaller by about `v/sR`. With
# `--source-addot=predicted` (`Dynamic`'s `source_addot`) the source
# holds the predictor's `ä`, and the mismatch moves down a rate:
# `σ = W (s² ∂_aΨ̂ + s³ ∂_ȧΨ̂)·δa`. With `--source-addot=corrected` it
# holds the predictor's `ä` plus a running mean over `--tau` of the
# solved `ä`'s departure from it, `1/(1 + sτ)` of it at `s`, and since `σ`
# is linear in the source's `ä`, `G = G_pred + (G_solved − G_pred)/(1 + sτ)`.
# The breathing's `δa` lies along the separation, which the correction's
# basis of the predictor's `ä` and `ȧ` takes whole. Its retarded
# field `∫ σ e^{−sR}/4πR` exerts `F = −Q∇Φ` on particle 1, the sign
# `field_force` gives (checked below). The script integrates it in
# spherical coordinates about particle 1, over the window's ball, for the
# breathing mode, `δa₁ = n`, `δa₂ = −n` along the separation `n`, and
# solves `G(s) = Mγ`, the equation of motion `Mγ δa = F`, for the modes'
# `s = λ + iω`. Physical forces on the perturbation (the charges'
# attraction, Newton's) are of order `q²` and `G M/d³`, far below `s²`
# at `ω ≈ 8`, and left out.
#
# `--mode=com` takes the centre of mass's mode instead, `δa₂ = +n`, and
# `--mode=lone` particle 1's own term alone, a lone particle of the same
# charge at the same distance from the window's centre, as on
# `bin/dynamic.jl`'s spring circle of radius 0.5.
#
# The binary is `bin/binary.jl`'s: `M = 1`, `Q = ±q`, `d = 1`, each at
# `v`, about the origin, the window centred there. `--split` also prints
# the gain's parts: particle 1's own term and particle 2's, `ȧ`'s and
# `ä`'s, at `ω = 6` to 9, and where in `R` the `ä` term comes from.

using TreeWaveOLDG, StaticArrays, ForwardDiff, LinearAlgebra
using TreeWaveOLDG: field_force, SupportRule, gauss_nodes_weights

function arg(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, '=', limit=2)[2]
    end
    return default
end

function breathing_loop()
    q2 = parse(Float64, arg("q2", "0.2"))
    ℓ = parse(Float64, arg("ell", "0.04"))
    R₁, R₂ = parse.(Float64, split(arg("window", "1.25,2.5"), ','))
    order = parse(Int, arg("order", "3"))
    v = parse(Float64, arg("v", "0.2"))
    addot = arg("source-addot", "solved")
    addot in ("solved", "predicted", "corrected") ||
        error("--source-addot is solved, predicted or corrected")
    τ = parse(Float64, arg("tau", "1"))
    q = sqrt(q2)
    ms = (ParticleModel(; Q=q, ℓ, M=1.0), ParticleModel(; Q=-q, ℓ, M=1.0))
    o = SVector(0.0, 0.0, 0.0)
    _, past = newtonian_binary(ms, 1.0, v; centre=o)
    trs = past.path(0.0)
    w = Window(o, R₁, R₂; continuity=4)
    z1, z2 = trs[1].z, trs[2].z
    n = (z1 - z2) / norm(z1 - z2)
    F, _ = field_force(ms[1], trs[1], SupportRule(Float64), x -> (x[1], 0.0))
    println("force of g = x₁ on particle 1: ", round(F[1], sigdigits=4), " (−Q = ",
            round(-ms[1].Q, sigdigits=4), "), so F = −Q∇g")
    # ∂Ψ̂ₖ/∂a, ∂Ψ̂ₖ/∂ȧ and ∂Ψ̂ₖ/∂ä along the breathing's ±n
    function dpsi(k, x, jet)
        tr, sgn = trs[k], (k == 1 ? 1.0 : -1.0)
        g(ε) = particle_field(ms[k], x,
                              Trajectory(; z=tr.z, v=tr.v,
                                         a=jet === :a ? tr.a + ε * sgn * n : tr.a,
                                         ȧ=jet === :ȧ ? tr.ȧ + ε * sgn * n : tr.ȧ,
                                         ä=jet === :ä ? tr.ä + ε * sgn * n : tr.ä,
                                         a⃛=tr.a⃛, a⃜=tr.a⃜); order)
        return ForwardDiff.derivative(g, 0.0)
    end
    nR, nθ, nφ = 240, 40, 80
    Rmax = norm(z1 - o) + R₂
    gauss(m) = map(x -> Float64.(x), gauss_nodes_weights(m; precision=128))
    xr, wr = gauss(nR)
    Rs, wR = (xr .+ 1) .* Rmax ./ 2, wr .* Rmax ./ 2
    xc, wc = gauss(nθ)
    # B[i, :]: ∫dΩ (ê·n) W ∂Ψ̂ at R = Rs[i] about particle 1, for particle
    # 1's ȧ and ä, particle 2's ȧ and ä, and particle 1's and 2's a
    B = zeros(nR, 6)
    Threads.@threads for i in 1:nR
        for j in 1:nθ, kφ in 1:nφ
            c = xc[j]
            sθ = sqrt(1 - c^2)
            φ = 2π * (kφ - 0.5) / nφ
            e = SVector(sθ * cos(φ), sθ * sin(φ), c)
            x = z1 + Rs[i] * e
            W = window_terms(w, x)[1]
            iszero(W) && continue
            wgt = wc[j] * 2π / nφ * dot(e, n) * W
            B[i, 1] += wgt * dpsi(1, x, :ȧ)
            B[i, 2] += wgt * dpsi(1, x, :ä)
            B[i, 3] += wgt * dpsi(2, x, :ȧ)
            B[i, 4] += wgt * dpsi(2, x, :ä)
            B[i, 5] += wgt * dpsi(1, x, :a)
            B[i, 6] += wgt * dpsi(2, x, :a)
        end
    end
    # n·∇ of the kernel e^{−sR}/4πR at particle 1, per shell, times −Q₁
    K(s, i) = -ms[1].Q * wR[i] * (1 + s * Rs[i]) * exp(-s * Rs[i]) / 4π
    part(s, c, p) = sum(K(s, i) * s^p * B[i, c] for i in 1:nR)
    mode = arg("mode", "breathing")
    mode in ("breathing", "com", "lone") || error("--mode is breathing, com or lone")
    other = mode == "breathing" ? 1 : mode == "com" ? -1 : 0
    Gs(s) = part(s, 1, 3) + part(s, 2, 4) + other * (part(s, 3, 3) + part(s, 4, 4))
    Gp(s) = part(s, 5, 2) + part(s, 1, 3) + other * (part(s, 6, 2) + part(s, 3, 3))
    G(s) = addot == "solved" ? Gs(s) : addot == "predicted" ? Gp(s) :
           Gp(s) + (Gs(s) - Gp(s)) / (1 + s * τ)
    γ = 1 / sqrt(1 - v^2)
    println("q² = $q2, ℓ = $ℓ, window $(R₁)–$(R₂), order $order, the $mode mode, ",
            "the $addot ä in the source", addot == "corrected" ? " (τ = $τ)" : "",
            ", Mγ = ", round(γ, digits=4))
    println("the gain on the imaginary axis, |G(iω)| and its phase:")
    for ω in 1:16
        g = G(im * ω)
        println("  ω = $ω: ", round(abs(g), sigdigits=3), ", ",
                round(rad2deg(angle(g)), digits=0), "°")
    end
    if "--split" in ARGS
        println("the parts at iω (own ȧ, own ä, other's ȧ, other's ä), and the breathing and centre-of-mass modes' |G|:")
        for ω in (6.0, 7.0, 8.0, 9.0)
            p = [part(im * ω, c, c in (1, 3) ? 3 : 4) for c in 1:4]
            println("  ω = $ω: ", join((round(x, sigdigits=3) for x in p), ", "),
                    "; breathing ", round(abs(sum(p)), sigdigits=3), ", centre of mass ",
                    round(abs(p[1] + p[2] - p[3] - p[4]), sigdigits=3))
        end
        println("the ä term at ω = 8 by the distance R from particle 1:")
        for (lo, hi) in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, Rmax))
            x = sum(K(8im, i) * (8im)^4 * (B[i, 2] + B[i, 4]) for i in 1:nR if lo <= Rs[i] < hi)
            println("  R ∈ [$lo, $hi): ", round(abs(x), sigdigits=3))
        end
    end
    # The modes: G(s) = Mγ by Newton's method from a grid of first guesses.
    roots = ComplexF64[]
    for λ0 in 0.2:0.6:3.0, ω0 in 1.0:1.0:16.0
        s = complex(λ0, ω0)
        for _ in 1:60
            h = 1e-6
            s -= (G(s) - γ) / ((G(s + h) - G(s - h)) / 2h)
            abs(s) > 100 && break
        end
        abs(G(s) - γ) < 1e-8 && imag(s) > 0 && real(s) > -1 &&
            all(r -> abs(r - s) > 1e-4, roots) && push!(roots, s)
    end
    sort!(roots; by=real, rev=true)
    println("the modes G(s) = Mγ, fastest first (λ, ω):")
    isempty(roots) && println("  none with λ > −1")
    for r in roots[1:min(end, 8)]
        println("  ", round(real(r), digits=3), ", ", round(imag(r), digits=3))
    end
    return nothing
end

breathing_loop()

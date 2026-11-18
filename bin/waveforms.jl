# The waveforms of `bin/prescribed.jl --waves` against their reference, at
# several resolutions (INSPIRAL-PLAN.md's M8e and M8f; `CODE.md`,
# "Waveforms" under "Measured results"):
#
#     julia -t N,0 --project=. bin/waveforms.jl [--reference=file|kinked] \
#         [--cache=FILE] [--R1=100] [--R2=2.5] [--fast=1.58] [--window=5] \
#         [--clean=retarded|uniform] [--bands] [--bandwindow=2.5] \
#         [--Omega=0.4] [--rc=0.5] [--v=0.2] [--ntheta] HDIV=FILE …
#
# Each `HDIV=FILE` is one run's `--waves` file at `hₑ = 1/HDIV`; the
# orders are taken between consecutive resolutions, `HDIV` ascending. The
# files hold their spheres' radii and `lmax` in their header and must
# share their times.
#
# The reference is, by default, each file's own `ref_` columns, the
# retarded field of the run's path at the same points (`--data=retarded`
# or `uniform`). With `--reference=kinked` it is instead the retarded
# field of M8e's uniform start's path, the circle's tangent line before
# `t = 0` and the circle from then on, which is the exact solution outside
# the window from the uniform data (`--data=uniform`): the circle about
# `bin/prescribed.jl`'s offset from the origin, of radius `rc` at speed
# `v`, read at the files' points (Gauss–Legendre in `cos θ` at `--ntheta`
# points, `2lmax + 12` by default, as the driver's). It takes some
# minutes; `--cache=FILE` keeps it.
#
# Prints, with `r` the sphere's radius and `‖·‖` the `L²` norm on the
# unit sphere of the modes (`m` and `-m` of a real field):
#
# - the reference's `r|Φ_ℓℓ|` for `ℓ ≤ 3` at the last time;
# - per run, the largest `r‖Φ - ref‖` over all modes in windows of
#   `--window` in time, per radius, and the orders between the runs;
# - the clean window per radius, where nothing from the sphere at `R1`
#   or from outside the data's clean region has arrived: for
#   `--clean=retarded` (the default, but for `kinked`) `t < R1 - r`; for
#   `--clean=uniform` (the default for `kinked`) also after the window's
#   pulse, `r + R₂ + 2 ≤ t < (R1 - r)/fast`, `fast` the speed of the
#   scheme's fastest modes; its largest error per run and the orders;
# - with `--bands`, the same windows, `--bandwindow` wide, per band of
#   `ℓ = 0`, `1`, `2` and `3…lmax`: what arrives where, and ahead of
#   light;
# - the sphere's return, from `t = R1 - r` and one window later: the mean
#   and largest
#   `|δΦ₁₁|/|Φ₁₁|`, the largest `r|δΦ₁₀|`, and `δΦ₁₁`'s frequency;
# - per orbit of `2π/Ω`, the mean amplitude error `|q| - 1` and phase
#   `arg q` of `q = Φ_ℓℓ/ref_ℓℓ` for `ℓ = 1, 2, 3`.

using TreeWaveOLDG
using TreeWaveOLDG: SVector
using Printf, Serialization, Statistics

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

const OFFSET = SVector(1.23, 0.71, 0.93) * 1e-2   # bin/prescribed.jl's

# One `--waves` file: its times, radii, modes, and the modes `[mode,
# radius, time]` of the run and of its reference (`nothing` if it has
# none).
function load(file)
    lines = readlines(file)
    hdr = split(lines[1], '\t')
    A = Matrix{Float64}(undef, length(lines) - 1, length(hdr))
    for (i, l) in enumerate(lines[2:end])
        A[i, :] = parse.(Float64, split(l, '\t'))
    end
    cols = Dict(h => j for (j, h) in enumerate(hdr))
    radii = Float64[]
    lmax = 0
    for h in hdr[2:end]
        m = match(r"^r([0-9.]+)_l(\d+)m(\d+)_re$", h)
        m === nothing && continue
        r = parse(Float64, m[1])
        r in radii || push!(radii, r)
        lmax = max(lmax, parse(Int, m[2]))
    end
    modes = harmonic_modes(lmax)
    name(r) = isinteger(r) ? string(Int(r)) : string(r)
    function grab(prefix)
        haskey(cols, "$(prefix)r$(name(radii[1]))_l0m0_re") || return nothing
        M = Array{ComplexF64}(undef, length(modes), length(radii), size(A, 1))
        for (ri, r) in enumerate(radii), (mi, (l, m)) in enumerate(modes)
            base = "$(prefix)r$(name(r))_l$(l)m$(m)"
            M[mi, ri, :] = complex.(A[:, cols[base * "_re"]], A[:, cols[base * "_im"]])
        end
        return M
    end
    return (; ts=A[:, 1], radii, lmax, modes, sim=grab(""), ref=grab("ref_"))
end

# The tangent line before `t = 0`, the circle after: M8e's uniform start.
struct Kinked{P}
    circle::P
end

function (k::Kinked)(t)
    t >= 0 && return k.circle(t)
    tr = k.circle(zero(t))[1]
    return (Trajectory(; z=tr.z + t * tr.v, v=tr.v),)
end

# The modes of `path`'s retarded field on the files' spheres at `ts`.
function retarded_modes(path, run, ts)
    Q = sqrt(0.005)
    nθ = parse(Int, option("ntheta", string(2run.lmax + 12)))
    table, dirs = TreeWaveOLDG.sphere_rule(Float64, run.lmax, nθ)
    pts = [r * d for r in run.radii for d in dirs]
    K = Array{ComplexF64}(undef, length(run.modes), length(run.radii), length(ts))
    vals = Vector{Float64}(undef, length(pts))
    for (i, t) in enumerate(ts)
        Threads.@threads for j in eachindex(pts)
            vals[j] = retarded_field(path, 1, Q, pts[j], t)[1]
        end
        K[:, :, i] = table * reshape(complex.(vals), length(dirs), length(run.radii))
    end
    return K
end

function kinked_reference(run)
    cache = option("cache", "")
    if !isempty(cache) && isfile(cache)
        c = deserialize(cache)
        c.ts == run.ts && c.radii == run.radii && return c.K
        @warn "the cache $cache is of other times or radii; computing anew"
    end
    circle = circular_motion(OFFSET, parse(Float64, option("rc", "0.5")),
                             parse(Float64, option("v", "0.2"))).path
    # The circle's reference rebuilt here must be the file's, or the points
    # or the modes' order are not the driver's.
    if run.ref !== nothing
        i = length(run.ts) ÷ 2
        C = retarded_modes(circle, run, run.ts[i:i])
        @printf("the circle's reference rebuilt at t = %g: largest difference %.1e of %.1e\n",
                run.ts[i], maximum(abs, C[:, :, 1] - run.ref[:, :, i]),
                maximum(abs, run.ref[:, :, i]))
    end
    K = retarded_modes(Kinked(circle), run, run.ts)
    isempty(cache) || serialize(cache, (; run.ts, run.radii, K))
    return K
end

# `r` times the norm of the modes `M[:, ri, it]` with `ℓ ∈ ls`.
function rnorm(run, M, ri, it; ls=0:run.lmax)
    s = 0.0
    for (mi, (l, m)) in enumerate(run.modes)
        l in ls || continue
        s += (m == 0 ? 1 : 2) * abs2(M[mi, ri, it])
    end
    return run.radii[ri] * sqrt(s)
end

function wmax(f, ts, ri, a, b)
    return maximum((f(ri, it) for it in eachindex(ts) if a <= ts[it] < b + 1e-9);
                   init=0.0)
end

order(e, h) = [log(e[k] / e[k + 1]) / log(h[k + 1] / h[k]) for k in 1:(length(e) - 1)]

function header(radii, width; label="t")
    print(lpad(label, 11))
    for r in radii
        print(lpad(isinteger(r) ? string(Int(r)) : string(r), width))
    end
    println()
end

function windows_table(label, run, E, h, ws; ls=0:run.lmax)
    println("\n", label)
    header(run.radii, 10 * length(h) + 2)
    for (a, b) in ws
        @printf("%5.1f-%-5.1f", a, b)
        for ri in eachindex(run.radii)
            print("  ")
            for e in E
                @printf("%10.2e", wmax((r, i) -> rnorm(run, e, r, i; ls), run.ts, ri, a, b))
            end
        end
        println()
    end
    length(h) > 1 || return nothing
    println("\n", label, ": orders")
    header(run.radii, 6 * (length(h) - 1) + 2)
    for (a, b) in ws
        @printf("%5.1f-%-5.1f", a, b)
        for ri in eachindex(run.radii)
            print("  ")
            e = [wmax((r, i) -> rnorm(run, x, r, i; ls), run.ts, ri, a, b) for x in E]
            for o in order(e, h)
                @printf("%6.1f", o)
            end
        end
        println()
    end
    return nothing
end

function main()
    runs = Pair{Int,String}[]
    for a in ARGS
        startswith(a, "--") && continue
        k, f = split(a, "="; limit=2)
        push!(runs, parse(Int, k) => f)
    end
    isempty(runs) && error("give the runs as HDIV=FILE")
    sort!(runs; by=first)
    h = first.(runs)
    R = [load(f) for f in last.(runs)]
    run = R[1]
    all(r -> r.ts == run.ts && r.radii == run.radii && r.lmax == run.lmax, R) ||
        error("the files' times, radii or lmax differ")
    kind = option("reference", "file")
    kind in ("file", "kinked") || error("--reference is file or kinked, got $kind")
    refs = if kind == "kinked"
        K = kinked_reference(run)
        [K for _ in R]
    else
        any(r -> r.ref === nothing, R) && error("a file has no ref_ columns")
        [r.ref for r in R]
    end
    E = [r.sim .- ref for (r, ref) in zip(R, refs)]
    ts, radii = run.ts, run.radii
    R1 = parse(Float64, option("R1", "100"))
    R2 = parse(Float64, option("R2", "2.5"))
    fast = parse(Float64, option("fast", "1.58"))
    width = parse(Float64, option("window", "5"))
    clean = option("clean", kind == "kinked" ? "uniform" : "retarded")
    Ω = parse(Float64, option("Omega", "0.4"))
    println("runs: ", join(("d/$k $f" for (k, f) in runs), ", "), "; reference: $kind")

    ref = refs[end]
    println("\nthe reference's r|Φ_ℓℓ| at t = $(ts[end])")
    for (ri, r) in enumerate(radii)
        @printf("r = %5g:", r)
        for l in 0:min(3, run.lmax)
            @printf("  ℓ = %d %.3e", l, r * abs(ref[l * (l + 1) ÷ 2 + l + 1, ri, end]))
        end
        println()
    end

    ws = [(a, min(a + width, ts[end])) for a in 0:width:(ts[end] - width / 2)]
    windows_table("r‖Φ − ref‖ over all modes, largest per window, " *
                  join(("d/$k" for k in h), " | "), run, E, h, ws)

    println("\nthe clean window ($clean): largest r‖Φ − ref‖, " *
            join(("d/$k" for k in h), ", "), ", and the orders")
    for (ri, r) in enumerate(radii)
        a, b = clean == "retarded" ? (0.0, R1 - r) : (r + R2 + 2, (R1 - r) / fast)
        b = min(b, ts[end] + 1e-9)
        b > a || continue
        e = [maximum((rnorm(run, x, ri, i) for i in eachindex(ts) if a <= ts[i] < b);
                     init=0.0) for x in E]
        @printf("r = %5g, t ∈ [%5.1f, %5.1f):", r, a, b)
        for x in e
            @printf(" %.2e", x)
        end
        length(h) > 1 && print("  orders ", join((@sprintf("%.1f", o) for o in order(e, h)), " "))
        println()
    end

    if "--bands" in ARGS
        bw = parse(Float64, option("bandwindow", "2.5"))
        bws = [(a, min(a + bw, ts[end])) for a in 0:bw:(ts[end] - bw / 2)]
        for ls in (0:0, 1:1, 2:2, 3:run.lmax)
            windows_table("ℓ ∈ $ls: r‖Φ − ref‖, largest per window, " *
                          join(("d/$k" for k in h), " | "), run, E, h, bws; ls)
        end
    end

    println("\nthe sphere's return, from t = R1 − r + $width: mean and largest |δΦ₁₁|/|Φ₁₁|, " *
            "largest r|δΦ₁₀|, δΦ₁₁'s frequency")
    i11, i10 = 3, 2
    dt = ts[2] - ts[1]
    for (ri, r) in enumerate(radii), (k, e) in zip(h, E)
        its = [i for i in eachindex(ts) if ts[i] >= R1 - r + width]
        length(its) > 2 || continue
        q = [abs(e[i11, ri, i]) / abs(ref[i11, ri, i]) for i in its]
        ph = angle.(e[i11, ri, its])
        ω = mean(mod(ph[j + 1] - ph[j] + π, 2π) - π for j in 1:(length(ph) - 1)) / dt
        @printf("r = %5g, d/%d: %.2e %.2e  r|δΦ₁₀| %.2e  ω %.3f\n", r, k, mean(q),
                maximum(q), maximum(r * abs(e[i10, ri, i]) for i in its), ω)
    end

    period = 2π / Ω
    for l in 1:min(3, run.lmax)
        mi = l * (l + 1) ÷ 2 + l + 1
        println("\nmode ($l,$l) per orbit: mean |q| − 1 and arg q (rad), q = Φ/ref, " *
                join(("d/$k" for k in h), " | "))
        header(radii, 18 * length(h) + 2; label="orbit")
        for k in 0:(ceil(Int, ts[end] / period) - 1)
            a, b = k * period, min((k + 1) * period, ts[end] + 1e-9)
            its = [i for i in eachindex(ts) if a <= ts[i] < b]
            @printf("%11d", k + 1)
            for ri in eachindex(radii)
                print("  ")
                for (x, rf) in zip(R, refs)
                    q = [x.sim[mi, ri, i] / rf[mi, ri, i] for i in its]
                    @printf("%+.1e %+.1e ", mean(abs.(q)) - 1, mean(angle.(q)))
                end
            end
            println()
        end
    end
    return nothing
end

main()

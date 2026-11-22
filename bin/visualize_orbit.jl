#!/usr/bin/env julia
#
# The movie of two charges on a circle ("The movie" in `CODE.md`), from
# the frames of `bin/orbit_movie.jl` or `bin/orbit_exact.jl`, or of the
# binary that inspirals, from `bin/inspiral_movie.jl`'s:
#
#     julia --project=bin -t 4,0 bin/visualize_orbit.jl DIR [--out=FILE.mp4] \
#         [--frame=N] [--quantity=accel|rate|total] [--ref=DIR] [--error] \
#         [--waves=FILE] [--wradius=50] [--modes=2:2,4:4] [--rmax] \
#         [--clim] [--zclim] [--colormap=plasma] [--fps=30] [--every=1] \
#         [--wide=w96] [--zoom=w8] [--grid] [--track=T]
#
# Each frame shows, in the orbit's plane, `r` times a quantity, `r` the
# distance from the orbit's centre, so that the outgoing waves keep their
# brightness:
#
# - `accel` (the default): the total field less each charge's field of
#   uniform motion at its present position and velocity (the particle
#   field of order 0, `field_of_order(m, x, tr, 0)`): what is left is the
#   field the acceleration makes, the two-armed spiral, without the `1/r`
#   of the charges or their peaks;
# - `rate`: `Π = ∂ₜΦ`, which has no static part;
# - `total`: `Φ` itself.
#
# With `--ref=DIR`, the exact frames, and `--error`, the quantity less
# the reference's instead. The wide grid fills the left, masked to
# `r < rmax`; the zoomed grid, with the charges and their tracks over the
# last quarter orbit, the upper right; below it, with `--waves`, `r Re Φ_ℓm`
# on the sphere of `wradius` against `t`, solid from the run and dashed
# from the retarded field when the file has it, with the frame's time
# marked. For the inspiral (`dynamic` in `meta.tsv`), the tracks run from
# the start, the title counts the orbits by the phase of `z₁ − z₂`, and a
# panel at the bottom right shows the separation `d` against `t`, with
# the leading order's `d³ = d₀³ − (Δq²K/2π)(t − t₀)` from the end of the
# hand-over `t₀` dashed. `--track=T` sets how far back in time the tracks
# reach, by default a quarter orbit, or the whole run for the inspiral. The colour scales are fixed for the whole movie: symmetric, at
# `--clim` (`--zclim` for the zoom), by default the 99.5th percentile of
# `|q|` over the last frame, in `--colormap` (any of Makie's, by default
# `plasma`; `balance` is a diverging one). `--frame=N` writes one frame (from 0) as a
# PNG; otherwise the movie goes to `--out`, every `every`-th frame.
# `--grid` draws the elements' faces where the plane cuts them, in grey,
# on both maps: the forest is built again from the run's layout in
# `meta.tsv` (`bin/ball.jl`), and the edges of each element of the cube
# and of the four shells the plane crosses are mapped to it, curved where
# the shells are; a run's frames have the layout, the exact frames do not.
# Output goes to `bin/output/` unless `--out` names a path.

using CairoMakie
using Printf
using Statistics: quantile
using TreeWaveOLDG
using TreeWaveOLDG: SVector, Trajectory, field_of_order, plane_grid, TreeAMR

include(joinpath(@__DIR__, "plane_io.jl"))
include(joinpath(@__DIR__, "ball.jl"))

CairoMakie.activate!(; type="png", px_per_unit=1)

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

metanum(meta, key) = parse(Float64, meta[key][1])

# The grid's points' distances from `o` and their positions.
function geometry(g, o)
    xs = plane_grid(SVector(g.centre), g.halfwidth, g.n)
    return xs, [hypot(x[1] - o[1], x[2] - o[2]) for x in xs]
end

# `r` times the quantity on grid `gi` of frame `i` (from 1).
function quantity(F, i, gi, fields, xs, rs, models, which)
    Φ, Π = fields[gi]
    q = similar(Φ)
    if which == "rate"
        q .= Π
    elseif which == "total"
        q .= Φ
    elseif which == "accel"
        trs = [Trajectory(; z=SVector(F.zs[i][k]), v=SVector(F.vs[i][k])) for k in 1:F.K]
        Threads.@threads for j in eachindex(q)
            isnan(Φ[j]) && (q[j] = NaN32; continue)
            s = 0.0
            for (m, tr) in zip(models, trs)
                s += field_of_order(m, xs[j], tr, 0)
            end
            q[j] = Φ[j] - s
        end
    else
        error("--quantity is accel, rate or total, got $which")
    end
    return q .* Float32.(reshape(rs, size(q)))
end

# The faces of the elements in the plane `z = o[3]`, relative to `o`, as
# polylines separated by `NaN`: the run's forest built again from its
# layout, and in each block of the cube or of a shell about `x` or `y`
# that the plane crosses, the lines of constant logical `ξ₁` and `ξ₂`
# through its elements' faces, mapped by the ball's map at the plane's
# logical `ξ₃`, `n/2` in the shells to within `o[3]/Lc`, sixteen points
# per element edge in the shells, where the map is not affine.
function grid_lines(meta, o)
    num(key) = metanum(meta, key)
    haskey(meta, "hdiv") || error("--grid needs a run's frames, whose meta.tsv has its layout")
    hdiv, levels = parse(Int, meta["hdiv"][1]), parse(Int, meta["levels"][1])
    w0, R1, hs = num("w0"), num("R1"), num("hshell")
    # The shells' width rounded up a little, so that `ball_layout` takes
    # the run's number of radial roots rather than one more.
    lay = ball_layout(hdiv, levels, w0; R1, hshell=hs * (1 + 1e-9))
    haskey(meta, "m") && parse(Int, meta["m"][1]) != lay.m &&
        error("the layout gives $(lay.m) radial roots, the run had $(meta["m"][1])")
    refined = levels == 1 ? false : nested_cubes(Val(3), lay.n / 2^levels, levels)
    forest, geo = sevenpatch_forest(Float64, 3, lay.E; roots=(lay.n, lay.m), Lc=lay.Lc,
                                    R1, refined)
    origins = TreeAMR.block_origins(forest, Float64)
    spacings = TreeAMR.block_spacings(forest, Float64)
    zcube = (o[3] + lay.Lc) * lay.n / (2lay.Lc)
    pts = Point2f[]
    for (b, key) in enumerate(forest.leaves)
        p = TreeAMR.patch(forest, key)
        p <= 5 || continue                      # the cube and the shells about x and y
        w = forest.N * spacings[b]
        ξ3 = p == 1 ? zcube : lay.n / 2
        lo = origins[b]
        lo[3] <= (p == 1 ? ξ3 : ξ3 + 1e-9) < lo[3] + w || continue
        ns = p == 1 ? 1 : 16 * lay.E
        for d in 1:2, i in 0:lay.E
            c = lo[d] + i * w / lay.E
            for q in 0:ns
                u = lo[3 - d] + q * w / ns
                x = physical_coordinates(geo, d == 1 ? (c, u, ξ3) : (u, c, ξ3), p)
                push!(pts, Point2f(x[1] - o[1], x[2] - o[2]))
            end
            push!(pts, Point2f(NaN, NaN))
        end
    end
    return pts
end

# The modes' columns of a `--waves` file, `r Re Φ_ℓm` at radius `R`, from
# the run and, where the file has them, from the retarded field.
function waveforms(file, R, modes)
    lines = collect(eachline(file))
    head = split(lines[1], '\t')
    rows = [parse.(Float64, split(l, '\t')) for l in lines[2:end] if !isempty(l)]
    t = [r[1] for r in rows]
    name = isinteger(R) ? string(Int(R)) : string(R)
    curves = map(modes) do (l, m)
        col(prefix) = findfirst(==(prefix * "r$(name)_l$(l)m$(m)_re"), head)
        c = col("")
        c === nothing && error("$file has no mode ($l, $m) at r = $name")
        cr = col("ref_")
        (; l, m, run=[R * r[c] for r in rows],
         ref=cr === nothing ? nothing : [R * r[cr] for r in rows])
    end
    return t, curves
end

function main()
    dirs = filter(a -> !startswith(a, "--"), ARGS)
    length(dirs) == 1 || error("usage: visualize_orbit.jl DIR [options]")
    F = read_frames(dirs[1])
    meta = F.meta
    o = SVector(parse.(Float64, meta["centre"])...)
    Qs = parse.(Float64, meta["Q"])
    models = [ParticleModel(; Q=Qs[min(k, length(Qs))], ℓ=metanum(meta, "ell"), M=1.0)
              for k in 1:F.K]
    Ω = metanum(meta, "Omega")
    period = 2π / Ω
    dynamic = haskey(meta, "dynamic")
    # The separation and the orbits by the unwrapped phase of `z₁ − z₂`.
    sep = [hypot((F.zs[i][1] .- F.zs[i][2])[1:2]...) for i in eachindex(F.t)]
    phases = let φ = [atan(reverse((F.zs[i][1] .- F.zs[i][2])[1:2])...) for i in eachindex(F.t)]
        for i in 2:length(φ)
            φ[i] = φ[i - 1] + rem2pi(φ[i] - φ[i - 1], RoundNearest)
        end
        φ .- φ[1]
    end
    track = parse(Float64, option("track", string(dynamic ? Inf : period / 4)))
    which = option("quantity", "accel")
    refdir = option("ref", "")
    R = isempty(refdir) ? nothing : read_frames(refdir)
    showerror = "--error" in ARGS
    showerror && R === nothing && error("--error needs --ref=DIR")
    gname(name) = findfirst(g -> g.name == name, F.grids)
    wi = gname(option("wide", F.grids[1].name))
    zi = gname(option("zoom", F.grids[end].name))
    (wi === nothing || zi === nothing) && error("no such grid among $(F.grids)")
    gw, gz = F.grids[wi], F.grids[zi]
    xw, rw = geometry(gw, o)
    xz, rz = geometry(gz, o)
    R1 = metanum(meta, "R1")
    rmax = parse(Float64, option("rmax", string(min(0.9R1, gw.halfwidth))))
    axis(g) = [g.centre[1] + g.halfwidth * (2i - 1 - g.n) / g.n for i in 1:g.n] .- o[1]
    axisy(g) = [g.centre[2] + g.halfwidth * (2i - 1 - g.n) / g.n for i in 1:g.n] .- o[2]

    function frameq(i)
        fields = F.frame(i)
        qw = quantity(F, i, wi, fields, xw, rw, models, which)
        qz = quantity(F, i, zi, fields, xz, rz, models, which)
        if showerror
            j = findfirst(t -> abs(t - F.t[i]) < 1e-9, R.t)
            j === nothing && error("the reference has no frame at t = $(F.t[i])")
            rf = R.frame(j)
            qw .-= quantity(R, j, wi, rf, xw, rw, models, which)
            qz .-= quantity(R, j, zi, rf, xz, rz, models, which)
        end
        qw[reshape(rw, size(qw)) .> rmax] .= NaN32
        return qw, qz
    end

    nf = length(F.t)
    scale(q, r, lo) = begin
        a = [abs(q[j]) for j in eachindex(q) if !isnan(q[j]) && r[j] >= lo]
        isempty(a) ? 1.0 : 1.1 * quantile(a, 0.995)
    end
    lastw, lastz = frameq(nf)
    clim = parse(Float64, option("clim", string(scale(lastw, rw, rmax / 4))))
    zclim = parse(Float64, option("zclim", string(scale(lastz, rz, 0.0))))

    cmap = Symbol(option("colormap", "plasma"))
    fig = Figure(; size=(1600, 1000), fontsize=18)
    label = Dict("accel" => "r (Φ − Σ Ψ̂₀)", "rate" => "r ∂ₜΦ", "total" => "r Φ")[which]
    showerror && (label *= " − exact")
    title = Observable("")
    Label(fig[0, 1:3], title; fontsize=24, tellwidth=false)
    rows = 1 + (isempty(option("waves", "")) ? 0 : 1) + (dynamic ? 1 : 0)
    axw = Axis(fig[1:max(rows, 2), 1]; aspect=DataAspect(), title="r ≤ $(round(Int, rmax))",
               xlabel="x", ylabel="y")
    qwo = Observable(lastw)
    hm = heatmap!(axw, axis(gw), axisy(gw), qwo; colormap=cmap,
                  colorrange=(-clim, clim), nan_color=:transparent)
    Colorbar(fig[1:max(rows, 2), 2], hm; label)
    limits!(axw, -rmax, rmax, -rmax, rmax)
    axz = Axis(fig[1, 3]; aspect=DataAspect(), title="near zone", xlabel="x", ylabel="y")
    qzo = Observable(lastz)
    hz = heatmap!(axz, axis(gz), axisy(gz), qzo; colormap=cmap,
                  colorrange=(-zclim, zclim), nan_color=:transparent)
    limits!(axz, -gz.halfwidth, gz.halfwidth, -gz.halfwidth, gz.halfwidth)
    if "--grid" in ARGS
        lines = grid_lines(meta, o)
        # The wide map's lines end where its field does, at `rmax`.
        wlines = [hypot(q[1], q[2]) <= rmax ? q : Point2f(NaN, NaN) for q in lines]
        lines!(axw, wlines; color=(:grey60, 0.6), linewidth=0.4)
        lines!(axz, lines; color=(:grey60, 0.8), linewidth=0.8)
    end
    tracks = [Observable(Point2f[]) for _ in 1:F.K]
    heads = Observable(Point2f[])
    for k in 1:F.K
        lines!(axz, tracks[k]; color=(:black, 0.6), linewidth=2)
    end
    scatter!(axz, heads; color=:black, markersize=12)
    scatter!(axw, heads; color=:black, markersize=6)

    wavesfile = option("waves", "")
    cursor = Observable(0.0)
    if !isempty(wavesfile)
        modes = [Tuple(parse.(Int, split(m, ":"))) for m in split(option("modes", "2:2,4:4"), ",")]
        Rw = parse(Float64, option("wradius", "50"))
        tw, curves = waveforms(wavesfile, Rw, modes)
        axs = Axis(fig[2, 3]; xlabel="t", ylabel="r Re Φ_ℓm at r = $(Int(round(Rw)))")
        for (c, color) in zip(curves, Makie.wong_colors())
            lines!(axs, tw, c.run; color, label="ℓ = $(c.l), m = $(c.m)")
            c.ref === nothing || lines!(axs, tw, c.ref; color, linestyle=:dash)
        end
        vlines!(axs, cursor; color=:gray)
        axislegend(axs; position=:lt)
        xlims!(axs, first(tw), last(tw))
    end
    if dynamic
        axd = Axis(fig[rows, 3]; xlabel="t", ylabel="separation d")
        lines!(axd, F.t, sep; color=:black, label="run")
        # The leading order from the end of the hand-over on.
        t₀ = metanum(meta, "onset")
        i₀ = something(findfirst(>=(t₀), F.t), 1)
        Δq = Qs[1] - Qs[end]
        C = Δq^2 * metanum(meta, "binding") / 2π
        tl = [t for t in F.t if t >= F.t[i₀]]
        dl = [cbrt(sep[i₀]^3 - C * (t - F.t[i₀])) for t in tl]
        lines!(axd, tl, dl; color=:gray, linestyle=:dash, label="leading order")
        vlines!(axd, cursor; color=:gray)
        axislegend(axd; position=:lb)
        xlims!(axd, first(F.t), last(F.t))
    end
    colsize!(fig.layout, 1, Relative(0.55))

    function show!(i)
        qw, qz = frameq(i)
        qwo[] = qw
        qzo[] = qz
        t = F.t[i]
        title[] = dynamic ?
                  @sprintf("t = %.2f, %.2f orbits, d = %.3f", t, phases[i] / 2π, sep[i]) :
                  @sprintf("t = %.2f, %.2f orbits", t, t / period)
        cursor[] = t
        pts(i) = [Point2f(F.zs[i][k][1] - o[1], F.zs[i][k][2] - o[2]) for k in 1:F.K]
        heads[] = pts(i)
        back = [j for j in 1:i if F.t[i] - F.t[j] <= track]
        for k in 1:F.K
            tracks[k][] = [Point2f(F.zs[j][k][1] - o[1], F.zs[j][k][2] - o[2]) for j in back]
        end
        return nothing
    end

    out = String(option("out", ""))        # `save` of a movie takes a `String`
    mkpath(joinpath(@__DIR__, "output"))
    frame = option("frame", "")
    if !isempty(frame)
        i = parse(Int, frame) + 1
        show!(i)
        file = isempty(out) ?
               joinpath(@__DIR__, "output",
                        (dynamic ? "inspiral-" : "orbit-") * "$(lpad(i - 1, 5, '0')).png") : out
        save(file, fig)
        println(file)
        return nothing
    end
    file = isempty(out) ? joinpath(@__DIR__, "output",
                                   (dynamic ? "inspiral" : "orbit") *
                                   ("--grid" in ARGS ? "-grid.mp4" : ".mp4")) : out
    every = parse(Int, option("every", "1"))
    record(fig, file, 1:every:nf; framerate=parse(Int, option("fps", "30"))) do i
        show!(i)
    end
    println(file)
    return nothing
end

main()

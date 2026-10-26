#!/usr/bin/env julia
#
# Visualize the 2D radial blast wave on OLDG elements: the ring expanding
# across the element and block faces, how well the scheme keeps it radial,
# and how its error and energy evolve.
#
#     julia --project=bin bin/visualize2d.jl
#     julia --project=bin bin/visualize2d.jl --P=3 --roots=8 --out=/tmp
#     julia --project=bin bin/visualize2d.jl --plain
#
# TreeWave's viewer of the same case, without what this package does not
# have: there is no refinement, so no levels, block counts or indicator,
# and the run is `Float64` on the CPU. One figure, three parts:
#
#   1. a filmstrip of `u` at four times, drawn at the Gauss–Lobatto
#      nodes, with the element faces and the block faces on top;
#   2. `u` against radius at every node of the final frame, against the
#      exact free-space profile. The exact solution is a function of `r`
#      alone, so a perfect solution lands on one curve. Face nodes are
#      drawn apart from interior nodes: the two nodes of a face sit at one
#      position and may hold two values, and a jump between them shows as
#      a face node off the curve. Past the `L/2` rule the ring meets its
#      own periodic images and the exact solution stops being radial too;
#   3. the error of `u` against the exact ring, and the energy lost, over
#      time. Summation by parts makes the energy an invariant of the
#      semi-discrete system, so what is lost is RK4's.
#
# The run comes from `uniform_blast` via its `observer` keyword, so this
# script contains no time-stepping loop of its own. See "The viewer"
# in `CODE.md`.

using CairoMakie
# `Relative` is also TreeAMR's boundary hook; here it is Makie's layout size.
using CairoMakie: Relative
using Printf
using SixelTerm
using TreeAMR
using TreeWaveOLDG

# PNG, as in TreeWave: CairoMakie's image path pads each heatmap's edges
# so that abutting blocks leave no hairline seam, and vector output loses
# that and draws one polygon per cell.
CairoMakie.activate!(; type="png", px_per_unit=2)

"""
The `N + 1` cell edges along one dimension of a block whose `N` nodes are
`xs`, `E` elements of `P1` nodes each: per element, its two faces and the
midpoints between consecutive nodes. Each node gets the cell of points
closer to it than to its neighbours within its element, so the two face
nodes at one position each get the half-cell on their own side, and the
cells tile the block exactly.
"""
function nodeedges(xs, P1)
    edges = [xs[1]]
    for q in 0:(length(xs) ÷ P1 - 1)
        el = xs[(q * P1 + 1):(q * P1 + P1)]
        append!(edges, (el[1:(end - 1)] .+ el[2:end]) ./ 2)
        push!(edges, el[end])
    end
    return edges
end

"""
One frame of a run: every block's node positions, `u` and extent, the
element faces, and the errors and energy for the time series. Built inside
the observer, where `u` and the exact state `ue` are the run's own.
"""
function snapshot(p, t, u, ue)
    fs, el = p.fs, p.element
    P1 = length(el.nodes)
    N = fs.forest.N
    ua = statearray(u, fs)
    blocks = map(1:nblocks(fs)) do b
        xs = [node_coordinates(fs, el, b, (i, 1))[1] for i in 1:N]
        ys = [node_coordinates(fs, el, b, (1, j))[2] for j in 1:N]
        (ext=block_extent(fs.forest, blockkey(fs, b)), xs=xs, ys=ys,
         xedges=nodeedges(xs, P1), yedges=nodeedges(ys, P1),
         u=Array(view(ua, :, :, 1, b)))
    end
    err = u .- ue
    return (t=Float64(t), blocks=blocks, P1=P1,
            linf=quadrature_norm(fs, el, err; p=Inf, vars=1),
            l2=quadrature_norm(fs, el, err; vars=1), energy=wave_energy(p, u))
end

"""
One filmstrip panel: one heatmap per block on its node cells, then the
element faces faintly and the block faces on top. Element and block
outlines are each one `lines!` with `NaN` breaks rather than one plot per
rectangle, which Makie builds a thousand times faster.
"""
function fieldpanel!(ax, snap; colorrange, colorscale)
    local hm
    for b in snap.blocks
        hm = heatmap!(ax, b.xedges, b.yedges, b.u; colormap=:balance,
                      colorrange=colorrange, colorscale=colorscale)
    end
    faces = Point2f[]
    blocks = Point2f[]
    for b in snap.blocks
        (x1, x2), (y1, y2) = b.ext
        for x in b.xs[(snap.P1 + 1):snap.P1:end]
            append!(faces, (Point2f(x, y1), Point2f(x, y2), Point2f(NaN, NaN)))
        end
        for y in b.ys[(snap.P1 + 1):snap.P1:end]
            append!(faces, (Point2f(x1, y), Point2f(x2, y), Point2f(NaN, NaN)))
        end
        append!(blocks, (Point2f(x1, y1), Point2f(x2, y1), Point2f(x2, y2),
                         Point2f(x1, y2), Point2f(x1, y1), Point2f(NaN, NaN)))
    end
    lines!(ax, faces; color=(:gray40, 0.35), linewidth=0.3)
    lines!(ax, blocks; color=(:black, 0.6), linewidth=0.6)
    return hm
end

"""
Every node of one frame as `(radius, u)`, split into face nodes (on an
element face in either dimension) and interior nodes. The radius is the
minimum-image distance to the centre, so the corners of the periodic box
do not form a branch of their own.
"""
function radialpoints(snap, L, x₀)
    face = (r=Float64[], u=Float64[])
    inner = (r=Float64[], u=Float64[])
    P1 = snap.P1
    for b in snap.blocks, j in eachindex(b.ys), i in eachindex(b.xs)
        dx = mod(b.xs[i] - x₀[1] + L / 2, L) - L / 2
        dy = mod(b.ys[j] - x₀[2] + L / 2, L) - L / 2
        onface = (i - 1) % P1 in (0, P1 - 1) || (j - 1) % P1 in (0, P1 - 1)
        dest = onface ? face : inner
        push!(dest.r, hypot(dx, dy))
        push!(dest.u, b.u[i, j])
    end
    return face, inner
end

function makefigure(snaps, profile, L, x₀, title)
    last = snaps[end]
    frames = snaps[round.(Int, range(1, length(snaps); length=4))]

    # One symmetric range and one colorbar for all four frames, so the
    # strip shows the ring fading and not merely moving; the symlog scale
    # keeps the late frames, ten times smaller than the first, visible.
    amp = maximum(maximum(abs, b.u) for s in frames for b in s.blocks)
    crange = (-amp, amp)
    cscale = Makie.Symlog10(amp / 100)

    fig = Figure(; size=(1100, 1250))
    Label(fig[0, 1:2], title; fontsize=18, font=:bold, tellwidth=false)

    strip = GridLayout(fig[1, 1])
    local hm
    for (k, snap) in enumerate(frames)
        ax = Axis(strip[1, k]; aspect=DataAspect(), titlesize=11,
                  title=@sprintf("t = %.2f", snap.t))
        hm = fieldpanel!(ax, snap; colorrange=crange, colorscale=cscale)
        hidedecorations!(ax)
    end
    Colorbar(strip[1, 5], hm; label="u", width=12, height=Relative(0.92),
             ticklabelsize=9, labelsize=11)
    colgap!(strip, 6)

    ax_prof = Axis(fig[2, 1]; xlabel="r", ylabel="u",
                   title="u against radius, every node of the final frame \
                          — one curve means the scheme kept the solution radial")
    face, inner = radialpoints(last, L, x₀)
    CairoMakie.scatter!(ax_prof, inner.r, inner.u; color=(:steelblue, 0.35),
                        markersize=1.5)
    CairoMakie.scatter!(ax_prof, face.r, face.u; color=(:darkorange, 0.35),
                       markersize=1.5)
    # The free-space profile, drawn only out to L/2: past that a node is
    # near a face or a corner of the box, where it feels the neighbouring
    # images as well as the source.
    keep = profile.rs .<= L / 2
    lines!(ax_prof, profile.rs[keep], profile.us[keep];
                color=(:black, 0.75), linewidth=1.2, linestyle=:dash)
    vlines!(ax_prof, [L / 2]; color=(:gray, 0.6), linewidth=1, linestyle=:dashdot)
    text!(ax_prof, L / 2, 0.0; text="  past here the ring meets its own \
                                      periodic images",
          align=(:left, :bottom), color=:gray, fontsize=10)

    ax_err = Axis(fig[3, 1]; xlabel="t", ylabel="error of u", yscale=log10,
                  title="the error against the exact ring, and the energy \
                         RK4 loses")
    ax_en = Axis(fig[3, 1]; ylabel="energy lost, (E₀ - E)/E₀",
                 yaxisposition=:right, yscale=log10)
    hidespines!(ax_en)
    hidexdecorations!(ax_en)
    linkxaxes!(ax_err, ax_en)
    ts = [s.t for s in snaps]
    li = lines!(ax_err, ts, [s.linf for s in snaps]; linewidth=2)
    l2 = lines!(ax_err, ts, [s.l2 for s in snaps]; linewidth=2, linestyle=:dot)
    # The first frame has lost nothing, which a log axis cannot draw.
    E0 = snaps[1].energy
    en = lines!(ax_en, ts[2:end], [(E0 - s.energy) / E0 for s in snaps[2:end]];
                linewidth=2, linestyle=:dash, color=:firebrick)

    # Legend markers of their own: the plotted ones are 1.5 px and
    # translucent, which a legend draws as nothing.
    Legend(fig[2, 2],
           [MarkerElement(; color=:steelblue, marker=:circle, markersize=8),
            MarkerElement(; color=:darkorange, marker=:circle, markersize=8),
            LineElement(; color=:black, linestyle=:dash)],
           ["interior node", "element face node", "exact"];
           framevisible=false, tellheight=false, valign=:top)
    Legend(fig[3, 2], [li, l2, en], ["L∞ error", "L2 error", "energy lost"];
           framevisible=false, tellheight=false, valign=:top)

    rowsize!(fig.layout, 1, Aspect(1, 0.25))
    rowgap!(fig.layout, 10)
    return fig
end

"""
The blast wave on a uniform box, `roots × roots` blocks of `E × E`
elements of order `P`, with truncation unless `truncate = false`.
"""
function blastcase(; P=4, E=2, roots=16, L=1.0, σ=0.08, t_end=0.4, truncate=true)
    x₀ = (L / 2, L / 2)
    snaps = []
    observer(p, t, u, ue) = push!(snaps, snapshot(p, t, u, ue))
    r = uniform_blast(; P=P, E=E, roots=roots, L=L, σ=σ, t_end=t_end,
                      truncate=truncate, observer=observer)
    scheme = truncate ? "truncated" : "plain"
    # The errors of `u` alone, as the bottom panel draws them; the
    # driver's own errors include `∂ₜu`, whose amplitude is larger.
    last = snaps[end]
    title = @sprintf("Radial blast wave, σ = %.3g, OLDG P = %d, %d×%d blocks \
                      of %d×%d elements, %s\nerror of u at t = %.2f: \
                      L∞ %.3g, L2 %.3g; energy lost %.2g",
                     σ, P, roots, roots, E, E, scheme, last.t, last.linf,
                     last.l2, (r.energy0 - r.energy) / r.energy0)
    # The free-space radial solution at the final time, straight from the
    # Hankel table rather than sampled along a ray through the box, which
    # past r = L/2 would meet a neighbouring image.
    ref = blast_reference(L, x₀, σ; rmax=t_end + 6σ)
    tab = blast_radial_table(ref, snaps[end].t)
    return (snaps=snaps, profile=(rs=collect(ref.rs), us=tab.us), L=L, x₀=x₀,
            title=title)
end

function main(args)
    outdir = joinpath(@__DIR__, "output")
    P, E, roots = 4, 2, 16
    truncate = true
    # Sixel is for a human looking at a terminal; a pipe gets the path.
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--out=")
            outdir = a[7:end]
        elseif startswith(a, "--P=")
            P = parse(Int, a[5:end])
        elseif startswith(a, "--E=")
            E = parse(Int, a[5:end])
        elseif startswith(a, "--roots=")
            roots = parse(Int, a[9:end])
        elseif a == "--plain"
            truncate = false
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --out=, --P=, --E=, --roots=, \
                   --plain, --display, --no-display")
        end
    end

    mkpath(outdir)
    @info "running the blast case"
    c = blastcase(; P=P, E=E, roots=roots, truncate=truncate)
    fig = makefigure(c.snaps, c.profile, c.L, c.x₀, c.title)
    path = joinpath(outdir, "blast_2d" * (truncate ? "" : "_plain") * ".png")
    save(path, fig)
    inline && display(fig)
    println("wrote $path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)

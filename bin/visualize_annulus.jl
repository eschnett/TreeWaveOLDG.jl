#!/usr/bin/env julia
#
# Visualize the scalar wave on TreeAMR's four-patch annulus: two runs on
# the same mesh, drawn in physical space with the patches' glued faces on
# top, to show that the glue is invisible to the solution and that the
# energy is kept across it.
#
#     julia --project=bin bin/visualize_annulus.jl
#     julia --project=bin bin/visualize_annulus.jl --P=3 --roots=8,4 --out=/tmp
#     julia --project=bin bin/visualize_annulus.jl --plain
#
# One figure, three rows:
#
#   1. a Gaussian pulse at rest, off-centre in patch 1, at four times: it
#      splits into a ring, reflects off both walls, and runs through the
#      glued faces into patches 2 and 4 and round to patch 3. There is no
#      exact solution; what to look for is that nothing reflects off or
#      bends at a glued face (the four bold radial lines);
#   2. the rotating Neumann Bessel mode `m = 3` (`annulus_exact`) at `t = 0`
#      and a quarter turn later, when its pattern has moved by one patch,
#      and its pointwise error then, on a log scale. Three
#      lobe pairs on four patches put each glue at a different phase of
#      the pattern, so an error made at a glue would show there as a
#      feature of its own;
#   3. the energy change of both runs, and the mode's L2 and L∞ errors
#      over a full turn. With truncation the error swells and shrinks
#      with the period `2π/k` rather than growing: truncation freezes
#      `u`'s top mode at its initial value, and the exact mode's top mode
#      turns away from it and back once a period ("Truncation" in
#      `CODE.md`); with `--plain` it stays at its smaller level. The energy is an exact invariant of the semi-discrete
#      system across the glued faces and the walls; what changes is RK4's.
#
# Each element's polynomial is drawn, sampled on a fine grid inside the
# element by its Lagrange basis at its Gauss–Lobatto nodes and mapped to
# `(x, y)` by `physical_coordinates`, so the picture is the solution the
# scheme holds rather than its nodes joined by straight lines. See "The
# annulus" in `CODE.md`.

using CairoMakie
# `Relative` is also TreeAMR's boundary hook; here it is Makie's layout size.
using CairoMakie: Relative
using Printf
using SixelTerm
using TreeAMR
using TreeWaveOLDG

CairoMakie.activate!(; type="png", px_per_unit=2)

# Samples per element along each dimension.
const SAMPLES = 12

"""
The weights of the Lagrange basis at the nodes `xn`, evaluated at `x`.
"""
function lagrange(xn, x)
    return [prod((x - xn[k]) / (xn[j] - xn[k]) for k in eachindex(xn) if k != j;
                 init=1.0)
            for j in eachindex(xn)]
end

"""
One frame: every element as `(xy, u, p, logical)`, the element's polynomial
sampled on `SAMPLES × SAMPLES` points, with their physical positions, the
patch, and the logical positions (for the exact mode). Copied out of the
integrator's state, which the next step overwrites.
"""
function snapshot(prob, geo, t, u)
    fs, el = prob.fs, prob.element
    P1 = length(el.nodes)
    N = fs.forest.N
    ua = statearray(u, fs)
    elements = []
    for b in 1:nblocks(fs)
        p = Int(TreeAMR.patch(fs.forest, blockkey(fs, b)))
        xs = [Float64(node_coordinates(fs, el, b, (i, 1))[1]) for i in 1:N]
        ys = [Float64(node_coordinates(fs, el, b, (1, j))[2]) for j in 1:N]
        U = Array{Float64}(view(ua, :, :, 1, b))
        for qy in 0:(N ÷ P1 - 1), qx in 0:(N ÷ P1 - 1)
            ix = (qx * P1 + 1):(qx * P1 + P1)
            iy = (qy * P1 + 1):(qy * P1 + P1)
            sx = range(xs[ix[1]], xs[ix[end]]; length=SAMPLES)
            sy = range(ys[iy[1]], ys[iy[end]]; length=SAMPLES)
            Wx = reduce(vcat, (lagrange(xs[ix], x)' for x in sx))
            Wy = reduce(vcat, (lagrange(ys[iy], y)' for y in sy))
            vals = Wx * U[ix, iy] * Wy'
            logical = [(x, y) for x in sx, y in sy]
            xy = [Point2f(physical_coordinates(geo, l, p)...) for l in logical]
            push!(elements, (xy=xy, u=vals, p=p, logical=logical))
        end
    end
    return (t=Float64(t), elements=elements)
end

"""
All elements of a frame as one triangle mesh with a value per vertex, which
`mesh!` draws a thousand times faster than one plot per element.
"""
function framemesh(elements, value)
    V = Point2f[]
    F = NTuple{3,Int}[]
    C = Float64[]
    for e in elements
        o = length(V)
        S = size(e.u, 1)
        append!(V, vec(e.xy))
        append!(C, vec(value(e)))
        idx(i, j) = o + i + (j - 1) * S
        for j in 1:(S - 1), i in 1:(S - 1)
            push!(F, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
            push!(F, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))
        end
    end
    return V, [f[k] for f in F, k in 1:3], C
end

"""
The mesh on top of a field: element faces faintly, block faces, the two
walls, and the four glued faces bold, each one `lines!` with `NaN` breaks.
"""
function meshlines!(ax, geo, forest, P1, E)
    r0, r1 = Float64(geo.r0), Float64(geo.r1)
    n1, n2 = geo.roots
    arc(r, θa, θb) = [Point2f(r * cos(θ), r * sin(θ)) for θ in range(θa, θb; length=64)]
    ray(θ) = [Point2f(r0 * cos(θ), r0 * sin(θ)), Point2f(r1 * cos(θ), r1 * sin(θ))]
    gap = Point2f(NaN, NaN)
    elem, block = Point2f[], Point2f[]
    for k in 0:(4 * n1 * E)
        θ = 2π * k / (4 * n1 * E)
        append!(k % E == 0 ? block : elem, ray(θ), (gap,))
    end
    for k in 1:(n2 * E - 1)
        r = r0 + (r1 - r0) * k / (n2 * E)
        append!(k % E == 0 ? block : elem, arc(r, 0, 2π), (gap,))
    end
    lines!(ax, elem; color=(:gray40, 0.35), linewidth=0.4)
    lines!(ax, block; color=(:black, 0.45), linewidth=0.7)
    for r in (r0, r1)
        lines!(ax, arc(r, 0, 2π); color=:black, linewidth=1.2)
    end
    glue = reduce(vcat, [vcat(ray(θ), [gap]) for θ in (0, π / 2, π, 3π / 2)])
    lines!(ax, glue; color=(:darkorange, 0.95), linewidth=2.2)
    for p in 1:4
        θ = (p - 1) * π / 2 + π / 4
        text!(ax, 1.17 * r1 * cos(θ), 1.17 * r1 * sin(θ); text="patch $p",
              align=(:center, :center), fontsize=10, color=:gray30)
    end
    return nothing
end

function fieldpanel!(ax, geo, forest, P1, E, snap, value; kw...)
    V, F, C = framemesh(snap.elements, value)
    m = mesh!(ax, V, F; color=C, shading=NoShading, kw...)
    meshlines!(ax, geo, forest, P1, E)
    limits!(ax, -1.3 * geo.r1, 1.3 * geo.r1, -1.3 * geo.r1, 1.3 * geo.r1)
    hidedecorations!(ax)
    hidespines!(ax)
    return m
end

"""
One run on the annulus: evolve `init` (a `WithPatch` callback) to `tend`,
keeping a snapshot at each of `times` and the energy, and with `exact` the
errors of `u`, at `nseries` points.
"""
function annulus_run(prob, geo, init, tend, times; exact=nothing, nseries=200, cfl=0.5)
    u0 = initial_state(prob, init)
    nsteps = ceil(Int, tend / oldg_timestep(prob; cfl=cfl))
    dt = tend / nsteps
    integ = wave_integrator(prob, u0, 0.0, tend; dt=dt)
    at = Dict(round(Int, t / dt) => t for t in times)
    every = max(1, nsteps ÷ nseries)
    snaps = []
    series = (t=Float64[], energy=Float64[], l2=Float64[], linf=Float64[])
    for k in 0:nsteps
        # IMEXRungeKutta's, which this environment reaches only through
        # the package.
        k > 0 && TreeWaveOLDG.step!(integ)
        haskey(at, k) && push!(snaps, snapshot(prob, geo, integ.t, integ.u))
        if k % every == 0 || k == nsteps
            push!(series.t, integ.t)
            push!(series.energy, wave_energy(prob, integ.u))
            if exact !== nothing
                err = integ.u .- initial_state(prob, exact(integ.t))
                push!(series.l2, wave_norm(prob, err; vars=1))
                push!(series.linf, wave_norm(prob, err; p=Inf, vars=1))
            end
        end
    end
    return (snaps=snaps, series=series, nsteps=nsteps, dt=dt)
end

"""
A Gaussian `u` of width `σ` at rest, centred at radius `rc` and angle
`θc`, as a `WithPatch` callback.
"""
struct Pulse{G}
    geo::G
    rc::Float64
    θc::Float64
    σ::Float64
end
function (f::Pulse)(x, p, var)
    var == 2 && return zero(x[1])
    X, Y = physical_coordinates(f.geo, x, p)
    d2 = (X - f.rc * cos(f.θc))^2 + (Y - f.rc * sin(f.θc))^2
    return oftype(x[1], exp(-d2 / (2 * f.σ^2)))
end

function main(args)
    outdir = joinpath(@__DIR__, "output")
    P, E, roots = 4, 2, (4, 2)
    r0, r1 = 1.0, 2.0
    m = 3
    σ = 0.12
    truncate = true
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--out=")
            outdir = a[7:end]
        elseif startswith(a, "--P=")
            P = parse(Int, a[5:end])
        elseif startswith(a, "--E=")
            E = parse(Int, a[5:end])
        elseif startswith(a, "--roots=")
            roots = Tuple(parse.(Int, split(a[9:end], ",")))
        elseif startswith(a, "--m=")
            m = parse(Int, a[5:end])
        elseif startswith(a, "--sigma=")
            σ = parse(Float64, a[9:end])
        elseif a == "--plain"
            truncate = false
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --out=, --P=, --E=, --roots=n1,n2, \
                   --m=, --sigma=, --plain, --display, --no-display")
        end
    end

    forest, geo = annulus_forest(Float64, P, E; roots=roots, r0=r0, r1=r1)
    prob = WaveProblem(forest, Element(Float64, P); truncate=truncate, geometry=geo)
    P1 = P + 1

    @info "running the pulse"
    pulse = WithPatch(Pulse(geo, (r0 + r1) / 2, π / 8, σ))
    ptimes = [0.0, 0.75, 2.0, 4.0]
    pr = annulus_run(prob, geo, pulse, ptimes[end], ptimes)

    @info "running the Bessel mode"
    k = annulus_wavenumber(m, 1, r0, r1)
    turn = 2π * m / k
    exact(t) = annulus_exact(geo, m, k, t)
    mr = annulus_run(prob, geo, exact(0.0), turn, [0.0, turn / 4]; exact=exact)

    fig = Figure(; size=(1250, 1300))
    scheme = truncate ? "truncated" : "plain"
    nblk = nleaves(forest)
    Label(fig[0, 1:2], @sprintf("The scalar wave on TreeAMR's four-patch annulus, \
                                 r ∈ [%g, %g]: OLDG P = %d, %s, %d blocks (%d×%d per \
                                 patch) of %d×%d elements", r0, r1, P, scheme, nblk,
                                roots..., E, E);
          fontsize=18, font=:bold, tellwidth=false)

    # Row 1: the pulse, one symmetric range, symlog so the late frames show.
    row1 = GridLayout(fig[1, 1:2])
    Label(row1[0, 1:4], @sprintf("a Gaussian pulse at rest, σ = %g, at r = %g, θ = π/8: \
                                  it crosses the glued faces (orange) without \
                                  reflecting", σ, (r0 + r1) / 2); fontsize=13)
    amp = maximum(maximum(abs, e.u) for s in pr.snaps for e in s.elements)
    local hm
    for (j, s) in enumerate(pr.snaps)
        ax = Axis(row1[1, j]; aspect=DataAspect(), title=@sprintf("t = %.2f", s.t),
                  titlesize=12)
        hm = fieldpanel!(ax, geo, forest, P1, E, s, e -> e.u; colormap=:balance,
                         colorrange=(-amp, amp), colorscale=Makie.Symlog10(amp / 50))
    end
    Colorbar(row1[1, 5], hm; label="u", width=12, height=Relative(0.8))
    colgap!(row1, 6)

    # Row 2: the mode, then its error after a full turn.
    row2 = GridLayout(fig[2, 1:2])
    Label(row2[0, 1:3], @sprintf("the rotating Neumann Bessel mode m = %d, k = %.4f: \
                                  three lobe pairs on four patches", m, k); fontsize=13)
    mamp = maximum(maximum(abs, e.u) for e in mr.snaps[1].elements)
    local hmm
    for (j, (s, what)) in enumerate(zip(mr.snaps[1:2], ("", "a quarter turn, one \
                                                       patch on, ")))
        ax = Axis(row2[1, j]; aspect=DataAspect(), titlesize=12,
                  title=@sprintf("%st = %.2f", what, s.t))
        hmm = fieldpanel!(ax, geo, forest, P1, E, s, e -> e.u; colormap=:balance,
                          colorrange=(-mamp, mamp))
    end
    Colorbar(row2[1, 3], hmm; label="u", width=12, height=Relative(0.8))
    last = mr.snaps[2]
    # At the snapshot's own time, the step nearest a quarter turn: half a
    # step's phase is ten times the error.
    mode = exact(last.t).f
    errof(e) = [log10(max(abs(e.u[i] - mode(e.logical[i], e.p, 1)), 1e-16))
                for i in eachindex(e.u)]
    lo, hi = extrema(reduce(vcat, vec(errof(e)) for e in last.elements))
    ax = Axis(row2[1, 4]; aspect=DataAspect(), titlesize=12,
              title=@sprintf("|error of u| a quarter turn on, t = %.2f", last.t))
    he = fieldpanel!(ax, geo, forest, P1, E, last, errof; colormap=:viridis,
                     colorrange=(max(lo, hi - 4), hi))
    Colorbar(row2[1, 5], he; label="log₁₀ |u - exact|", width=12, height=Relative(0.8))
    colgap!(row2, 6)

    # Row 3: energy and errors over time.
    ax_en = Axis(fig[3, 1]; xlabel="t", ylabel="|E - E₀| / E₀", yscale=log10,
                 title="energy change: RK4's alone, the glue loses none")
    for (r, lbl, c) in ((pr, "pulse", :steelblue), (mr, "Bessel mode", :firebrick))
        s = r.series
        E0 = s.energy[1]
        keep = 2:length(s.t)
        lines!(ax_en, s.t[keep], max.(abs.(s.energy[keep] .- E0) ./ E0, 1e-17);
               label=@sprintf("%s, dt = %.2g", lbl, r.dt), color=c, linewidth=2)
    end
    axislegend(ax_en; position=:rb, framevisible=false)
    ax_err = Axis(fig[3, 2]; xlabel="t", ylabel="error of u", yscale=log10,
                  title=truncate ?
                        "the mode's error over a full turn: u's top mode frozen \
                         by truncation, period 2π/k" :
                        "the mode's error over a full turn, √g-weighted L2 and \
                         nodal L∞")
    s = mr.series
    keep = 2:length(s.t)
    lines!(ax_err, s.t[keep], s.l2[keep]; label="L2", linewidth=2)
    lines!(ax_err, s.t[keep], s.linf[keep]; label="L∞", linewidth=2, linestyle=:dot)
    vlines!(ax_err, [turn / 4]; color=(:gray, 0.6), linestyle=:dashdot)
    axislegend(ax_err; position=:rb, framevisible=false)

    rowsize!(fig.layout, 1, Relative(0.3))
    rowsize!(fig.layout, 2, Relative(0.3))
    mkpath(outdir)
    path = joinpath(outdir, "annulus_2d" * (truncate ? "" : "_plain") * ".png")
    save(path, fig)
    inline && display(fig)
    @printf("pulse: %d steps, energy change %.2e; mode: %d steps, L2 %.3e, L∞ %.3e, \
             energy change %.2e\n", pr.nsteps,
            abs(pr.series.energy[end] / pr.series.energy[1] - 1), mr.nsteps,
            s.l2[end], s.linf[end], abs(s.energy[end] / s.energy[1] - 1))
    println("wrote $path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)

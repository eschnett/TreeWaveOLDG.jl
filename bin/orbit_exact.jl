# The exact frames of `bin/orbit_movie.jl`'s run ("The movie" in
# `CODE.md`): the two point charges' retarded field (`retarded_field`) and
# its rate on the same grids at the same times, `NaN` beyond the ball's
# sphere as in the run, written in the same format (`bin/plane_io.jl`). It
# evolves nothing, so it runs on the host, threaded over the points, in
# about a second per million points and frame on four threads:
#
#     julia -t N,0 --project=. bin/orbit_exact.jl --plane=DIR [--frames=a:b] \
#         [the orbit's and the grids' options of bin/orbit_movie.jl]
#
# `--frames=a:b` writes only the frames numbered `a` to `b` (from 0, at
# `t = i pevery`), for trying out the viewer on a few. The point charges'
# field is the run's but for the charges' finite size, `ℓ = 10⁻³`, which
# no grid point comes near.

include(joinpath(@__DIR__, "orbit_movie.jl"))

function exact()
    opts = orbit_options()
    dir = option("plane", "")
    isempty(dir) && error("--plane=DIR is needed")
    o = SVector(0.0, 0.0, 0.0) + OFFSET
    path = orbit_motion(o, opts).path
    gs = grids(o)
    nframes = floor(Int, opts.tend / opts.pevery + 1e-9) + 1
    fr = option("frames", "0:$(nframes - 1)")
    a, b = parse.(Int, split(fr, ":"))
    writer = plane_writer(dir, orbit_meta(o, opts, "exact"), gs, 2)
    points = [TreeWaveOLDG.plane_grid(SVector(g.centre), g.halfwidth, g.n) for g in gs]
    for i in a:min(b, nframes - 1)
        t = i * opts.pevery
        start = time()
        fields = map(points) do xs
            Φ = fill(NaN32, length(xs))
            Π = fill(NaN32, length(xs))
            Threads.@threads for j in eachindex(xs)
                x = xs[j]
                norm(x) < opts.R1 || continue
                s, r = 0.0, 0.0
                for k in 1:2
                    f, ft = retarded_field(path, k, MODEL.Q, x, t)
                    s += f
                    r += ft
                end
                Φ[j], Π[j] = s, r
            end
            n = isqrt(length(xs))
            (reshape(Φ, n, n), reshape(Π, n, n))
        end
        write_frame(writer, t, path(t), fields; number=i)
        @info "frame $i at t = $t in $(round(time() - start; digits=2)) s"
    end
    close(writer)
    return nothing
end

exact()

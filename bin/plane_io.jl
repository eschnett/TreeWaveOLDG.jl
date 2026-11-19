# The frames of `bin/orbit_movie.jl` and `bin/orbit_exact.jl`, which
# `bin/visualize_orbit.jl` reads ("The movie" in `CODE.md`): the total
# field `Φ` and its rate `Π = ∂ₜΦ` on square grids in the orbit's plane,
# at a cadence fit for a movie. A frame directory holds
#
# - `meta.tsv`: one row per key, the key and its values, tab-separated;
#   among them a `grid` row per grid, its name, half-width, `n` and
#   centre, in the order the frames hold the grids;
# - `frames.tsv`: a header row, then per frame its number, `t` and each
#   particle's position and velocity (`z1x … v1z`, `z2x …`);
# - `frame-NNNNN.f32`: raw little-endian `Float32`, per grid in
#   `meta.tsv`'s order `Φ` then `Π`, each `n × n` in column order, `x`
#   running fastest, `NaN` outside the ball.
#
# The grids' points are `TreeWaveOLDG.plane_grid`'s, the cell centres.

struct PlaneWriter
    dir::String
    grids::Vector{NamedTuple{(:name, :halfwidth, :n, :centre),
                             Tuple{String,Float64,Int,NTuple{3,Float64}}}}
    K::Int
    io::IOStream
    count::Base.RefValue{Int}
end

# A frame directory `dir`, made if it is not there, with `meta.tsv` from
# `meta`, pairs of a key and a value or a tuple of values, and the grids,
# `(name, halfwidth, n, centre)`; `frames.tsv` opened with its header for
# `K` particles.
function plane_writer(dir, meta, grids, K)
    mkpath(dir)
    gs = [(; name=String(g.name), halfwidth=Float64(g.halfwidth), n=Int(g.n),
           centre=Tuple(Float64.(g.centre))) for g in grids]
    open(joinpath(dir, "meta.tsv"), "w") do io
        for (key, value) in meta
            println(io, join((key, (value isa Tuple ? value : (value,))...), '\t'))
        end
        for g in gs
            println(io, join(("grid", g.name, g.halfwidth, g.n, g.centre...), '\t'))
        end
    end
    io = open(joinpath(dir, "frames.tsv"), "w")
    cols = ["$q$k$c" for k in 1:K for q in ("z", "v") for c in ("x", "y", "z")]
    println(io, join(("frame", "t", cols...), '\t'))
    flush(io)
    return PlaneWriter(String(dir), gs, K, io, Ref(0))
end

framefile(dir, i) = joinpath(dir, "frame-" * lpad(i, 5, '0') * ".f32")

# One frame at time `t`: the particles' trajectories `trajs` and, per
# grid, `(Φ, Π)`; numbered on from the last one written, or `number`.
function write_frame(w::PlaneWriter, t, trajs, fields; number=w.count[])
    length(fields) == length(w.grids) ||
        error("$(length(fields)) grids' fields for $(length(w.grids)) grids")
    length(trajs) == w.K || error("$(length(trajs)) trajectories for $(w.K) particles")
    i = number
    open(framefile(w.dir, i), "w") do io
        for ((Φ, Π), g) in zip(fields, w.grids)
            size(Φ) == size(Π) == (g.n, g.n) ||
                error("grid $(g.name) is $(g.n) × $(g.n), got $(size(Φ)) and $(size(Π))")
            write(io, htol.(Float32.(Φ)))
            write(io, htol.(Float32.(Π)))
        end
    end
    row = Any[i, t]
    for tr in trajs
        append!(row, tr.z)
        append!(row, tr.v)
    end
    println(w.io, join(row, '\t'))
    flush(w.io)
    w.count[] = i + 1
    return nothing
end

Base.close(w::PlaneWriter) = close(w.io)

# A frame directory: `meta`, a `Dict` from each key to its values as
# strings; `grids` as written; `t` and the particles' `z` and `v`,
# `zs[i][k]`, per frame; and `frame(i)`, the `i`-th frame (from 1) as a
# vector of `(Φ, Π)` per grid.
function read_frames(dir)
    meta = Dict{String,Vector{String}}()
    grids = NamedTuple[]
    for line in eachline(joinpath(dir, "meta.tsv"))
        f = split(line, '\t')
        if f[1] == "grid"
            push!(grids, (; name=String(f[2]), halfwidth=parse(Float64, f[3]),
                          n=parse(Int, f[4]),
                          centre=Tuple(parse.(Float64, f[5:7]))))
        else
            meta[String(f[1])] = String.(f[2:end])
        end
    end
    lines = collect(eachline(joinpath(dir, "frames.tsv")))
    rows = [parse.(Float64, split(l, '\t')) for l in lines[2:end] if !isempty(l)]
    K = (length(split(lines[1], '\t')) - 2) ÷ 6
    t = [r[2] for r in rows]
    vec3(r, o) = (r[o + 1], r[o + 2], r[o + 3])
    zs = [[vec3(r, 2 + 6(k - 1)) for k in 1:K] for r in rows]
    vs = [[vec3(r, 5 + 6(k - 1)) for k in 1:K] for r in rows]
    numbers = [Int(r[1]) for r in rows]
    function frame(i)
        out = Tuple{Matrix{Float32},Matrix{Float32}}[]
        open(framefile(dir, numbers[i]), "r") do io
            for g in grids
                Φ = Matrix{Float32}(undef, g.n, g.n)
                Π = Matrix{Float32}(undef, g.n, g.n)
                read!(io, Φ)
                read!(io, Π)
                push!(out, (ltoh.(Φ), ltoh.(Π)))
            end
        end
        return out
    end
    return (; meta, grids, K, t, zs, vs, frame)
end

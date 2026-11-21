# `--waves=FILE` of the particle drivers `bin/prescribed.jl`,
# `bin/orbit_movie.jl` and `bin/inspiral_movie.jl`: the field's modes
# `Φ_ℓm` on spheres about the centre (`SphereExtraction`, "Waveforms" in
# `CODE.md`), one row per output time, and beside them the retarded
# field's when the data are the
# retarded field. The options are `--wradii` (10,20,30,50,70,90),
# `--lmax` (6), `--ntheta` (`2lmax + 12`) and, read by the drivers,
# `--wevery` (0.0625); `option` is the including script's. `bin/waveforms.jl`
# reads the file.

# `--waves`: the extraction's spheres and its file, with the header row
# written; `nothing` without the option.
function wave_writer(p, c, retarded)
    file = option("waves", "")
    isempty(file) && return nothing
    ws = option("wradii", "10,20,30,50,70,90")
    radii = [parse(Float64, strip(r)) for r in split(ws, ",")]
    lmax = parse(Int, option("lmax", "6"))
    nθ = parse(Int, option("ntheta", string(2lmax + 12)))
    ex = SphereExtraction(p.wave, radii; lmax, nθ, centre=c)
    io = open(file, "w")
    name(r) = isinteger(r) ? string(Int(r)) : string(r)
    cols = ["r$(name(r))_l$(l)m$(m)_$(part)" for r in radii
            for (l, m) in harmonic_modes(lmax) for part in ("re", "im")]
    retarded === nothing || append!(cols, "ref_" .* cols)
    println(io, join(("t", cols...), '\t'))
    flush(io)
    return (; ex, io)
end

# One row of `--waves`: the state's modes, and the retarded field's, the
# sum of every particle's, along `path`, by default the motion's
# (`bin/inspiral_movie.jl` passes the path it records).
function write_waves(w, p, u, t, retarded; path=nothing)
    w === nothing && return nothing
    path = path === nothing ? p.motion.path : path
    row = [t]
    flat(modes) = vec(permutedims(hcat(real.(vec(modes)), imag.(vec(modes)))))
    append!(row, flat(sphere_modes(w.ex, p, u)))
    if retarded !== nothing
        K = length(p.models)
        vals = Vector{Float64}(undef, length(w.ex.points))
        Threads.@threads for j in eachindex(vals)
            vals[j] = sum(k -> retarded_field(path, k, p.models[k].Q,
                                              w.ex.points[j], t)[1], 1:K)
        end
        append!(row, flat(sphere_modes(w.ex, vals)))
    end
    println(w.io, join(row, '\t'))
    flush(w.io)
    return nothing
end

# `--flux=FILE`: the energy flux `-∮ Π ∂_rΦ dA` out through spheres about
# the centre (`sphere_flux`, "Waveforms" in `CODE.md`), one row per output
# time, and beside it the retarded field's flux at the same points when
# the data are the retarded field. The spheres must lie in the ball's cube
# (or the brick), where the gradient is read: `--fradii` (4,6,8,12), with
# `--fntheta` (24) Gauss points in `cos θ`; the drivers read
# `--fluxevery` (0.0625). Beyond `R₂` the remainder is the field, so the
# flux is the field's. `bin/flux.jl` reads the file.
function flux_writer(p, c, retarded)
    file = option("flux", "")
    isempty(file) && return nothing
    fs = option("fradii", "4,6,8,12")
    radii = [parse(Float64, strip(r)) for r in split(fs, ",")]
    nθ = parse(Int, option("fntheta", "24"))
    ex = SphereExtraction(p.wave, radii; lmax=0, nθ, centre=c, gradient=true)
    io = open(file, "w")
    name(r) = isinteger(r) ? string(Int(r)) : string(r)
    cols = ["F_r$(name(r))" for r in radii]
    retarded === nothing || append!(cols, "ref_" .* cols)
    println(io, join(("t", cols...), '\t'))
    flush(io)
    return (; ex, io)
end

# One row of `--flux`: the state's flux through each sphere, and the
# retarded field's, of the sum of every particle's field.
function write_flux(w, p, u, t, retarded)
    w === nothing && return nothing
    row = [t]
    append!(row, sphere_flux(w.ex, p, u))
    if retarded !== nothing
        K = length(p.models)
        n = length(w.ex.points)
        Π = Vector{Float64}(undef, n)
        ∇Φ = Vector{SVector{3,Float64}}(undef, n)
        Threads.@threads for j in 1:n
            x = w.ex.points[j]
            Π[j] = sum(k -> retarded_field(p.motion.path, k, p.models[k].Q, x, t)[2],
                       1:K)
            ∇Φ[j] = sum(k -> retarded_gradient(p.motion.path, k, p.models[k].Q, x, t),
                        1:K)
        end
        append!(row, sphere_flux(w.ex, Π, ∇Φ))
    end
    println(w.io, join(row, '\t'))
    flush(w.io)
    return nothing
end

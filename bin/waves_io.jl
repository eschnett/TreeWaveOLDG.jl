# `--waves=FILE` of the particle drivers `bin/prescribed.jl` and
# `bin/orbit_movie.jl`: the field's modes `Φ_ℓm` on spheres about the
# centre (`SphereExtraction`, "Waveforms" in `CODE.md`), one row per
# output time, and beside them the retarded field's when the data are the
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
# sum of every particle's.
function write_waves(w, p, u, t, retarded)
    w === nothing && return nothing
    row = [t]
    flat(modes) = vec(permutedims(hcat(real.(vec(modes)), imag.(vec(modes)))))
    append!(row, flat(sphere_modes(w.ex, p, u)))
    if retarded !== nothing
        K = length(p.models)
        vals = Vector{Float64}(undef, length(w.ex.points))
        Threads.@threads for j in eachindex(vals)
            vals[j] = sum(k -> retarded_field(p.motion.path, k, p.models[k].Q,
                                              w.ex.points[j], t)[1], 1:K)
        end
        append!(row, flat(sphere_modes(w.ex, vals)))
    end
    println(w.io, join(row, '\t'))
    flush(w.io)
    return nothing
end

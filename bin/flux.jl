# The energy flux through spheres about a charge's circle
# (INSPIRAL-PLAN.md's M6b, "The flux check"; "Diagnostics" in `CODE.md`),
# in the package environment:
#
#     julia --project=. -t 4,0 bin/flux.jl [--v=0.05,0.1,0.2] [--rc=0.5] \
#         [--radii=4,6,8,12] [--ntheta=24]
#     julia --project=. bin/flux.jl [--v=0.2] [--rc=0.5] [--tmin=0] \
#         4=flux-h4.tsv 6=flux-h6.tsv 8=flux-h8.tsv
#
# Without files: the flux `-∮ Π ∂_rΦ dA` of the point charge
# `Q = √0.005` on `circular_motion`'s circle of radius `rc` about the
# origin at each speed `--v`, from its retarded field
# (`retarded_field`, `retarded_gradient`) at the points of
# `SphereExtraction`'s rule on spheres of `--radii`, against the dipole
# power `|d̈|²/12π = Q²a²/12π`, `a = v²/rc`. The circle's field rotates
# rigidly about the spheres' centre, so its flux is the same through every
# sphere and at every time, and the dipole formula holds to `O(v²)`:
# the table gives `F/P − 1` and that over `v²`. It is `γ⁴ − 1`: the
# flux is `γ⁴Q²a²/12π` (measured, 2026-11-20).
#
# With files, `HDIV=FILE` pairs of `bin/prescribed.jl --flux=FILE` runs
# from the retarded field: per radius, the flux's mean over the whole
# orbits from `tmin` (its circle is off the spheres' centre, so only the
# mean is the power),
# the retarded field's at the same points, the run's largest relative
# error against it and its order between successive files, and the
# retarded field's flux against the dipole power at `--v`, `--rc`.

using TreeWaveOLDG
using TreeWaveOLDG: SVector, sphere_rule, sphere_weights
using LinearAlgebra: dot
using Printf: @printf

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

numbers(s) = [parse(Float64, strip(x)) for x in split(s, ",")]

const Q = sqrt(0.005)

dipole_power(v, rc) = Q^2 * (v^2 / rc)^2 / 12π

# The retarded field's flux through the sphere of radius `R` about the
# origin, of the charge on the circle, at `t`.
function retarded_flux(path, R, nθ, t)
    _, dirs = sphere_rule(Float64, 0, nθ)
    w = sphere_weights(Float64, nθ)
    terms = zeros(length(dirs))
    Threads.@threads for q in eachindex(dirs)
        x = R * dirs[q]
        Π = retarded_field(path, 1, Q, x, t)[2]
        ∇Φ = retarded_gradient(path, 1, Q, x, t)
        terms[q] = w[q] * Π * dot(dirs[q], ∇Φ)
    end
    return -R^2 * sum(terms)
end

function series()
    rc = parse(Float64, option("rc", "0.5"))
    radii = numbers(option("radii", "4,6,8,12"))
    nθ = parse(Int, option("ntheta", "24"))
    println("# the retarded field's flux F against the dipole power P = Q²a²/12π")
    println("v\tR\tt\tF\tP\tF/P-1\t(F/P-1)/v^2")
    for v in numbers(option("v", "0.05,0.1,0.2"))
        path = circular_motion(SVector(0.0, 0.0, 0.0), rc, v).path
        P = dipole_power(v, rc)
        for R in radii, t in (0.0, 1.3)
            F = retarded_flux(path, R, nθ, t)
            @printf("%g\t%g\t%g\t%.10e\t%.10e\t%.4e\t%.4f\n", v, R, t, F, P, F / P - 1,
                    (F / P - 1) / v^2)
        end
    end
    return nothing
end

function readflux(file)
    lines = readlines(file)
    head = split(lines[1], '\t')
    rows = [parse.(Float64, split(l, '\t')) for l in lines[2:end] if !isempty(l)]
    return head, reduce(hcat, rows)'
end

function compare(files)
    v = parse(Float64, option("v", "0.2"))
    rc = parse(Float64, option("rc", "0.5"))
    tmin = parse(Float64, option("tmin", "0"))
    P = dipole_power(v, rc)
    period = 2π * rc / v
    @printf("# dipole power P = %.10e at v = %g, rc = %g; ", P, v, rc)
    @printf("whole orbits of %.4f from t = %g\n", period, tmin)
    println("hdiv\tradius\tmean F\tmean ref\tref/P-1\tmax |F-ref|/ref\torder")
    previous = Dict{String,Float64}()
    for (h, file) in files
        head, data = readflux(file)
        # Whole orbits: a sphere off the circle's centre sees the energy
        # inside it change as the charge goes round, so its flux is the
        # power only on average over an orbit.
        tlast = data[end, 1]
        norbits = floor(Int, (tlast - tmin) / period + 1e-9)
        norbits >= 1 || error("$file reaches t = $tlast, short of an orbit from $tmin")
        keep = tmin .<= data[:, 1] .< tmin + norbits * period
        for (j, name) in enumerate(head)
            startswith(name, "F_") || continue
            r = findfirst(==("ref_" * name), head)
            r === nothing && error("$file has no column ref_$name: not from retarded data")
            F, ref = data[keep, j], data[keep, r]
            e = maximum(abs.(F - ref) ./ abs.(ref))
            order = haskey(previous, name) ?
                    log(previous[name] / e) / log(h / previous[name * "h"]) : NaN
            F̄, ref̄ = sum(F) / length(F), sum(ref) / length(ref)
            @printf("%g\t%s\t%.10e\t%.10e\t%.4e\t%.3e\t%.2f\n", h, name[3:end], F̄,
                    ref̄, ref̄ / P - 1, e, order)
            previous[name] = e
            previous[name * "h"] = h
        end
    end
    return nothing
end

function main()
    files = [(parse(Float64, split(a, "=")[1]), split(a, "="; limit=2)[2])
             for a in ARGS if !startswith(a, "--")]
    isempty(files) ? series() : compare(files)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()

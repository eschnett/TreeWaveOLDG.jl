# Waveforms: the field's spherical-harmonic modes on spheres about a
# centre, read from the solution's own element polynomials (INSPIRAL-PLAN.md,
# M8e; `CODE.md`, "Waveforms").

"""
    ball_patch_point(geo::SevenPatchBall, x)

The patch and logical position of the physical point `x` of the
seven-patch ball, as a TreeAMR `PatchPoint`: the inverse of
[`physical_coordinates`](@ref). A point with every coordinate within
`Lc` of the origin is the cube's, patch 1, whose map is affine. Any
other lies in the shell of its largest coordinate `k` and that
coordinate's sign `s`, on the ray through its point `f = x Lc/|x_k|` on
the cube's face, since a shell's blend `(1 − ρ)f + ρR₁f/|f|` keeps the
direction of `f`: the tangential position is `f`'s, and the radial one
inverts `ρ = (|x| − |f|)/(R₁ − |f|)` and the grading in closed form. On a
face between patches the point is the cube's, or the first shell's in
`k`'s order; the solution is the same there to its own jump. A point
beyond the sphere is refused.
"""
function ball_patch_point(geo::SevenPatchBall{T}, x) where {T}
    X = SVector{3,T}(Tuple(x))
    n, m = geo.roots
    Lc, R1 = geo.Lc, geo.R1
    r = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
    r <= R1 * (1 + 8eps(T)) || throw(ArgumentError(
        "the point $(Tuple(X)) lies at |x| = $r, beyond the ball's sphere at R1 = $R1"))
    a = abs.(X)
    k = a[1] >= a[2] && a[1] >= a[3] ? 1 : a[2] >= a[3] ? 2 : 3
    if a[k] <= Lc
        ξ = clamp.((X .+ Lc) .* (n / (2Lc)), zero(T), T(n))
        return PatchPoint(Tuple(ξ), 1)
    end
    s = X[k] > 0 ? 1 : -1
    f = X .* (Lc / a[k])
    rf = sqrt(f[1]^2 + f[2]^2 + f[3]^2)
    ρ = clamp((r - rf) / (R1 - rf), zero(T), one(T))
    α = geo.grading
    η = iszero(α) ? ρ : log1p(ρ * expm1(α)) / α
    ξk = clamp(s == 1 ? m * η : m * (1 - η), zero(T), T(m))
    ξ = ntuple(j -> j == k ? ξk : clamp(n * (f[j] / Lc + 1) / 2, zero(T), T(n)), Val(3))
    return PatchPoint(ξ, sevenpatch_index(k, s))
end

"""
    spherical_harmonics(lmax, θ, φ)

The orthonormal spherical harmonics `Y_ℓm(θ, φ)` for `0 ≤ ℓ ≤ lmax` and
`0 ≤ m ≤ ℓ`, with the Condon–Shortley phase, in the order of
[`harmonic_modes`](@ref): `Y_ℓm = N_ℓm P_ℓ^m(cos θ) e^{imφ}`, the
normalized associated Legendre functions by the standard three-term
recurrence in `ℓ` from the sectoral `P_m^m`, stable to `ℓ` in the
hundreds. A real field's modes of negative `m` are
`Φ_{ℓ,−m} = (−1)^m conj(Φ_ℓm)`.
"""
function spherical_harmonics(lmax::Integer, θ::T, φ::T) where {T<:AbstractFloat}
    x, y = cos(θ), sin(θ)
    out = Vector{Complex{T}}(undef, (lmax + 1) * (lmax + 2) ÷ 2)
    pmm = sqrt(1 / (4 * T(π)))
    for m in 0:lmax
        m > 0 && (pmm *= -sqrt(T(2m + 1) / T(2m)) * y)
        e = cis(m * φ)
        out[mode_index(m, m)] = pmm * e
        m == lmax && continue
        p1 = sqrt(T(2m + 3)) * x * pmm
        out[mode_index(m + 1, m)] = p1 * e
        p0 = pmm
        for l in m+2:lmax
            a = sqrt(T(4l^2 - 1) / T(l^2 - m^2))
            b = sqrt(T((l - 1)^2 - m^2) / T(4(l - 1)^2 - 1))
            p0, p1 = p1, a * (x * p1 - b * p0)
            out[mode_index(l, m)] = p1 * e
        end
    end
    return out
end

"""
    harmonic_modes(lmax)

The modes `(ℓ, m)`, `0 ≤ m ≤ ℓ ≤ lmax`, in the order in which
[`spherical_harmonics`](@ref) and [`sphere_modes`](@ref) return them:
`ℓ` by `ℓ`, and `m` from 0 to `ℓ` within each.
"""
harmonic_modes(lmax::Integer) = [(l, m) for l in 0:lmax for m in 0:l]

@inline mode_index(l, m) = l * (l + 1) ÷ 2 + m + 1

"""
    sphere_rule(T, lmax, nθ)

The quadrature on the unit sphere that [`SphereExtraction`](@ref) uses:
Gauss–Legendre in `cos θ` at `nθ` points times `nφ = 2nθ` equally spaced
`φ`, point `q = (i − 1)nφ + j` at the `i`-th `θ` and the `j`-th `φ`.
Returns the table `w_q conj(Y_ℓm)`, one row per mode of
[`harmonic_modes`](@ref)`(lmax)` and one column per point, so that the
table times the values at the points is their modes, and the points'
directions. Built in `Float64` at least and rounded to `T` once.
"""
function sphere_rule(::Type{T}, lmax::Integer, nθ::Integer) where {T}
    nφ = 2nθ
    xg, wg = gauss_nodes_weights(nθ)
    S = promote_type(T, Float64)
    θs = [acos(S(x)) for x in xg]
    φs = [2S(π) * (j - 1) / nφ for j in 1:nφ]
    table = Matrix{Complex{T}}(undef, (lmax + 1) * (lmax + 2) ÷ 2, nθ * nφ)
    dirs = Vector{SVector{3,S}}(undef, nθ * nφ)
    for (i, θ) in enumerate(θs), (j, φ) in enumerate(φs)
        q = (i - 1) * nφ + j
        w = S(wg[i]) * 2S(π) / nφ
        table[:, q] = Complex{T}.(w .* conj.(spherical_harmonics(lmax, θ, φ)))
        dirs[q] = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
    end
    return table, dirs
end

"""
    SphereExtraction(wave::WaveProblem, radii; lmax = 6, nθ = 2lmax + 12,
                     centre = (0, 0, 0))

The points at which [`sphere_modes`](@ref) reads the field of `wave` on
spheres of `radii` about `centre`, and the quadrature that turns the
values there into the modes `Φ_ℓm`, `0 ≤ m ≤ ℓ ≤ lmax`. On each sphere:
Gauss–Legendre in `cos θ` at `nθ` points times `nφ = 2nθ` equally spaced
`φ`, which integrates `Y_ℓm Ȳ_ℓ′m′` exactly for `ℓ + ℓ′ < 2nθ`, so a
field's modes up to `lmax` alias only its modes from `2nθ − lmax` on.

The points are mapped to the forest once: on the seven-patch ball by
[`ball_patch_point`](@ref), on a brick as they are. TreeAMR locates them
on the host and gives each one's element and its weights there, those of
its `ElementPolynomial`, the element's own degree-`P` polynomial through
its Gauss–Lobatto nodes; what `ex` keeps on the problem's backend is, for
each point, the `(P+1)^D` positions of those nodes in a state vector and
their weights, so that a read is a gather and a weighted sum, plain array
operations on any backend ([`sphere_values`](@ref)). A regrid makes them
stale. Other geometries, a point TreeAMR would fold across a
reflecting face or a seam, a point beyond a periodic brick, which would
read its image, and a distributed forest are refused. `points` holds the physical
points on the host, sphere by sphere, so that a reference's modes are
the same quadrature of its values there (`sphere_modes(ex, values)`).
"""
struct SphereExtraction{T,IA,WA}
    radii::Vector{T}
    centre::SVector{3,T}
    lmax::Int
    nθ::Int
    nφ::Int
    points::Vector{SVector{3,T}}
    table::Matrix{Complex{T}}    # w_j conj(Y_ℓm) at one sphere's points
    index::IA                    # each point's nodes in a state vector, var 1
    weights::WA                  # and their weights, (P+1)^D per point
    varstride::Int               # a variable's offset in a state vector's block
    statelength::Int
end

function SphereExtraction(wave::WaveProblem{T}, radii; lmax::Integer=6,
                          nθ::Integer=2lmax + 12, centre=(0, 0, 0)) where {T}
    lmax >= 0 || throw(ArgumentError("lmax must be at least 0, got $lmax"))
    nθ > lmax || throw(ArgumentError(
        "nθ = $nθ Gauss points in cos θ do not integrate the modes up to lmax = $lmax " *
        "exactly; nθ must exceed lmax"))
    rs = T.(collect(radii))
    all(>(0), rs) || throw(ArgumentError("every radius must be positive, got $radii"))
    c = SVector{3,T}(Tuple(centre))
    table, dirs = sphere_rule(T, lmax, nθ)
    nφ = 2nθ
    points = [SVector{3,T}(c .+ r .* d) for r in rs for d in dirs]
    geo = wave.metric === nothing ? nothing : wave.metric.geometry
    xs = if geo isa SevenPatchBall
        [ball_patch_point(geo, x) for x in points]
    elseif geo === nothing
        [Tuple(x) for x in points]
    else
        throw(ArgumentError(
            "SphereExtraction maps physical points to the forest on a brick or the " *
            "seven-patch ball only, not on a $(nameof(typeof(geo)))"))
    end
    index, weights = element_stencils(wave.fs, xs)
    backend = get_backend(wave.fs.work)
    N = wave.fs.forest.N
    D = length(first(xs) isa PatchPoint ? first(xs).ξ : first(xs))
    return SphereExtraction(rs, c, Int(lmax), Int(nθ), nφ, points, table,
                            to_backend(backend, index), to_backend(backend, weights),
                            N^D, statelength(wave.fs))
end

# Each point's element and weights, as TreeAMR's `ElementPolynomial`
# evaluates it from a state vector (`interpolate(fs, u, xs,
# ElementPolynomial())`): the point located on the host, its block the
# located leaf, its element and the offset in it by `query_stencil`, the
# weights by `basisweights`; then the linear positions of the element's
# `n^D` nodes in the state vector's `(N, …, N, nvars, nblocks)` layout
# (`statearray`), for variable 1, and the products of the weights, one
# column per point. TreeAMR's kernel reads the same nodes with the same
# weights, summed in another order (see "Waveforms" in `CODE.md` for why
# the extraction does not call it).
function element_stencils(fs::FieldSet{T,D}, xs) where {T,D}
    forest = fs.forest
    # A located leaf is the forest's, not a rank's block ("Running
    # distributed"); the tables below index blocks.
    TreeAMR.isdistributed(forest) && throw(ArgumentError(
        "the extraction reads a forest on one rank only, not a distributed one"))
    pts = TreeAMR.locate(T, forest, xs)
    all(pts.inside) || throw(ArgumentError("a point to extract lies outside the forest"))
    # A periodic brick locates a point beyond it at its image, which no
    # fold code marks; a physical point must be located where it is.
    all(j -> !(xs[j] isa Tuple) || SVector(pts.y[j]) == SVector(xs[j]), eachindex(xs)) ||
        throw(ArgumentError(
            "a point to extract lies beyond a periodic brick, which would read its " *
            "periodic image"))
    all(iszero, pts.code) || throw(ArgumentError(
        "a point to extract is folded across a reflecting face or a seam, whose " *
        "parity the extraction does not apply"))
    basis = TreeAMR.specialize_basis(ElementPolynomial(), fs)
    n = TreeAMR.stencilwidth(basis)
    N = forest.N
    origins = TreeAMR.block_origins(forest, T)
    spacings = TreeAMR.block_spacings(forest, T)
    G, C, S = ntuple(_ -> 0, D), TreeAMR.staggers(fs), ntuple(_ -> N, D)
    st = ntuple(d -> N^(d - 1), D)
    block = N^D * fs.nvars
    index = Matrix{Int}(undef, n^D, length(xs))
    weights = Matrix{T}(undef, n^D, length(xs))
    for j in eachindex(xs)
        b = Int(pts.leaf[j])
        first, ξ, _ = TreeAMR.query_stencil(basis, pts.y[j], origins[b], spacings[b],
                                            Val(G), Val(C), Val(S))
        W = ntuple(d -> TreeAMR.basisweights(basis, ξ[d], Val(0))[1], D)
        corner = 1 + sum(ntuple(d -> (first[d] - 1) * st[d], D)) + (b - 1) * block
        for (q, I) in enumerate(CartesianIndices(ntuple(_ -> n, D)))
            index[q, j] = corner + sum(ntuple(d -> (I[d] - 1) * st[d], D))
            weights[q, j] = prod(ntuple(d -> W[d][I[d]], D))
        end
    end
    return index, weights
end

"""
    sphere_values(ex::SphereExtraction, wave::WaveProblem, u; var = 1)
    sphere_values(ex::SphereExtraction, prob::InspiralProblem, u; var = 1)

The values of variable `var` of the state vector `u` (1 for `u`, 2 for
`v`; for an inspiral, of the field's part of its state) at the points
of `ex`, on the host, in the order of `ex.points`: the containing
element's own degree-`P` polynomial there, TreeAMR's `ElementPolynomial`,
which reads no ghosts. On `u`'s backend, a gather of each point's nodes
and a weighted sum.
"""
function sphere_values(ex::SphereExtraction, wave::WaveProblem, u::AbstractVector;
                       var::Integer=1)
    return stencil_values(ex, u, var)
end

# The gather of `sphere_values` and `plane_values`: each point's nodes of
# variable `var`, weighted and summed, on `u`'s backend, copied to the
# host.
function stencil_values(ex, u::AbstractVector, var::Integer)
    length(u) >= ex.statelength || throw(DimensionMismatch(
        "a state vector of $(length(u)) entries for a field set of $(ex.statelength)"))
    nodes = u[ex.index .+ (var - 1) * ex.varstride]
    return vec(Array(sum(ex.weights .* nodes; dims=1)))
end

# The field's part of an inspiral's state comes first in it, so the
# positions are the same in the whole state vector.
sphere_values(ex::SphereExtraction, prob::InspiralProblem, u::AbstractVector; kwargs...) =
    sphere_values(ex, prob.wave, u; kwargs...)

"""
    sphere_modes(ex::SphereExtraction, values)
    sphere_modes(ex::SphereExtraction, wave_or_prob, u; var = 1)

The modes `Φ_ℓm` of `values` at `ex.points`, a matrix with one row per
mode of [`harmonic_modes`](@ref)`(ex.lmax)` and one column per radius;
or of the state `u`, read by [`sphere_values`](@ref).
"""
function sphere_modes(ex::SphereExtraction{T}, values::AbstractVector) where {T}
    np = size(ex.table, 2)
    length(values) == np * length(ex.radii) || throw(DimensionMismatch(
        "$(length(values)) values for $(length(ex.radii)) spheres of $np points each"))
    return ex.table * reshape(Complex{T}.(values), np, length(ex.radii))
end

sphere_modes(ex::SphereExtraction, wave, u::AbstractVector; kwargs...) =
    sphere_modes(ex, sphere_values(ex, wave, u; kwargs...))

"""
    PlaneExtraction(wave::WaveProblem, centre, halfwidth, n)

The points of an `n × n` grid in the plane `z = centre[3]`, the square
of half-width `halfwidth` about `centre`, at the centres of its `n²`
cells, `x_i = centre[1] + halfwidth (2i - 1 - n)/n` and the same in `y`;
and how to read a state there, as [`SphereExtraction`](@ref) reads its
spheres, on any backend ([`plane_values`](@ref)). On the seven-patch
ball the points beyond the sphere are left out, on a brick those
outside the forest; `inside` marks the rest, and `points` holds every
grid point on the host, column by column, `x` running fastest. A
periodic brick shows once, its images left out. A regrid
makes it stale. Other geometries, and a distributed forest, are refused. It is for viewing the
field (`bin/orbit_movie.jl`; "The movie" in `CODE.md`).
"""
struct PlaneExtraction{T,IA,WA}
    centre::SVector{3,T}
    halfwidth::T
    n::Int
    points::Vector{SVector{3,T}}
    inside::Vector{Bool}
    index::IA                    # each inside point's nodes in a state vector, var 1
    weights::WA                  # and their weights, (P+1)^D per point
    varstride::Int
    statelength::Int
end

# The cell centres of an `n × n` grid of half-width `h` about `c` in the
# plane `z = c[3]`, column by column, `x` running fastest.
function plane_grid(c::SVector{3,T}, h::T, n::Integer) where {T}
    s = [h * T(2i - 1 - n) / n for i in 1:n]
    return [SVector{3,T}(c[1] + s[i], c[2] + s[j], c[3]) for j in 1:n for i in 1:n]
end

function PlaneExtraction(wave::WaveProblem{T}, centre, halfwidth::Real,
                         n::Integer) where {T}
    halfwidth > 0 || throw(ArgumentError("the half-width must be positive, got $halfwidth"))
    n >= 1 || throw(ArgumentError("the grid needs at least one point, got n = $n"))
    c = SVector{3,T}(Tuple(centre))
    h = T(halfwidth)
    points = plane_grid(c, h, n)
    geo = wave.metric === nothing ? nothing : wave.metric.geometry
    if geo isa SevenPatchBall
        inside = [sqrt(x[1]^2 + x[2]^2 + x[3]^2) < geo.R1 for x in points]
        xs = [ball_patch_point(geo, x) for x in points[inside]]
    elseif geo === nothing
        # A periodic brick wraps a point outside it to its image, which the
        # plane leaves out, so that the brick shows once.
        pts = TreeAMR.locate(T, wave.fs.forest, [Tuple(x) for x in points])
        inside = [pts.inside[j] && iszero(pts.code[j]) &&
                  SVector{3,T}(pts.y[j]) == points[j] for j in eachindex(points)]
        xs = [Tuple(x) for x in points[inside]]
    else
        throw(ArgumentError(
            "PlaneExtraction maps physical points to the forest on a brick or the " *
            "seven-patch ball only, not on a $(nameof(typeof(geo)))"))
    end
    isempty(xs) && throw(ArgumentError("no point of the plane lies in the forest"))
    index, weights = element_stencils(wave.fs, xs)
    backend = get_backend(wave.fs.work)
    N = wave.fs.forest.N
    return PlaneExtraction(c, h, Int(n), points, inside, to_backend(backend, index),
                           to_backend(backend, weights), N^3, statelength(wave.fs))
end

"""
    plane_values(ex::PlaneExtraction, wave::WaveProblem, u; var = 1)
    plane_values(ex::PlaneExtraction, prob::InspiralProblem, u, t)

The values of variable `var` of the state `u` on the plane's grid, an
`n × n` matrix on the host, `NaN` at the points outside the forest.
For an inspiral, the total field `Φ` and its rate `Π = ∂ₜΦ` instead,
`(Φ, Π)`: the remainder's two variables, plus `W Σ_k Ψ̂_k` and
`W Σ_k 𝒟Ψ̂_k` at the points where the window is not zero, evaluated on
the host, along the particles' trajectories of the state at `t`.
"""
function plane_values(ex::PlaneExtraction{T}, wave::WaveProblem, u::AbstractVector;
                      var::Integer=1) where {T}
    out = fill(T(NaN), ex.n, ex.n)
    out[ex.inside] = stencil_values(ex, u, var)
    return out
end

function plane_values(ex::PlaneExtraction{T}, p::InspiralProblem{T,K}, u::AbstractVector,
                      t) where {T,K}
    Φ = plane_values(ex, p.wave, u; var=1)
    Π = plane_values(ex, p.wave, u; var=2)
    trajs = trajectories(p, u, t)
    w = p.window
    near = [j for j in eachindex(ex.points)
            if ex.inside[j] && !iszero(w(ex.points[j]))]
    Threads.@threads for j in near
        x = ex.points[j]
        W = w(x)
        for k in 1:K
            m, tr = p.models[k], trajs[k]
            Φ[j] += W * field_of_order(m, x, tr, p.order)
            Π[j] += W * rate_of_order(m, x, tr, p.order)
        end
    end
    return Φ, Π
end

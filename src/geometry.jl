# The map from a forest's logical coordinates to physical space.
#
# TreeAMR's coordinates are logical, patch by patch: on a forest of
# patches (its M17) the map to physical space, with its Jacobian and its
# metric, is the application's. A `Geometry` is that map, for metrics that
# are diagonal in the logical coordinates and whose volume element is a
# product of one-dimensional factors, which is what an annulus in polar
# coordinates and a shell in spherical ones have. With it the wave
# equation is
#
#     ∂ₜu = v
#     √g ∂ₜv = Σ_d ∂_d (√g g^{dd} ∂_d u)
#
# discretized by the same wide derivative, applied to the flux
# `F_d = A_d D_d u` with `A_d = √g g^{dd}` at the nodes. See "Multi-patch
# forests" in `CODE.md`.

"""
    Geometry

A map from the logical coordinates of a forest's patches to physical
space, whose metric is diagonal in them. A geometry is `isbits`, so that it
can be a kernel and a fold argument, and provides:

- `geometry_terms(geo, x, p) -> (J, A)`: at the logical position `x` of
  patch `p`, the volume element `J = √g` and the tuple `A` of the flux
  coefficients `A_d = √g g^{dd}`, one per dimension;
- `jacobian_factor(geo, Val(d), x_d, p)`: the factor of `J` that depends
  on `x_d` alone, `J = ∏_d jacobian_factor(geo, Val(d), x_d, p)`. The
  truncation is orthogonal in the `J`-weighted inner product, and is a
  tensor product of one-dimensional projections only because `J`
  factorizes ("Truncation with a metric" in `CODE.md`). It is evaluated
  in `BigFloat` when the projections are built;
- `physical_coordinates(geo, x, p)`: the physical position;
- `check_geometry(geo, forest)`: refuse a forest the map does not fit.

[`PolarAnnulus`](@ref) is the one implemented.
"""
abstract type Geometry end

"""
    PolarAnnulus{T}(r0, r1, roots)
    PolarAnnulus(r0, r1, roots)

The annulus `r0 ≤ r ≤ r1` as four patches of `roots = (n₁, n₂)` roots
each, the angle along dimension 1 and the radius along dimension 2, as
[`annulus_forest`](@ref) builds them: patch `p`'s logical box
`[0, n₁] × [0, n₂]` maps to

    θ = (p - 1)π/2 + θ′x₁,    θ′ = (π/2)/n₁
    r = r0 + r′x₂,            r′ = (r1 - r0)/n₂

so that `√g = r θ′ r′`, `A₁ = r′/(r θ′)` and `A₂ = r θ′/r′`. The
constants are built in `BigFloat` and rounded to `T` once.
"""
struct PolarAnnulus{T} <: Geometry
    r0::T
    r1::T
    roots::NTuple{2,Int}
    dθ::T                        # θ′
    dr::T                        # r′
    halfπ::T
end

function PolarAnnulus{T}(r0, r1, roots::NTuple{2,Integer}) where {T}
    0 < r0 < r1 || throw(ArgumentError(
        "an annulus needs 0 < r0 < r1, got r0 = $r0 and r1 = $r1"))
    all(>(0), roots) || throw(ArgumentError(
        "every patch of the annulus needs a positive number of roots along each " *
        "dimension, got $roots"))
    n = Int.(roots)
    dθ, dr, halfπ = setprecision(BigFloat, 512) do
        big(π) / 2 / n[1], (big(r1) - big(r0)) / n[2], big(π) / 2
    end
    return PolarAnnulus{T}(T(r0), T(r1), n, T(dθ), T(dr), T(halfπ))
end

PolarAnnulus(r0, r1, roots) = PolarAnnulus{float(promote_type(typeof(r0), typeof(r1)))}(
    r0, r1, roots)

geometry_npatches(::PolarAnnulus) = 4

@inline annulus_radius(geo::PolarAnnulus, x2) = geo.r0 + geo.dr * x2
@inline annulus_angle(geo::PolarAnnulus, x1, p) = (p - 1) * geo.halfπ + geo.dθ * x1

@inline function geometry_terms(geo::PolarAnnulus, x, p)
    r = annulus_radius(geo, x[2])
    J = r * geo.dθ * geo.dr
    return J, (geo.dr / (r * geo.dθ), r * geo.dθ / geo.dr)
end

@inline jacobian_factor(geo::PolarAnnulus, ::Val{1}, x1, p) = oftype(x1, geo.dθ)
@inline jacobian_factor(geo::PolarAnnulus, ::Val{2}, x2, p) =
    annulus_radius(geo, x2) * geo.dr

@inline function physical_coordinates(geo::PolarAnnulus, x, p)
    r = annulus_radius(geo, x[2])
    θ = annulus_angle(geo, x[1], p)
    return (r * cos(θ), r * sin(θ))
end

function check_geometry(geo::PolarAnnulus, forest::Forest{D}) where {D}
    D == 2 || throw(ArgumentError(
        "the annulus is two-dimensional, and the forest has D = $D"))
    TreeAMR.npatches(forest) == geometry_npatches(geo) || throw(ArgumentError(
        "the annulus has $(geometry_npatches(geo)) patches, and the forest " *
        "$(TreeAMR.npatches(forest)): build both with annulus_forest"))
    for p in 1:geometry_npatches(geo)
        roots = Tuple(TreeAMR.patch_roots(forest, p))
        ext = TreeAMR.patch_extents(forest, p)
        roots == geo.roots &&
            all(d -> ext[d][1] == 0 && ext[d][2] == geo.roots[d], 1:2) ||
            throw(ArgumentError(
                "patch $p of the forest has roots $roots and extents $ext, and the " *
                "annulus maps the box [0, n₁] × [0, n₂] of roots $(geo.roots): build " *
                "both with annulus_forest"))
    end
    return nothing
end

"""
    annulus_forest(T, P, E; roots=(2, 1), r0=1, r1=2) -> (forest, geometry)

The annulus `r0 ≤ r ≤ r1` as TreeAMR's ring of four patches, each of
`roots` roots holding `E²` elements of order `P` per block, and its
[`PolarAnnulus`](@ref) map. Each patch's high face of dimension 1 is glued
to the low face of dimension 1 of the next, by the identity, both
directions listed; the radial faces are solid (Neumann) walls, TreeAMR
reflection symmetries, as on a reflecting box. `E` must be even, as for
[`wave_forest`](@ref).

Hand both to [`WaveProblem`](@ref): `WaveProblem(forest, el; geometry)`.
"""
function annulus_forest(::Type{T}, P::Integer, E::Integer; roots=(2, 1), r0=1,
                        r1=2) where {T}
    N = E * (P + 1)
    iseven(E) || throw(ArgumentError(
        "E = $E elements of order P = $P per block is odd, and E must be even: " *
        "TreeAMR's element centering needs N = $N to be a multiple of " *
        "2(P+1) = $(2(P + 1))"))
    glue = reduce(vcat, [[(from=(p, 1, 2), to=(mod1(p + 1, 4), 1, 1), map=(1, 2)),
                          (from=(mod1(p + 1, 4), 1, 1), to=(p, 1, 2), map=(1, 2))]
                         for p in 1:4])
    forest = Forest{T}([(roots=Tuple(roots),) for _ in 1:4]; N=N, glue=glue,
                       reflecting=((false, false), (true, true)))
    return forest, PolarAnnulus{T}(r0, r1, Tuple(roots))
end

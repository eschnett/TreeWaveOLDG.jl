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
- `check_geometry(geo, forest)`: refuse a forest the map does not fit;
- optionally `affine_patch(geo, p)`: whether patch `p`'s map is affine,
  so that its metric is constant there; `false` by default. Only such a
  patch may be refined ([`affine_patch`](@ref TreeWaveOLDG.affine_patch)).

[`PolarAnnulus`](@ref) is the one implemented. A map whose metric is not
diagonal, or whose `√g` does not factorize, is a [`FullGeometry`](@ref).
"""
abstract type Geometry end

"""
    FullGeometry <: Geometry

A map from the logical coordinates of a forest's patches to physical
space with a general metric: neither diagonal in the logical coordinates
nor with a volume element that factorizes, as a cubed sphere's patches
have. A full geometry is `isbits` and provides:

- `geometry_frame(geo, x, p) -> (J, S)`: at the logical position `x` of
  patch `p`, the volume element `J = √g` and the area vectors
  `S[d] = √g ∇ξ^d`, `S[d][i] = √g ∂ξ^d/∂x^i`, one `D`-tuple of Cartesian
  components per logical dimension `d`. It must be generic in the number
  type of `x`, since the truncation's tables evaluate it in `Float64x2` or
  `BigFloat` (`table_type`);
- `physical_coordinates(geo, x, p)`: the physical position;
- `check_geometry(geo, forest)`: refuse a forest the map does not fit;
- `geometry_boundary(geo)`: whether the forest has outer faces, which
  then take Dirichlet data ([`WaveProblem`](@ref)'s `boundary`);
- optionally `affine_patch(geo, p)`, as for a [`Geometry`](@ref).

On a full geometry the gradient is carried in Cartesian components, one
field set of `D` variables with a ghost plane on every face, because a
glue that swaps axes carries a variable onto itself, and a Cartesian
component is the same number on both sides; the logical flux
`F^d = S[d] ⋅ ∇u` is formed from it where the Laplacian reads it,
ghosts included. The truncation is the `√g`-orthogonal projection onto
`Q_{P-1}`, built per element. See "The seven-patch ball" in `CODE.md`.
[`SevenPatchBall`](@ref) is the one implemented.
"""
abstract type FullGeometry <: Geometry end

geometry_boundary(::FullGeometry) = true

"""
    affine_patch(geo, p) -> Bool

Whether patch `p`'s map to physical space is affine, so that `√g`, the
flux coefficients and the frame are constants there. A geometry takes a
refined forest only where this holds: across a coarse-fine face the
coarse ghost is the adjoint of the fine one in the flat weights, which
are the `√g`-weighted ones up to a constant only where `√g` is constant,
so only there does the energy stay an exact invariant; and there the
element's own projector is the `√g`-orthogonal truncation. `false` unless
a geometry says otherwise. See "The seven-patch ball" in `CODE.md`.
"""
affine_patch(geo::Geometry, p) = false

# A full geometry's gradient is Cartesian, so its folds weight by `√g`
# alone, the potential energy's `√g |∇u|²` included (`layout.jl`).
@inline metric_factor(geo::FullGeometry, x, p, which) = first(geometry_frame(geo, x, p))

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

"""
    SevenPatchBall{T}(Lc, R1, roots)
    SevenPatchBall(Lc, R1, roots)

The ball `|x| ≤ R1` as seven patches, as [`sevenpatch_forest`](@ref)
builds them, with `roots = (n, m)`:

- patch 1, the cube `[-Lc, Lc]³`, of `(n, n, n)` roots: its logical box
  `[0, n]³` maps to `x = Lc (2ξ/n - 1)`;
- patches 2–7, the shells over the cube's faces `(k, s)` in the order
  `(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)`, each of `m` roots
  along its radial dimension `k` and `n` along the other two. Its logical
  box maps to

      c_k = s,   c_j = 2ξ_j/n - 1 (j ≠ k),   f = Lc c
      ρ = ξ_k/m (s = 1),  1 - ξ_k/m (s = -1)
      x = (1 - ρ) f + ρ R1 f/|f|

  so that its inner face is the cube's face, flat, and its outer face the
  sphere `|x| = R1`. Each logical dimension `j` of a shell runs along
  `x_j`, and `ξ_k` increases outward for `s = 1` and inward for `s = -1`,
  so that every cube–shell glue is the identity.

The shells' metric is not diagonal, and `√g` does not factorize, so this
is a [`FullGeometry`](@ref). `R1 > √3 Lc`, so that the cube's corners lie
inside the sphere and the map is one to one.
"""
struct SevenPatchBall{T} <: FullGeometry
    Lc::T
    R1::T
    roots::NTuple{2,Int}

    function SevenPatchBall{T}(Lc, R1, roots::NTuple{2,Integer}) where {T}
        0 < Lc || throw(ArgumentError(
            "the cube's half-width Lc must be positive, got $Lc"))
        R1 > sqrt(3) * Lc || throw(ArgumentError(
            "the outer radius R1 = $R1 must exceed √3 Lc = $(sqrt(3) * Lc), so that " *
            "the cube's corners lie inside the sphere and every shell maps one to one"))
        all(>(0), roots) || throw(ArgumentError(
            "every patch of the ball needs a positive number of roots along each " *
            "dimension, got $roots"))
        return new{T}(T(Lc), T(R1), Int.(Tuple(roots)))
    end
end

SevenPatchBall(Lc, R1, roots) =
    SevenPatchBall{float(promote_type(typeof(Lc), typeof(R1)))}(Lc, R1, roots)

# The shells' cube faces `(k, s)`, in patch order 2–7.
const SEVENPATCH_SHELLS = ((1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))

sevenpatch_index(k, s) = 1 + 2(k - 1) + (s == 1 ? 2 : 1)

# The physical position of logical `ξ` (an `SVector{3}`) of patch `p`, in
# `ξ`'s number type, so that `ForwardDiff` differentiates it and the
# truncation's tables evaluate it in `Float64x2` or `BigFloat`.
@inline function sevenpatch_position(geo::SevenPatchBall, ξ, p)
    n, m = geo.roots
    Lc = geo.Lc
    if p == 1
        return Lc .* (2 .* ξ ./ n .- 1)
    end
    k, s = SEVENPATCH_SHELLS[p - 1]
    f = SVector(ntuple(j -> j == k ? s * Lc : Lc * (2 * ξ[j] / n - 1), Val(3)))
    ρ = s == 1 ? ξ[k] / m : 1 - ξ[k] / m
    return (1 - ρ) .* f .+ (ρ * geo.R1 / sqrt(f[1]^2 + f[2]^2 + f[3]^2)) .* f
end

outer_radius(geo::SevenPatchBall) = geo.R1

# The cube's map `x = Lc(2ξ/n - 1)` is affine; the shells' are not.
affine_patch(::SevenPatchBall, p) = p == 1

@inline function physical_coordinates(geo::SevenPatchBall, x, p)
    return Tuple(sevenpatch_position(geo, SVector(x[1], x[2], x[3]), p))
end

@inline function geometry_frame(geo::SevenPatchBall, x, p)
    ξ = SVector(x[1], x[2], x[3])
    Jm = ForwardDiff.jacobian(y -> sevenpatch_position(geo, y, p), ξ)
    J = abs(det(Jm))
    Ji = inv(Jm)
    S = ntuple(d -> ntuple(i -> J * Ji[d, i], Val(3)), Val(3))
    return J, S
end

# The glue of the seven-patch ball, both directions of every gluing: the
# cube's face `(k, s)` onto the inner face of shell `(k, s)` by the
# identity, and shell `(k, s)`'s face towards shell `(l, t)` onto that
# one's face towards `(k, s)`, by the quarter turn that carries each
# one's radial dimension onto the other's and each one's normal onto the
# other's, reversed where the step out of the one is a step into the
# other.
function sevenpatch_glue()
    side(s) = s == 1 ? 2 : 1
    glue = NamedTuple{(:from, :to, :map),
                      Tuple{NTuple{3,Int},NTuple{3,Int},NTuple{3,Int}}}[]
    for (k, s) in SEVENPATCH_SHELLS
        p = sevenpatch_index(k, s)
        push!(glue, (from=(1, k, side(s)), to=(p, k, side(-s)), map=(1, 2, 3)))
        push!(glue, (from=(p, k, side(-s)), to=(1, k, side(s)), map=(1, 2, 3)))
        for (l, t) in SEVENPATCH_SHELLS
            l == k && continue
            σ = ntuple(e -> e == k ? (s == t ? l : -l) :
                            e == l ? (s == t ? -k : k) : e, 3)
            push!(glue, (from=(p, l, side(t)), to=(sevenpatch_index(l, t), k, side(s)),
                         map=σ))
        end
    end
    return glue
end

sevenpatch_roots(geo::SevenPatchBall, p) =
    p == 1 ? ntuple(_ -> geo.roots[1], 3) :
    ntuple(j -> j == SEVENPATCH_SHELLS[p - 1][1] ? geo.roots[2] : geo.roots[1], 3)

function check_geometry(geo::SevenPatchBall, forest::Forest{D}) where {D}
    D == 3 || throw(ArgumentError(
        "the seven-patch ball is three-dimensional, and the forest has D = $D"))
    TreeAMR.npatches(forest) == 7 || throw(ArgumentError(
        "the seven-patch ball has 7 patches, and the forest " *
        "$(TreeAMR.npatches(forest)): build both with sevenpatch_forest"))
    for p in 1:7
        roots = Tuple(TreeAMR.patch_roots(forest, p))
        ext = TreeAMR.patch_extents(forest, p)
        want = sevenpatch_roots(geo, p)
        roots == want && all(d -> ext[d][1] == 0 && ext[d][2] == want[d], 1:3) ||
            throw(ArgumentError(
                "patch $p of the forest has roots $roots and extents $ext, and the " *
                "seven-patch ball maps the box [0, $(want[1])] × [0, $(want[2])] × " *
                "[0, $(want[3])] there: build both with sevenpatch_forest"))
    end
    norm_entry(g) = (Int.(Tuple(g.from)), Int.(Tuple(g.to)), Int.(Tuple(g.map)))
    Set(map(norm_entry, TreeAMR.patch_keywords(forest).glue)) ==
        Set(map(norm_entry, sevenpatch_glue())) || throw(ArgumentError(
            "the forest's patches are not glued as the seven-patch ball's: build both " *
            "with sevenpatch_forest"))
    return nothing
end

"""
    sevenpatch_forest(T, P, E; roots=(2, 2), Lc=1, R1=3, refined=false)
        -> (forest, geometry)

The ball `|x| ≤ R1` as TreeAMR's forest of seven patches, a cube of
half-width `Lc` and six shells around it (see [`SevenPatchBall`](@ref)),
each block holding `E³` elements of order `P`, and its map. Every patch
has the one logical spacing of a root, `1`. The cube is glued to the
shells' inner faces by the identity, and the shells to each other by
quarter turns, so the forest has patch corners, which an element set
never reaches; the outer sphere is the only unglued face, and takes
Dirichlet data ([`WaveProblem`](@ref)'s `boundary`). `E` must be even, as
for [`wave_forest`](@ref).

The ball is uniform by default. `refined` refines the cube, whose map is
affine, as [`wave_forest`](@ref)'s `refined` refines a box: `true` the
middle half, `(n/4, 3n/4)³`, a tuple of three intervals one region, or a
vector of such tuples one region per level, all in the cube's logical
coordinates `[0, n]³`, `n = roots[1]`. So `nested_cubes(Val(3), n/2^levels,
levels)` gives the inspiral's nested cubes inside the ball, with
`n % 4 == 0`. TreeAMR keeps a glued face conforming, so a region that
refines a block on the cube's face would refine the shells across it, and
is refused: a geometry takes refinement only where its map is affine
([`affine_patch`](@ref TreeWaveOLDG.affine_patch)).

Hand both to [`WaveProblem`](@ref): `WaveProblem(forest, el; geometry, boundary)`.
"""
function sevenpatch_forest(::Type{T}, P::Integer, E::Integer; roots=(2, 2), Lc=1,
                           R1=3, refined=false) where {T}
    N = E * (P + 1)
    iseven(E) || throw(ArgumentError(
        "E = $E elements of order P = $P per block is odd, and E must be even: " *
        "TreeAMR's element centering needs N = $N to be a multiple of " *
        "2(P+1) = $(2(P + 1))"))
    geo = SevenPatchBall{T}(Lc, R1, Tuple(roots))
    forest = Forest{T}([(roots=sevenpatch_roots(geo, p),) for p in 1:7]; N=N,
                       glue=sevenpatch_glue())
    refined === false && return forest, geo
    n = geo.roots[1]
    refine_regions!(forest, refinement_regions(refined, Val(3), n), n; p=1)
    for key in forest.leaves
        q = patch(forest, key)
        q == 1 || level(key) == 0 || throw(ArgumentError(
            "the refinement $(repr(refined)) reaches the cube's faces, and the glue keeps " *
            "the leaves across a glued face at one level, so it refined shell $q, whose " *
            "map is not affine: keep the regions at least one root inside the cube " *
            "[0, $n]³"))
    end
    return forest, geo
end

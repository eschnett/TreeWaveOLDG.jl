# Where the values are, and how to integrate them.
#
# The node-duplicated OLDG layout is TreeAMR's element-centered field set
# (`elementcentered(D)`, with `P` the element's order): a block of
# `N = E(P+1)` points per dimension holds `E` elements, owns all `N` (both
# face nodes of every element), and at `G = 1` the exchange puts the
# neighbouring block's adjacent face node into the one ghost plane across
# each face — exactly the value the wide derivative reads there, through
# the element-local operators where the neighbour is coarser or finer.
# See "The layout" in `CODE.md`.
#
# The positions and the quadrature are TreeAMR's too: `coordinates` and
# `fill_by_coordinates!` place an element set's points at its
# Gauss–Lobatto nodes, and its norms weight them by the element's
# quadrature. The functions here keep the package's names for them, take
# the `Element`, and refuse a set that is not element-centered at its
# order.

"""
    check_layout(forest, el)

Refuse a forest this package cannot run on, with the reason.

The block size has to be a whole number `E` of elements, and `E` has to
be even. A block of a refined forest covers half its parent's extent in
each dimension, so for it to cover whole elements of its parent, `N/2`
has to be a multiple of `P + 1`: TreeAMR's element centering requires `N`
to be a multiple of `2(P+1)`, which is what makes every coarse-fine
transfer read whole elements. The forest may be refined.
"""
function check_layout(forest::Forest, el::Element{T,P1}) where {T,P1}
    forest.N % P1 == 0 || throw(ArgumentError(
        "the block size N = $(forest.N) is not a whole number of elements of order " *
        "P = $(P1 - 1), which hold $P1 nodes each: choose N = E·$P1 for some even " *
        "number of elements E per block"))
    E = forest.N ÷ P1
    iseven(E) || throw(ArgumentError(
        "the block size N = $(forest.N) holds E = $E elements of order P = " *
        "$(P1 - 1), and E must be even: a block of a refined forest covers half " *
        "of its parent, and has to cover whole elements of it, so N must be a " *
        "multiple of 2(P+1) = $(2P1)"))
    return nothing
end

# Refuse a field set whose points are not this element's nodes.
function check_element_set(fs::FieldSet, el::Element)
    fs.P == order(el) && return nothing
    throw(ArgumentError(fs.P == 0 ?
        "this field set is not element-centered (its centering is " *
        "$(fs.centering)): build it with centering = elementcentered(D) and " *
        "P = $(order(el)), so that its points are the element's Gauss–Lobatto " *
        "nodes, its ghosts are filled by the element-local operators and its " *
        "norms are weighted by the element's quadrature" :
        "this field set holds elements of order P = $(fs.P), and the element is " *
        "of order P = $(order(el)): the nodes, the stencil and the quadrature " *
        "weights all follow from P, so the two must agree"))
end

"""
    node_coordinates(fs, el, b, I)

The physical position of owned point `I` (an `NTuple{D,Int}` over `1:N`,
*not* a stored index) of block `b`: the Gauss–Lobatto node it holds.
`fs` must be element-centered at `el`'s order.

This is TreeAMR's `coordinates` at the stored index `I + G`. Element `q =
(I-1) ÷ (P+1)` and node `i = (I-1) % (P+1)` sit at `origin + h·(q(P+1) +
offsets[i])`, with the block's own origin and spacing, and the offsets of
the two face nodes are the integers `0` and `P+1`, so two face nodes at
one position get bit-identical coordinates.
"""
function node_coordinates(fs::FieldSet{T,D}, el::Element{T}, b::Integer,
                          I::NTuple{D,<:Integer}) where {T,D}
    check_element_set(fs, el)
    return coordinates(fs, b, ntuple(d -> Int(I[d]) + fs.G[d], D))
end

"""
    fill_at_nodes!(f, fs, el)

Set every owned point of every block from `f(x, v) -> value`, with `x`
the point's Gauss–Lobatto node: TreeAMR's `fill_by_coordinates!`, for a
field set that is element-centered at `el`'s order. Ghosts are left for
the exchange.

`f` is a kernel argument, so it must be `isbits` and capture no `Type`;
see "Callbacks on a device" in TreeAMR's `fill_by_coordinates!`.
"""
function fill_at_nodes!(f, fs::FieldSet{T,D}, el::Element{T}) where {T,D}
    check_layout(fs.forest, el)
    check_element_set(fs, el)
    return fill_by_coordinates!(f, fs)
end

# The quadrature weight of owned point `I` in a block of spacing `h`: the
# tensor product of the element's weights, each scaled from `[-1, 1]` to an
# element `hₑ = (P+1)h` wide. TreeAMR weights an element set's norms the
# same way; this one is for the folds that select variables or read a
# working array, which its norms do not.
@inline function node_weight(I, h, el::Element{T,P1}, ::Val{D}) where {T,P1,D}
    w = one(h)
    for d in 1:D
        @inbounds w *= h * P1 / 2 * el.weights[(I[d] - 1) % P1 + 1]
    end
    return w
end

# A fold's `rf` for `quadrature_parts`: `Σ w |x|^p` over the selected
# variables, and `Σ w` once per variable, so that the ratio is a mean.
struct NormFold{EL,D}
    el::EL
    p::Int
end

function (nf::NormFold{EL,D})(acc, q) where {EL,D}
    w = node_weight(q.I, q.h, nf.el, Val(D))
    s = zero(w)
    for x in q.u
        s += abs(x)^nf.p
    end
    return TreeAMR.combine(acc, (Sum(w * s), Sum(w * length(q.u))))
end

function quadrature_parts(fs::FieldSet{T,D}, el::Element{T}, u, p::Integer,
                          vars) where {T,D}
    R = real(float(T))
    init = (Sum(zero(R)), Sum(zero(R)))
    rf = NormFold{typeof(el),D}(el, Int(p))
    acc = u === nothing ?
          mesh_fold(rf, TreeAMR.combine, init, fs; vars=vars) :
          mesh_fold(rf, TreeAMR.combine, init, fs, u; vars=vars)
    return acc[1].value, acc[2].value
end

# The same on a geometry, each node's weight also multiplied by a metric
# factor: `√g` for `which = 0` (the state's norms and kinetic energy), and
# `1/A_d` for `which = d` (the flux `F_d`'s potential energy, `F_d²/A_d`).
# The patch reaches the fold as each block's tag.
struct MetricFold{EL,G,D}
    el::EL
    geometry::G
    p::Int
    which::Int
end

function (mf::MetricFold{EL,G,D})(acc, q) where {EL,G,D}
    w = node_weight(q.I, q.h, mf.el, Val(D))
    c = metric_factor(mf.geometry, q.x, q.tag, mf.which)
    s = zero(w)
    for x in q.u
        s += abs(x)^mf.p
    end
    return TreeAMR.combine(acc, (Sum(w * c * s), Sum(w * c * length(q.u))))
end

# The fold's metric factor at `x` of patch `p`; a full geometry has its
# own method, `√g` alone (`geometry.jl`).
@inline function metric_factor(geo, x, p, which)
    J, A = geometry_terms(geo, x, p)
    return which == 0 ? J : inv(A[which])
end

function metric_parts(fs::FieldSet{T,D}, el::Element{T}, geometry, u, p::Integer, vars,
                      which::Integer) where {T,D}
    R = real(float(T))
    init = (Sum(zero(R)), Sum(zero(R)))
    rf = MetricFold{typeof(el),typeof(geometry),D}(el, geometry, Int(p), Int(which))
    forest = fs.forest
    tag = key -> Int(patch(forest, key))
    acc = u === nothing ?
          mesh_fold(rf, TreeAMR.combine, init, fs; vars=vars, tag=tag) :
          mesh_fold(rf, TreeAMR.combine, init, fs, u; vars=vars, tag=tag)
    return acc[1].value, acc[2].value
end

"""
    quadrature_norm(fs, el, u; p=2, vars=1:fs.nvars)

The `p`-norm of the state vector `u` under the Gauss–Lobatto quadrature,
normalized by the domain volume (and the number of variables), each
block's nodes weighted by its own spacing. `p = Inf` is the largest value
at any node. Over all variables it is TreeAMR's `volume_weighted_norm`
of the element set to rounding; this one also selects variables.

This is the inner product the operator's summation-by-parts property
holds in, and so the norm in which the scheme is stable.
"""
function quadrature_norm(fs::FieldSet{T,D}, el::Element{T}, u::AbstractVector;
                         p::Real=2, vars=1:fs.nvars) where {T,D}
    check_element_set(fs, el)
    R = real(float(T))
    if isinf(p)
        return mesh_mapreduce(abs, max, zero(R), fs, u; vars=vars)
    end
    isinteger(p) && p >= 1 || throw(ArgumentError(
        "quadrature_norm takes an integer p ≥ 1 or Inf, got p = $p"))
    total, volume = quadrature_parts(fs, el, u, Int(p), vars)
    volume == 0 && return zero(R)
    return (total / volume)^inv(R(p))
end

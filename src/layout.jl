# Where the values are, and how to integrate them.
#
# The node-duplicated OLDG layout fits TreeAMR's cell-centered field set
# with no change to the mesh: a block of `N = E(P+1)` points per dimension
# holds `E` elements, owns all `N` (both face nodes of every element), and
# at `G = 1` the same-level exchange puts the neighbouring block's adjacent
# face node into the one ghost plane on each side — exactly the value the
# wide derivative reads there. See "The layout" in `CODE.md`.
#
# What the mesh does *not* know is where the nodes are or how to weight
# them. `coordinates`, `fill_by_coordinates!` and `volume_weighted_norm`
# all assume `N` equally spaced cell centres, so this file replaces the
# three for Gauss–Lobatto nodes. Each takes the `Element` that says where
# its nodes sit.

"""
    check_layout(forest, el)

Refuse a forest this prototype cannot run on, with the reason.

The block size has to be a whole number of elements, and the forest has
to be uniform: on a refined forest a coarse-fine ghost would be
interpolated as if the points were equally spaced cell centres, which
they are not. Running OLDG there needs a Gauss–Lobatto centering and an
element-local operator family in TreeAMR; see "Scope and non-goals" in
`CODE.md`.
"""
function check_layout(forest::Forest, el::Element{T,P1}) where {T,P1}
    forest.N % P1 == 0 || throw(ArgumentError(
        "the block size N = $(forest.N) is not a whole number of elements of order " *
        "P = $(P1 - 1), which hold $P1 nodes each: choose N = E·$P1 for some " *
        "number of elements E per block (N must also be even, for TreeAMR)"))
    maxlevel(forest) == 0 || throw(ArgumentError(
        "this forest is refined (maxlevel = $(maxlevel(forest))), and OLDG runs on " *
        "a uniform forest only: TreeAMR fills a coarse-fine ghost by interpolating " *
        "equally spaced cell centres, and Gauss–Lobatto nodes are neither. See " *
        "\"Scope and non-goals\" in CODE.md"))
    return nothing
end

# The position of owned point `I` of a block, per dimension: element
# `q = (I-1) ÷ (P+1)` of the block, node `i = (I-1) % (P+1)` of the element.
# `q·(P+1) + offsets[i+1]` is an integer for both face nodes, so two face
# nodes at one position get bit-identical coordinates.
@inline function node_position(origin::NTuple{D}, h, I, el::Element{T,P1}) where {D,T,P1}
    return ntuple(Val(D)) do d
        q, i = divrem(I[d] - 1, P1)
        @inbounds origin[d] + h * (q * P1 + el.offsets[i + 1])
    end
end

"""
    node_coordinates(fs, el, b, I)

The physical position of owned point `I` (an `NTuple{D,Int}` over `1:N`,
*not* a stored index) of block `b`: the Gauss–Lobatto node it holds.

TreeAMR's `coordinates` would report the equally spaced cell centre with
the same index, which agrees with this only at the element's centre node
(odd `P + 1`) and nowhere else.
"""
function node_coordinates(fs::FieldSet{T,D}, el::Element{T}, b::Integer,
                          I::NTuple{D,<:Integer}) where {T,D}
    key = blockkey(fs, b)
    origin = block_origin(T, fs.forest, key)
    h = spacing(T, fs.forest, key)
    return node_position(origin, h, I, el)
end

@kernel function fill_nodes_kernel!(work, f, @Const(origins), @Const(spacings), el,
                                    ::Val{D}, ::Val{G}, ::Val{NV}) where {D,G,NV}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    @inbounds x = node_position(origins[b], spacings[b], I, el)
    c = ntuple(d -> I[d] + G[d], Val(D))
    ntuple(Val(NV)) do v
        @inbounds work[c..., v, b] = f(x, v)
        nothing
    end
end

"""
    fill_at_nodes!(f, fs, el)

Set every owned point of every block from `f(x, v) -> value`, with `x`
the point's Gauss–Lobatto node — `fill_by_coordinates!` for this layout.
Ghosts are left for the exchange.

`f` is a kernel argument, so it must be `isbits` and capture no `Type`;
see "Callbacks on a device" in TreeAMR's `fill_by_coordinates!`.
"""
function fill_at_nodes!(f, fs::FieldSet{T,D}, el::Element{T}) where {T,D}
    check_layout(fs.forest, el)
    backend = get_backend(fs.work)
    origins = to_backend(backend, block_origins(fs.forest, T))
    spacings = to_backend(backend, block_spacings(fs.forest, T))
    map_blocks!(fill_nodes_kernel!, fs, fs.work, f, origins, spacings, el,
                Val(D), Val(fs.G), Val(fs.nvars))
    return fs
end

# The quadrature weight of owned point `I` in a block of spacing `h`: the
# tensor product of the element's weights, each scaled from `[-1, 1]` to an
# element `hₑ = (P+1)h` wide.
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

"""
    quadrature_norm(fs, el, u; p=2, vars=1:fs.nvars)

The `p`-norm of the state vector `u` under the Gauss–Lobatto quadrature,
normalized by the domain volume (and the number of variables) — the
replacement for TreeAMR's `volume_weighted_norm`, whose equal weights
would over-count the clustered nodes near element faces. `p = Inf` is the
largest value at any node.

This is the inner product the operator's summation-by-parts property
holds in, and so the norm in which the scheme is stable.
"""
function quadrature_norm(fs::FieldSet{T,D}, el::Element{T}, u::AbstractVector;
                         p::Real=2, vars=1:fs.nvars) where {T,D}
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

# The reference element: everything OLDG needs to know about one element
# of polynomial order `P`, built once per run and handed to the kernels as
# plain bits.
#
# An element holds `P + 1` Gauss–Lobatto–Legendre nodes `ξ₀ = -1 < … <
# ξ_P = 1`. Both face nodes are stored, so a face position carries two
# values, one per side. Miller and Schnetter's wide derivative operator
# (arXiv:1604.00075, eq. 57) is
#
#     D = d - ½(b - F)
#
# with `d` the element's own pseudospectral derivative, `b` the boundary
# operator (a discrete delta at the two faces, reading this element's face
# value) and `F` the fetch operator (the same, reading the neighbour's face
# value). In nodal form `D` is a `(P+1) × (P+3)` matrix: row `i` is node
# `i`, and its columns are the nodes `-1 … P+1` — the element's own nodes
# with the neighbours' adjacent face nodes on either side. That makes it a
# finite-difference stencil with non-constant coefficients: the window of
# a node starts one point before its element and ends one point after it.
#
# The coefficients are irrational from `P = 3` on, so they are built in
# `BigFloat` and rounded to `T` once. See "Precision" in `CODE.md`.

"""
    Element{T,P1,PW}

The reference data of an OLDG element of order `P = P1 - 1`, in the run's
element type `T`. Every field is a tuple, so an `Element` is `isbits` and
can be a kernel argument on any backend.

- `nodes[i+1]`: the Gauss–Lobatto node `ξᵢ ∈ [-1, 1]`.
- `weights[i+1]`: its quadrature weight `ŵᵢ` on `[-1, 1]` (they sum to 2).
- `offsets[i+1]`: the node's position within its element in units of the
  mesh spacing `h`, `(P+1)(1+ξᵢ)/2`, so that `offsets[1] == 0` and
  `offsets[P1] == P1` exactly and the two face nodes of adjacent elements
  sit at bit-identical positions.
- `stencil[i+1][j+2]`: the wide derivative `D`, in units of `1/h`, for
  `j = -1 … P+1`. Columns `j = -1` and `j = P+1` are the neighbours' face
  nodes and are nonzero only in the first and last row.
- `truncate[i+1][j+1]`: the projector that removes the `P`-th Legendre
  mode, `Π = I - |L_P⟩⟨L_P|/γ` in the discrete inner product.
- `topmode[i+1]` and `topdual[j+1]`: the same projector as its rank-one
  correction, `Π = I - topmode ⊗ topdual`, with `topmode[i+1] = L_P(ξᵢ)`
  and `topdual[j+1] = ŵⱼ L_P(ξⱼ)/γ`. The truncation kernel applies it
  this way, one dimension at a time ("Truncation" in `CODE.md`); the
  matrix is what the tests compare against.

`h` here is TreeAMR's spacing, `block extent / N`, not the element width:
an element is `P + 1` cells wide, `hₑ = (P+1)h`. Writing the operator per
`h` is what lets the kernel divide by the same per-block spacing TreeWave's
does.
"""
struct Element{T,P1,PW}
    nodes::NTuple{P1,T}
    weights::NTuple{P1,T}
    offsets::NTuple{P1,T}
    stencil::NTuple{P1,NTuple{PW,T}}
    truncate::NTuple{P1,NTuple{P1,T}}
    topmode::NTuple{P1,T}
    topdual::NTuple{P1,T}
end

order(::Element{T,P1}) where {T,P1} = P1 - 1

# Legendre polynomials `L₀ … L_P` at `x`, by the three-term recurrence.
function legendre_all(P::Integer, x)
    L = Vector{typeof(x)}(undef, P + 1)
    L[1] = one(x)
    P >= 1 && (L[2] = x)
    for k in 2:P
        L[k + 1] = ((2k - 1) * x * L[k] - (k - 1) * L[k - 1]) / k
    end
    return L
end

"""
    gll_nodes_weights(P; precision=512)

The `P + 1` Gauss–Lobatto–Legendre nodes on `[-1, 1]` in ascending order,
and their weights, as `BigFloat` vectors.

Newton's iteration on `(1 - x²) L_P'(x)` written through the recurrence,
from the Chebyshev–Gauss–Lobatto points, as in Hesthaven and Warburton's
`JacobiGL` for `α = β = 0`. The endpoints are fixed points of the
iteration, so they come out as `±1` exactly; the nodes are then
symmetrized, so `ξᵢ == -ξ_{P-i}` holds bit for bit after rounding to any
type.
"""
function gll_nodes_weights(P::Integer; precision::Integer=512)
    P >= 1 || throw(ArgumentError("the element order P must be at least 1, got $P"))
    return setprecision(BigFloat, precision) do
        x = [-cospi(big(i) / P) for i in 0:P]
        tol = ldexp(one(BigFloat), 8 - precision)
        for _ in 1:200
            delta = zero(x)
            for i in eachindex(x)
                L = legendre_all(P, x[i])
                delta[i] = (x[i] * L[P + 1] - L[P]) / ((P + 1) * L[P + 1])
            end
            x .-= delta
            maximum(abs, delta) <= tol && break
        end
        x = [(x[i + 1] - x[P + 1 - i]) / 2 for i in 0:P]
        x[1], x[P + 1] = -one(BigFloat), one(BigFloat)
        w = [2 / (P * (P + 1) * legendre_all(P, xi)[P + 1]^2) for xi in x]
        x, w
    end
end

"""
    Element(T, P)

The reference element of order `P` in type `T`. See [`Element`](@ref) for
what it holds.

The derivative matrix uses the closed form for Gauss–Lobatto nodes,
`d̂ᵢⱼ = L_P(ξᵢ) / (L_P(ξⱼ)(ξᵢ - ξⱼ))` off the diagonal and `∓P(P+1)/4` at
the two corners. Divided by `hₑ/2 = (P+1)h/2` it is the element's own
derivative per `h`; the face terms of `-½(b - F)` add `P/2` times the jump
`u₀ - u₋₁` to the first row and `u_{P+1} - u_P` to the last, because
`1/(hₑ ŵ₀) = P/(2h)`.
"""
function Element(::Type{T}, P::Integer; precision::Integer=512) where {T}
    ξ, ŵ = gll_nodes_weights(P; precision=precision)
    P1 = P + 1
    stencil, trunc, mode, dual = setprecision(BigFloat, precision) do
        LP = [legendre_all(P, x)[P + 1] for x in ξ]
        d = zeros(BigFloat, P1, P1)
        for i in 1:P1, j in 1:P1
            i == j && continue
            d[i, j] = LP[i] / (LP[j] * (ξ[i] - ξ[j]))
        end
        d[1, 1] = -big(P) * (P + 1) / 4
        d[P1, P1] = big(P) * (P + 1) / 4

        # Columns 1 … P+3 are nodes -1 … P+1.
        W = zeros(BigFloat, P1, P1 + 2)
        W[:, 2:(P1 + 1)] .= 2 .* d ./ (P + 1)
        face = big(P) / 2
        W[1, 2] += face
        W[1, 1] -= face
        W[P1, P1 + 1] -= face
        W[P1, P1 + 2] += face

        γ = sum(ŵ .* LP .^ 2)
        Π = [(i == j ? one(BigFloat) : zero(BigFloat)) - LP[i] * ŵ[j] * LP[j] / γ
             for i in 1:P1, j in 1:P1]
        W, Π, LP, ŵ .* LP ./ γ
    end
    tup(v) = ntuple(i -> T(v[i]), length(v))
    rows(M) = ntuple(i -> tup(M[i, :]), size(M, 1))
    offsets = setprecision(BigFloat, precision) do
        [(P + 1) * (1 + x) / 2 for x in ξ]
    end
    offsets[1], offsets[P1] = 0, P1
    return Element{T,P1,P1 + 2}(tup(ξ), tup(ŵ), tup(offsets), rows(stencil), rows(trunc),
                                tup(mode), tup(dual))
end

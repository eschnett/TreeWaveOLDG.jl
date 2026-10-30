# The transfer between the particles and the grid (INSPIRAL.md, "The
# transfer operator" and "Quadrature"): the projection of a source onto
# the nodes, its adjoint pairing of a particle kernel with a grid
# function, the reconstruction of a grid function at a point, and the
# quadrature rules they need. Generic in the number of dimensions `D`;
# the particle physics on top is three-dimensional. See "The transfer"
# and "Quadrature" in `CODE.md`.
#
# Within an element, `φₙ` is the tensor-product Lagrange polynomial of
# Gauss–Lobatto node `n` and `wₙ` its quadrature weight. The projection is
#
#     (P S)ₙ = (1/wₙ) ∫_{element(n)} S φₙ,
#
# so that `⟨P S, g⟩_w = ∫ S g_h` for every grid function `g`, with
# `g_h = Σₙ gₙ φₙ` its reconstruction. It is element-local and needs no
# exchange.
#
# Four rules integrate over an element, chosen per element:
#
# - Far from every particle, beyond `r_far`, tensor Gauss–Legendre with
#   `q = P + 3` points per dimension, in two kernels: one evaluates `S` at
#   the points, the other contracts them against the Lagrange table one
#   dimension at a time.
# - Between `r_near` and `r_far`, the graded rule: tensor Gauss–Legendre
#   with more points the closer the element is, on the host, contracted
#   as the points are visited.
# - Within `r_near` of a particle, a nested rule: one dimension at a
#   time, each line split where it crosses the particle's support, with
#   Gauss–Legendre inside the support and Johnston and Elliott's sinh rule
#   outside, scaled by the line's distance from the particle. The profile
#   is only `C³` at the support's edge, where no Gauss rule converges
#   faster than algebraically, so the edge is a breakpoint.
# - The pairing over the elements that meet a particle's support uses the
#   same nested rule, without the parts outside the support.

"""
    gauss_nodes_weights(n; precision=512)

The `n` Gauss–Legendre nodes on `[-1, 1]` in ascending order, and their
weights, as `BigFloat` vectors: Newton's iteration on `L_n` from the
Chebyshev nodes, then symmetrized, so that `x[i] == -x[n+1-i]` holds bit
for bit after rounding to any type.
"""
function gauss_nodes_weights(n::Integer; precision::Integer=512)
    n >= 1 || throw(ArgumentError("a Gauss rule needs at least one point, got n = $n"))
    return setprecision(BigFloat, precision) do
        x = [-cospi((big(4i) - 1) / (4n + 2)) for i in 1:n]
        tol = ldexp(one(BigFloat), 8 - precision)
        legendre_and_derivative(ξ) = (L = legendre_all(n, ξ);
                                      (L[n + 1], n * (ξ * L[n + 1] - L[n]) / (ξ^2 - 1)))
        for _ in 1:200
            delta = map(ξ -> (Lξ = legendre_and_derivative(ξ); Lξ[1] / Lξ[2]), x)
            x .-= delta
            maximum(abs, delta) <= tol && break
        end
        x = [(x[i] - x[n + 1 - i]) / 2 for i in 1:n]
        w = [2 / ((1 - ξ^2) * legendre_and_derivative(ξ)[2]^2) for ξ in x]
        x, w
    end
end

"""
    GaussRule{T,P1,Q}

A tensor Gauss–Legendre rule of `Q` points per dimension for elements of
order `P = P1 - 1`: the points `nodes` on `[-1, 1]`, their `weights`, and
`contract[i+1][k] = wₖ φᵢ(ξₖ)/ŵᵢ`, its one-dimensional factor of the
projection `P`.
"""
struct GaussRule{T,P1,Q}
    nodes::NTuple{Q,T}
    weights::NTuple{Q,T}
    contract::NTuple{P1,NTuple{Q,T}}
end

"""
    TransferTables{T,P1,Q,NS,NM,NI,K,G}

The reference data of the transfer for elements of order `P = P1 - 1`,
built in `BigFloat` and rounded once to `T`, as tuples, so that the
tables are `isbits` and travel as a kernel argument beside the
[`Element`](@ref):

- `nodes`, `weights`: the element's Gauss–Lobatto nodes and weights;
  `bary[i+1] = 1/∏_{j≠i}(ξᵢ - ξⱼ)`, so that
  `φᵢ(ξ) = bary[i+1] ∏_{j≠i}(ξ - ξⱼ)` anywhere.
- `gauss`, `gaussweights`: the far rule, `Q = P + 3` Gauss–Legendre
  points on `[-1, 1]`.
- `lagrange[k][i+1] = φᵢ(ξᴳₖ)`, the `Q × (P+1)` Lagrange table at the
  far rule's points, and `contract[i+1][k] = wᴳₖ φᵢ(ξᴳₖ)/ŵᵢ`, the same
  divided by the node weight: the one-dimensional factor of `P` under the
  far rule. The element's Jacobian cancels between the integral and `wₙ`.
- `sinhnodes`, `sinhweights`: the `NS` reference Gauss points that the
  sinh rule transforms ([`sinh_rule`](@ref)) on a line closer to the
  particle than half its length, and `midnodes`, `midweights`, the `NM`
  it transforms on a line farther away.
- `innernodes`, `innerweights`: the `NI` Gauss points used inside a
  particle's support.
- `graded`, `gradedfrom`: the graded rule, `K` [`GaussRule`](@ref)s, of
  which an element at least `gradedfrom[i]·hₑ` from every particle and
  less than `rfar·hₑ` takes the last one that applies, and the first one
  when none does. Beyond `rfar·hₑ` the far rule applies.

The defaults are what "Quadrature" in `CODE.md` measured to give
`10⁻¹²` relative.
"""
struct TransferTables{T,P1,Q,NS,NM,NI,K,G}
    nodes::NTuple{P1,T}
    weights::NTuple{P1,T}
    bary::NTuple{P1,T}
    gauss::NTuple{Q,T}
    gaussweights::NTuple{Q,T}
    lagrange::NTuple{Q,NTuple{P1,T}}
    contract::NTuple{P1,NTuple{Q,T}}
    sinhnodes::NTuple{NS,T}
    sinhweights::NTuple{NS,T}
    midnodes::NTuple{NM,T}
    midweights::NTuple{NM,T}
    innernodes::NTuple{NI,T}
    innerweights::NTuple{NI,T}
    graded::G
    gradedfrom::NTuple{K,T}
    rfar::T
end

order(::TransferTables{T,P1}) where {T,P1} = P1 - 1

"""
    TransferTables(T, P; q = P + 3, graded = ((1, 14), (2, 10), (3, 8), (4, 7)),
                   rfar = 9, nsinh = 48, nmid = 16, ninner = 24,
                   precision = 512)

The transfer tables for elements of order `P` in type `T`. `graded` holds
the graded rule as `(distance, points)` pairs, the distance from the
nearest particle in units of `hₑ` from which each applies, increasing,
and `rfar` the distance from which the far rule's `q` points apply. See
[`TransferTables`](@ref) for what they hold.
"""
function TransferTables(::Type{T}, P::Integer; q::Integer=P + 3,
                        graded::Tuple=((1, 14), (2, 10), (3, 8), (4, 7)), rfar::Real=9,
                        nsinh::Integer=48, nmid::Integer=16, ninner::Integer=24,
                        precision::Integer=512) where {T}
    issorted(first.(graded); lt=<=) && all(r -> 0 <= first(r) < rfar, graded) ||
        throw(ArgumentError("the graded rule's distances must increase from 0 and stay " *
                            "below rfar = $rfar, got $graded"))
    ξ, ŵ = gll_nodes_weights(P; precision=precision)
    P1 = P + 1
    xg, wg = gauss_nodes_weights(q; precision=precision)
    xs, ws = gauss_nodes_weights(nsinh; precision=precision)
    xm, wm = gauss_nodes_weights(nmid; precision=precision)
    xi, wi = gauss_nodes_weights(ninner; precision=precision)
    bary = setprecision(BigFloat, precision) do
        [1 / prod(ξ[i] - ξ[j] for j in 1:P1 if j != i) for i in 1:P1]
    end
    # The Lagrange table at the points `x` and the contraction `wₖ φᵢ(xₖ)/ŵᵢ`.
    tables(x, w) = setprecision(BigFloat, precision) do
        φ = [bary[i] * prod(xk - ξ[j] for j in 1:P1 if j != i) for xk in x, i in 1:P1]
        φ, [w[k] * φ[k, i] / ŵ[i] for i in 1:P1, k in eachindex(x)]
    end
    tup(v) = ntuple(i -> T(v[i]), length(v))
    rows(M) = ntuple(i -> tup(M[i, :]), size(M, 1))
    lag, con = tables(xg, wg)
    rules = map(graded) do (_, n)
        x, w = gauss_nodes_weights(n; precision=precision)
        GaussRule{T,P1,n}(tup(x), tup(w), rows(tables(x, w)[2]))
    end
    return TransferTables{T,P1,q,nsinh,nmid,ninner,length(graded),typeof(rules)}(
        tup(ξ), tup(ŵ), tup(bary), tup(xg), tup(wg), rows(lag), rows(con), tup(xs), tup(ws),
        tup(xm), tup(wm), tup(xi), tup(wi), rules, map(r -> T(first(r)), graded), T(rfar))
end

# The Lagrange polynomials of the element's nodes at reference coordinate
# `ξ`, and their derivatives, in the product form: a polynomial in `ξ`
# everywhere, with no division by `ξ - ξⱼ`.
@inline function lagrange_values(t::TransferTables{T,P1}, ξ) where {T,P1}
    return ntuple(Val(P1)) do i
        acc = t.bary[i] * one(ξ)
        for j in 1:P1
            j == i || (acc *= ξ - t.nodes[j])
        end
        acc
    end
end

@inline function lagrange_derivatives(t::TransferTables{T,P1}, ξ) where {T,P1}
    return ntuple(Val(P1)) do i
        acc = zero(ξ)
        for k in 1:P1
            k == i && continue
            term = t.bary[i] * one(ξ)
            for j in 1:P1
                (j == i || j == k) || (term *= ξ - t.nodes[j])
            end
            acc += term
        end
        acc
    end
end

# `kron(a, b)` for static vectors, `b`'s index running fastest.
@inline skron(a::SVector{M}, b::SVector{N}) where {M,N} =
    SVector{M * N}(ntuple(j -> a[(j - 1) ÷ N + 1] * b[(j - 1) % N + 1], Val(M * N)))

# `fs[D] ⊗ … ⊗ fs[1]`, the first factor's index running fastest. A `foldl`
# over the tuple, whose partial products differ in type, allocated each
# one.
@inline kron_all(fs::Tuple{Any}) = fs[1]
@inline kron_all(fs::Tuple) = skron(last(fs), kron_all(Base.front(fs)))

# `φₙ(x)` for every node `n` of the element `[lo, hi]`, as one static
# vector in the state's order (dimension 1 fastest), with dimension `d`'s
# factor replaced by its derivative in `x_d` when `d == deriv`.
@inline function element_basis(t::TransferTables, lo::SVector{D}, hi::SVector{D},
                               x::SVector{D}, deriv::Int=0) where {D}
    φ = ntuple(Val(D)) do d
        ξ = (2x[d] - lo[d] - hi[d]) / (hi[d] - lo[d])
        d == deriv ?
        SVector(lagrange_derivatives(t, ξ)) * (2 / (hi[d] - lo[d])) :
        SVector(lagrange_values(t, ξ))
    end
    return kron_all(φ)
end

# The quadrature weight `wₙ` of every node of an element `hₑ` wide, in
# the order of `element_basis`.
@inline function element_weights(t::TransferTables{T,P1}, hₑ, ::Val{D}) where {T,P1,D}
    w = ntuple(_ -> SVector(t.weights) * (hₑ / 2), Val(D))
    return kron_all(w)
end

# Johnston and Elliott's transformation of `[-1, 1]` onto itself for a
# near-singularity at `ξc + iε`: `ξ = ξc + ε sinh(μu - η)`, with `μ` and
# `η` chosen so that `u = ±1` maps to `ξ = ±1`. It is smooth in `ξc`, also
# where `ξc` leaves the interval.
@inline function sinh_parameters(ξc, ε)
    lo, hi = asinh((1 + ξc) / ε), asinh((1 - ξc) / ε)
    return (hi + lo) / 2, (lo - hi) / 2
end

"""
    sinh_rule(ξc, ε, n)

Johnston and Elliott's sinh rule ("A sinh transformation for evaluating
nearly singular boundary element integrals", 2005): `n` Gauss–Legendre
points on `[-1, 1]`, transformed by `ξ = ξc + ε sinh(μu - η)` so that they
cluster about `ξc` on the scale `ε`. Returns the nodes and weights. The
number of points a near-singularity at distance `ε` needs grows only like
`log(1/ε)`, and the rule is a smooth function of `ξc`, including where
`ξc` lies outside `[-1, 1]`.
"""
function sinh_rule(ξc::Real, ε::Real, n::Integer)
    T = float(promote_type(typeof(ξc), typeof(ε)))
    u, w = gauss_nodes_weights(n)
    μ, η = sinh_parameters(T(ξc), T(ε))
    nodes = [T(ξc) + T(ε) * sinh(μ * T(uk) - η) for uk in u]
    weights = [T(wk) * T(ε) * μ * cosh(μ * T(uk) - η) for (uk, wk) in zip(u, w)]
    return nodes, weights
end

"""
    ParticleSupport(z, v, ℓ)
    ParticleSupport(m::ParticleModel, traj::Trajectory)

Where a particle is, for the transfer: its centre `z`, its velocity `v`
and its size `ℓ`, as `D`-vectors. Its support is the ellipsoid
`|Λ(v)(x - z)| ≤ ℓ`, the rest-frame ball of radius `ℓ` contracted along
`v`, and the scale on which sources near it vary is `ℓ`.
"""
struct ParticleSupport{D,T}
    z::SVector{D,T}
    v::SVector{D,T}
    ℓ::T
end

function ParticleSupport(z::SVector{D}, v::SVector{D}, ℓ) where {D}
    T = float(promote_type(eltype(z), eltype(v), typeof(ℓ)))
    return ParticleSupport{D,T}(z, v, ℓ)
end

ParticleSupport(m::ParticleModel, traj::Trajectory) = ParticleSupport(traj.z, traj.v, m.ℓ)

# The squared distance from `x` to the box `[lo, hi]`, over dimensions
# `dims`.
@inline function box_distance2(x, lo, hi, dims)
    acc = zero(eltype(x))
    for d in dims
        δ = max(lo[d] - x[d], x[d] - hi[d], zero(eltype(x)))
        acc += δ^2
    end
    return acc
end

# Where the line along dimension `d = F + 1` through the fixed coordinates
# `prefix` (dimensions `1:F`) needs a breakpoint in `(lo[d], hi[d])`, and
# the extent of the support's section along it.
#
# The support is `rᵀMr ≤ ℓ²`, `r = x - z`, `M = I + γ²vvᵀ`. Fixing some
# coordinates and minimizing over the free ones leaves the quadratic form
# `|r_K|² + c(v_K·r_K)²` on the fixed set `K`, with `c = γ²/(1 + γ²|v_free|²)`
# (Sherman–Morrison). The extent of the section along `d` is where that
# form, with `r_d` free and `1:F` fixed, equals `ℓ²`. An integral over the
# remaining dimensions `d+1:D` changes character, as a function of `x_d`,
# also where the section starts to cross a face of the element in those
# dimensions: there one of them is fixed at the face. So each of the
# remaining dimensions is free, at its low face, or at its high face, and
# every combination contributes the two roots of a quadratic. Assignment 0,
# all free, is the extent.
function support_breaks(s::ParticleSupport{D,T}, prefix::SVector{F}, lo, hi) where {D,T,F}
    d = F + 1
    γ² = 1 / (1 - dot(s.v, s.v))
    r²F = zero(T)
    vrF = zero(T)
    for j in 1:F
        r²F += (prefix[j] - s.z[j])^2
        vrF += s.v[j] * (prefix[j] - s.z[j])
    end
    breaks = T[]
    extent = nothing
    for code in 0:(3^(D - d) - 1)
        r² = r²F
        vr = vrF
        vfree = zero(T)
        c = code
        for j in (d + 1):D
            c, k = divrem(c, 3)
            if k == 0
                vfree += s.v[j]^2
            else
                f = (k == 1 ? lo[j] : hi[j]) - s.z[j]
                r² += f^2
                vr += s.v[j] * f
            end
        end
        cc = γ² / (1 + γ² * vfree)
        A = 1 + cc * s.v[d]^2
        B = cc * vr * s.v[d]
        C = r² + cc * vr^2 - s.ℓ^2
        disc = B^2 - A * C
        disc > 0 || continue
        q = sqrt(disc)
        x₁ = s.z[d] + (-B - q) / A
        x₂ = s.z[d] + (-B + q) / A
        code == 0 && (extent = (x₁, x₂))
        lo[d] < x₁ < hi[d] && push!(breaks, x₁)
        lo[d] < x₂ < hi[d] && push!(breaks, x₂)
    end
    sort!(breaks)
    return breaks, extent
end

# The nested rule over the box `[lo, hi]`: `Σ w f(x)` over its points.
# `f` returns anything with `+` and multiplication by a weight, and `init`
# is its zero. One dimension at a time, `Val(n)` dimensions remaining;
# each line is split at `support_breaks`. A segment inside the support's
# extent gets the inner Gauss rule, under the substitution
# `x = a + (b - a)(1 - cos θ)/2` except on the innermost line, because the
# integral over the remaining dimensions behaves like a half-integer
# power of the distance to the segment's ends there. A segment outside
# gets the sinh rule toward the particle's centre, projected onto the
# segment, with the scale its distance from the line. `compact` skips the
# segments outside, for a kernel that vanishes there.
@inline nested_rule(f, init, s::ParticleSupport, t::TransferTables, lo, hi,
                    prefix::SVector, ::Val{0}, compact::Bool) = f(prefix)

# The segments of one line of the nested rule, along dimension
# `d = F + 1`: `acc = op(acc, xs, ws)` for each segment's points `xs` and
# weights `ws`, tuples, in order along the line.
const LINE_FOLD = quote
    d = F + 1
    breaks, extent = support_breaks(s, prefix, lo, hi)
    # The squared distance from the particle's centre to the line, outside
    # dimension `d`: through the fixed coordinates, and to the box in the
    # remaining ones.
    off² = box_distance2(s.z, lo, hi, (d + 1):D)
    for j in 1:F
        off² += (prefix[j] - s.z[j])^2
    end
    a = lo[d]
    for k in 1:(length(breaks) + 1)
        b = k <= length(breaks) ? breaks[k] : hi[d]
        if b > a
            mid = (a + b) / 2
            if extent !== nothing && extent[1] < mid < extent[2]
                acc = op(acc, inner_points(t, a, b, innermost)...)::A
            elseif !compact
                c = clamp(s.z[d], a, b)
                β = max(sqrt(off² + (s.z[d] - c)^2), eps(T) * (b - a))
                # `NM` points when `β` is at least half the segment's
                # length, where they reach `10⁻¹⁶` on the first run's
                # source, and `NS` closer. Both are converged at the
                # switch, so it moves the result by roundoff only.
                acc = β >= (b - a) / 2 ?
                      op(acc, sinh_points(t.midnodes, t.midweights, a, b, c, β)...)::A :
                      op(acc, sinh_points(t.sinhnodes, t.sinhweights, a, b, c, β)...)::A
            end
        end
        a = b
    end
    return acc
end

# The nested rule, one level per remaining dimension `n`: its line's
# segments by `line_fold`, each summed by `rule_sum` over `inner(x)`, the
# integral over the remaining dimensions at `x_d = x`, which is the next
# level. Each of the three has a method per level, so that no method
# recurs from one level to the next: inference widens such a recursion,
# the next level was then dispatched dynamically, and every accumulator,
# `(P+1)^D` numbers in the projection, was boxed. `f` is specialized on
# (`::G`), since it is only passed on. Up to four dimensions.
for n in 1:4
    @eval @inline function line_fold(op::O, acc::A, s::ParticleSupport{D,T},
                                     t::TransferTables, lo, hi, prefix::SVector{F},
                                     innermost::Bool, compact::Bool,
                                     ::Val{$n}) where {O,A,D,T,F}
        $LINE_FOLD
    end
    @eval @inline function rule_sum(g::G, init::A, xs, ws, ::Val{$n}) where {G,A}
        acc::A = init
        for k in eachindex(xs, ws)
            acc += ws[k] * g(xs[k])
        end
        return acc
    end
    @eval function nested_rule(f::G, init::A, s::ParticleSupport, t::TransferTables, lo,
                               hi, prefix::SVector, ::Val{$n}, compact::Bool) where {G,A}
        inner(x) = nested_rule(f, init, s, t, lo, hi, push(prefix, x), Val($(n - 1)),
                               compact)::A
        return line_fold((acc, xs, ws) -> acc + rule_sum(inner, init, xs, ws, Val($n)),
                         init, s, t, lo, hi, prefix, $(n == 1), compact, Val($n))
    end
end

# The inner Gauss rule's points and weights on `[a, b]`, plain on the
# innermost line and under `x = a + (b - a)(1 - cos θ)/2` on the others.
# Both are `ntuple`s of a static length, since `map` over a tuple of more
# than 32 elements allocates.
@inline function inner_points(t::TransferTables{T,P1,Q,NS,NM,NI}, a, b,
                              plain::Bool) where {T,P1,Q,NS,NM,NI}
    half = (b - a) / 2
    u, w = t.innernodes, t.innerweights
    θ(k) = T(π) * (1 + u[k]) / 2
    xs = ntuple(k -> plain ? a + half * (1 + u[k]) : a + half * (1 - cos(θ(k))), Val(NI))
    ws = ntuple(k -> plain ? w[k] * half : w[k] * T(π) / 2 * half * sin(θ(k)), Val(NI))
    return xs, ws
end

# The sinh rule's points and weights on `[a, b]` toward `c` at distance
# `β`, from the reference Gauss points `nodes` and `weights`.
@inline function sinh_points(nodes::NTuple{N}, weights::NTuple{N}, a, b, c, β) where {N}
    half = (b - a) / 2
    mid = (a + b) / 2
    ξc = (c - mid) / half
    ε = β / half
    μ, η = sinh_parameters(ξc, ε)
    xs = ntuple(k -> mid + half * (ξc + ε * sinh(μ * nodes[k] - η)), Val(N))
    ws = ntuple(k -> weights[k] * ε * μ * cosh(μ * nodes[k] - η) * half, Val(N))
    return xs, ws
end

# The far rule over the box `[lo, hi]`, on the host: the tensor product of
# the `Q`-point Gauss rule. The kernels below apply the same rule.
function gauss_rule(f, init, t::TransferTables{T,P1,Q}, lo::SVector{D},
                    hi::SVector{D}) where {T,P1,Q,D}
    acc = init
    for K in CartesianIndices(ntuple(_ -> Q, Val(D)))
        x = SVector(ntuple(d -> lo[d] + (hi[d] - lo[d]) * (1 + t.gauss[K[d]]) / 2, Val(D)))
        w = prod(ntuple(d -> t.gaussweights[K[d]] * (hi[d] - lo[d]) / 2, Val(D)))
        acc += w * f(x)
    end
    return acc
end

# The graded rule over the box `[lo, hi]`: `P S` at the element's nodes,
# in the order of `element_basis`, by the tensor product of `g`'s points,
# contracted one dimension at a time as they are visited. Dimension `d`
# is integrated here and `1:d-1` inside, with `x` holding the coordinates
# of `d+1:D`; `x` keeps its type, so that inference follows the recursion.
@inline function tensor_rule(S, g::GaussRule{T,P1,Q}, lo, hi, x::SVector,
                             ::Val{d}) where {T,P1,Q,d}
    acc = zero(SVector{P1^d,T})
    for k in 1:Q
        xk = setindex(x, lo[d] + (hi[d] - lo[d]) * (1 + g.nodes[k]) / 2, d)
        column = SVector(ntuple(i -> g.contract[i][k], Val(P1)))
        acc += d == 1 ? S(xk) * column :
               skron(column, tensor_rule(S, g, lo, hi, xk, Val(max(d - 1, 1))))
    end
    return acc
end

# What the projection does with an element: nothing when it misses the
# source's ball, the far rule in the kernels, or one of the host's rules
# when it lies within `rfar` of a particle.
const SKIP_ELEMENT = 0
const FAR_ELEMENT = 1
const HOST_ELEMENT = 2

@inline function element_class(lo, hi, ball, particles::Tuple, rfar)
    D = length(lo)
    box_distance2(ball[1], lo, hi, 1:D) >= ball[2]^2 && return SKIP_ELEMENT
    near = any(p -> box_distance2(p.z, lo, hi, 1:D) < rfar^2, particles)
    return near ? HOST_ELEMENT : FAR_ELEMENT
end

# The box of element `e` (zero-based, per dimension) of a block. Both
# faces are computed as `origin + h·(multiple of P+1)`, as `node_position`
# computes the face nodes, so neighbouring elements share their faces bit
# for bit.
@inline function element_box(origin, h, e, ::Val{P1}, ::Val{D}) where {P1,D}
    lo = SVector(ntuple(d -> origin[d] + h * (e[d] * P1), Val(D)))
    hi = SVector(ntuple(d -> origin[d] + h * ((e[d] + 1) * P1), Val(D)))
    return lo, hi
end

# The far rule, pass 1: `S` at every Gauss point of every far element, one
# work item per point, into `scratch`, whose extent per dimension is `E·Q`.
# Other elements' points are not written.
@kernel function source_points_kernel!(scratch, S, @Const(origins), @Const(spacings),
                                       @Const(blockmask), t::TransferTables, ball,
                                       particles, rfar, ::Val{D},
                                       ::Val{P1}) where {D,P1}
    I = @index(Global, NTuple)                 # (j1..jD, block)
    b = I[D + 1]
    Q = length(t.gauss)
    @inbounds if blockmask[b]
        h = spacings[b]
        e = ntuple(d -> (I[d] - 1) ÷ Q, Val(D))
        lo, hi = element_box(origins[b], h, e, Val(P1), Val(D))
        if element_class(lo, hi, ball, particles, rfar) == FAR_ELEMENT
            x = SVector(ntuple(d -> lo[d] + (hi[d] - lo[d]) *
                                         (1 + t.gauss[(I[d] - 1) % Q + 1]) / 2, Val(D)))
            scratch[ntuple(d -> I[d], Val(D))..., b] = S(x)
        end
    end
end

# The far rule, pass 2, one work item per far element: the values at its
# `Q^D` points are contracted with `contract` one dimension at a time, in
# place in `scratch` (each line is read into a tuple before its first
# `P + 1` entries are overwritten), until its first `(P+1)^D` entries are
# `P S` at the element's nodes. `factor` times that is added to variable
# `var` of `du`.
@kernel function contract_kernel!(du, scratch, @Const(origins), @Const(spacings),
                                  @Const(blockmask), t::TransferTables, ball, particles,
                                  rfar, factor, var, ::Val{D}, ::Val{P1}) where {D,P1}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    Q = length(t.gauss)
    @inbounds if blockmask[b]
        e = ntuple(d -> J[d] - 1, Val(D))
        lo, hi = element_box(origins[b], spacings[b], e, Val(P1), Val(D))
        if element_class(lo, hi, ball, particles, rfar) == FAR_ELEMENT
            base = ntuple(d -> e[d] * Q, Val(D))
            for d in 1:D
                lines = ntuple(k -> k < d ? P1 : (k == d ? 1 : Q), Val(D))
                for K in CartesianIndices(lines)
                    at(j) = ntuple(k -> base[k] + (k == d ? j : K[k]), Val(D))
                    line = ntuple(j -> scratch[at(j)..., b], Val(length(t.gauss)))
                    for i in 1:P1
                        acc = zero(eltype(scratch))
                        for j in 1:Q
                            acc += t.contract[i][j] * line[j]
                        end
                        scratch[at(i)..., b] = acc
                    end
                end
            end
            for K in CartesianIndices(ntuple(_ -> P1, Val(D)))
                node = ntuple(d -> e[d] * P1 + K[d], Val(D))
                local_ = ntuple(d -> base[d] + K[d], Val(D))
                du[node..., var, b] += factor * scratch[local_..., b]
            end
        end
    end
end

"""
    projection_scratch(p::WaveProblem, t::TransferTables)

The scratch array [`project_source!`](@ref) evaluates the source into:
`E·Q` points per dimension and block, on the field sets' backend. Pass it
as `scratch` to reuse it across calls.
"""
function projection_scratch(p::WaveProblem{T,D}, t::TransferTables{T,P1,Q}) where {T,D,P1,Q}
    E = p.fs.forest.N ÷ P1
    return allocate(get_backend(p.fs.work), T, ntuple(_ -> E * Q, D)..., nblocks(p.fs))
end

function check_tables(p::WaveProblem, t::TransferTables)
    order(t) == order(p.element) || throw(ArgumentError(
        "the transfer tables are for order P = $(order(t)), the problem's element for " *
        "P = $(order(p.element))"))
    return nothing
end

"""
    project_source!(du, p::WaveProblem, t::TransferTables, S, particles; ball,
                    rnear = 1, factor = -1, var = 2, scratch = nothing)

Add `factor · (P S)` to variable `var` of the state vector `du`, where
`(P S)ₙ = (1/wₙ) ∫_{element(n)} S φₙ` is the projection of the source `S`
onto the nodes (INSPIRAL.md, "The transfer operator"). The defaults
subtract it from `v`'s rate, as the remainder equations want.

- `S(x)` takes an `SVector{D}` and returns a number. It is a kernel
  argument, so it must be `isbits`.
- `particles` is a tuple of [`ParticleSupport`](@ref)s: where `S` is
  sharply peaked.
- `ball = (centre, R)` holds `S`'s support; elements that miss it are
  skipped, and blocks that miss it by a per-block mask.
- `rnear` is in units of the element width `hₑ`: elements closer than
  `rnear·hₑ` to a particle use the nested rule, those closer than
  `t.rfar·hₑ` the graded rule, and the others the far one. At the
  defaults the rules agree to `10⁻¹²` relative where they meet
  ("Quadrature" in `CODE.md`).

The far elements run as two kernels, one over the quadrature points and
one over the elements. The others, several hundred per particle, run on
the host, threaded over elements; each writes only its own nodes, so the
result does not depend on the thread count. An element near two
particles takes its rule from the nearer; one that two particles'
supports meet is refused.
"""
function project_source!(du, p::WaveProblem{T,D}, t::TransferTables, S, particles::Tuple;
                         ball, rnear::Real=1, factor=-one(T), var::Int=2,
                         scratch=nothing) where {T,D}
    check_tables(p, t)
    plan = projection_plan(p, t, particles, ball, rnear)
    scratch === nothing && (scratch = projection_scratch(p, t))
    dua = statearray(du, p.fs)
    project_far!(dua, plan, t, S, scratch, T(factor), var)
    project_graded!(dua, plan, t, S, T(factor), var)
    project_near!(dua, plan, t, S, T(factor), var)
    return du
end

# Where a projection's work is: the blocks' geometry on the host and on
# the kernels' backend, which blocks meet the source's ball, and the
# host's elements, nested and graded. Built once per call;
# `bin/projection.jl` times the passes that read it separately.
function projection_plan(p::WaveProblem{T,D}, t::TransferTables{T,P1}, particles, ball,
                         rnear) where {T,D,P1}
    0 <= rnear <= t.rfar || throw(ArgumentError(
        "rnear = $rnear must lie between 0 and the far rule's distance rfar = $(t.rfar)"))
    fs = p.fs
    forest = fs.forest
    backend = get_backend(fs.work)
    hₑ = minimum_spacing(forest) * P1
    rn = T(rnear) * hₑ
    rf = t.rfar * hₑ
    region = (SVector{D,T}(ball[1]), T(ball[2]))
    origins = block_origins(forest, T)
    spacings = block_spacings(forest, T)
    mask = map(eachindex(origins)) do b
        lo = SVector(origins[b])
        box_distance2(region[1], lo, lo .+ spacings[b] * forest.N, 1:D) < region[2]^2
    end
    near, graded = host_elements(fs, t, origins, spacings, mask, region, particles, rn, rf,
                                 Val(P1))
    return (; fs, backend, hₑ, rn, rf, region, particles, origins, spacings,
            dev_origins=to_backend(backend, origins),
            dev_spacings=to_backend(backend, spacings),
            dev_mask=to_backend(backend, mask), near, graded)
end

# The far elements: the source at their Gauss points, then the
# contraction into `dua`.
function project_far!(dua, plan, t::TransferTables{T,P1,Q}, S, scratch, factor,
                      var) where {T,P1,Q}
    (; fs, backend) = plan
    D = length(plan.region[1])
    E = fs.forest.N ÷ P1
    TreeAMR.launch_by_owner!(source_points_kernel!, backend, scratch, S, plan.dev_origins,
                             plan.dev_spacings, plan.dev_mask, t, plan.region,
                             plan.particles, plan.rf, Val(D), Val(P1);
                             ndrange=(ntuple(_ -> E * Q, D)..., nblocks(fs)))
    synchronize(backend)
    TreeAMR.launch_by_owner!(contract_kernel!, backend, dua, scratch, plan.dev_origins,
                             plan.dev_spacings, plan.dev_mask, t, plan.region,
                             plan.particles, plan.rf, factor, var, Val(D), Val(P1);
                             ndrange=(ntuple(_ -> E, D)..., nblocks(fs)))
    synchronize(backend)
    return nothing
end

# The graded elements, on the host, one band at a time, threaded over
# elements.
function project_graded!(dua, plan, t::TransferTables, S, factor, var)
    foreach((g, list) -> project_tensor!(dua, plan, t, g, list, S, factor, var), t.graded,
            plan.graded)
    return nothing
end

function project_tensor!(dua, plan, t::TransferTables{T,P1}, g::GaussRule, list, S, factor,
                         var) where {T,P1}
    isempty(list) && return nothing    # and the loop is not compiled
    Threads.@threads :dynamic for i in eachindex(list)
        D = length(plan.region[1])     # here, so that the closure knows it
        b, e = list[i]
        lo, hi = element_box(plan.origins[b], plan.spacings[b], e, Val(P1), Val(D))
        vals = tensor_rule(S, g, lo, hi, lo, Val(D))
        for (n, K) in enumerate(CartesianIndices(ntuple(_ -> P1, D)))
            node = ntuple(d -> e[d] * P1 + K[d], D)
            dua[node..., var, b] += factor * vals[n]
        end
    end
    return nothing
end

# The near elements, on the host. One element, the one holding the
# particle, takes about a third of the near work, so threading over
# elements left it on one thread whatever the thread count. The work is
# threaded over the points of each element's outermost line instead: each
# point's integral over the other dimensions is one item, and the items
# are summed per segment and the segments per element in the order the
# nested rule sums them, so the result does not depend on the thread
# count, and is the serial nested rule's to roundoff (the compiler may fuse
# a `muladd` in one and not in the other).
function project_near!(dua, plan, t::TransferTables{T,P1}, S, factor, var) where {T,P1}
    near = plan.near
    isempty(near) && return nothing    # and the loops are not compiled
    D = ndims(dua) - 2                 # from the type, so that `Val(D)` is static
    origins, spacings, particles = plan.origins, plan.spacings, plan.particles
    box(i) = element_box(origins[near[i][1]], spacings[near[i][1]], near[i][2], Val(P1),
                         Val(D))
    # The outermost line's segments of every element, as `(element, xs, ws)`.
    segments = Tuple{Int,Vector{T},Vector{T}}[]
    for i in eachindex(near)
        lo, hi = box(i)
        line_fold((list, xs, ws) -> push!(list, (i, collect(T, xs), collect(T, ws))),
                  segments, particles[near[i][3]], t, lo, hi, SVector{0,T}(), D == 1, false,
                  Val(D))
    end
    items = [(j, k) for j in eachindex(segments) for k in eachindex(segments[j][2])]
    wₙ = element_weights(t, plan.hₑ, Val(D))
    values = Vector{typeof(wₙ)}(undef, length(items))
    Threads.@threads :dynamic for n in eachindex(items)
        j, k = items[n]
        i = segments[j][1]
        lo, hi = box(i)
        integrand(x) = S(x) * element_basis(t, lo, hi, x)
        values[n] = nested_rule(integrand, zero(wₙ), particles[near[i][3]], t, lo, hi,
                                SVector{1,T}(segments[j][2][k]), Val(D - 1), false)
    end
    sums = fill(zero(wₙ), length(near))
    n = 0
    for (i, xs, ws) in segments
        acc = zero(wₙ)
        for k in eachindex(xs, ws)
            acc += ws[k] * values[n += 1]
        end
        sums[i] += acc
    end
    for (i, (b, e, _)) in enumerate(near)
        for (m, K) in enumerate(CartesianIndices(ntuple(_ -> P1, D)))
            node = ntuple(d -> e[d] * P1 + K[d], D)
            dua[node..., var, b] += factor * sums[i][m] / wₙ[m]
        end
    end
    return nothing
end

# The host's elements of the projection, found on the host: the nested
# rule's as `(block, element, particle)`, and the graded rule's as
# `(block, element)`, one list per band. The nested rule clusters toward
# one particle, the nearest: another particle within `rnear` is then at
# least about half their separation away, where the rule's 16 to 48 points
# per line resolve its field better than the graded rule's do. Only an
# element that two supports meet is refused, since the rule splits at one.
# The graded rule's band is chosen by the distance to the nearest
# particle.
function host_elements(fs::FieldSet{T,D}, t::TransferTables, origins, spacings, mask,
                       region, particles, rn, rf, ::Val{P1}) where {T,D,P1}
    E = fs.forest.N ÷ P1
    hₑ = minimum_spacing(fs.forest) * P1
    near = Tuple{Int,NTuple{D,Int},Int}[]
    graded = map(_ -> Tuple{Int,NTuple{D,Int}}[], t.graded)
    for b in 1:nblocks(fs)
        mask[b] || continue
        for J in CartesianIndices(ntuple(_ -> E, D))
            e = Tuple(J) .- 1
            lo, hi = element_box(origins[b], spacings[b], e, Val(P1), Val(D))
            element_class(lo, hi, region, particles, rf) == HOST_ELEMENT || continue
            dist² = map(s -> box_distance2(s.z, lo, hi, 1:D), particles)
            if minimum(dist²) >= rn^2
                δ = sqrt(minimum(dist²)) / hₑ
                band = something(findlast(<=(δ), t.gradedfrom), 1)
                push!(graded[band], (b, e))
                continue
            end
            count(k -> dist²[k] < particles[k].ℓ^2, eachindex(particles)) <= 1 ||
                throw(ArgumentError(
                    "the supports of two particles meet the element at $(Tuple(lo)), " *
                    "and the near rule splits at one: the particles are closer than an " *
                    "element"))
            push!(near, (b, e, argmin(dist²)))
        end
    end
    return near, graded
end

# The block and element that hold the point `x`, and the element's box.
# A point on a face between elements belongs to the upper one, as
# `locate_point` gives a point on a face between blocks to the upper block.
function locate_element(fs::FieldSet{T,D}, ::Val{P1}, x) where {T,D,P1}
    b = locate_point(fs.forest, Tuple(x))
    b === nothing && throw(ArgumentError("the point $(Tuple(x)) is outside the domain"))
    key = blockkey(fs, b)
    origin = block_origin(T, fs.forest, key)
    h = spacing(T, fs.forest, key)
    E = fs.forest.N ÷ P1
    e = ntuple(d -> clamp(floorint((x[d] - origin[d]) / (h * P1)), 0, E - 1), D)
    lo, hi = element_box(origin, h, e, Val(P1), Val(D))
    return b, e, lo, hi
end

# The element's nodal values of variable `var`, in the order of
# `element_basis`.
function element_values(ua, b, e::NTuple{D}, var, ::Val{P1}) where {D,P1}
    return SVector{P1^D}(ntuple(Val(P1^D)) do n
        K = CartesianIndices(ntuple(_ -> P1, Val(D)))[n]
        @inbounds ua[ntuple(d -> e[d] * P1 + K[d], Val(D))..., var, b]
    end)
end

"""
    support_elements(p::WaveProblem, s::ParticleSupport)

The elements that meet the box `[z - ℓ, z + ℓ]` around the particle's
support, at most `2^D` since `2ℓ < hₑ`, as `(block, element)` pairs with
zero-based element indices. Found from the box's corners through
`TreeAMR.locate_point` and the element arithmetic of
[`node_coordinates`](@ref), with no pass over the mesh. The box must lie
inside the domain: periodic images are not looked for.
"""
function support_elements(p::WaveProblem{T,D}, s::ParticleSupport{D}) where {T,D}
    P1 = order(p.element) + 1
    forest = p.fs.forest
    hₑ = minimum_spacing(forest) * P1
    2s.ℓ < hₑ || throw(ArgumentError(
        "the particle's support, 2ℓ = $(2s.ℓ) across, is not smaller than an " *
        "element, hₑ = $hₑ"))
    lo, hi = first.(forest.extents), last.(forest.extents)
    inside(d) = lo[d] <= s.z[d] - s.ℓ && s.z[d] + s.ℓ < hi[d]
    all(inside, 1:D) || throw(ArgumentError(
        "the particle's support around z = $(Tuple(s.z)) reaches past the domain's " *
        "faces; the transfer does not look for periodic images"))
    list = Tuple{Int,NTuple{D,Int}}[]
    for corner in CartesianIndices(ntuple(_ -> 2, D))
        x = SVector(ntuple(d -> s.z[d] + (corner[d] == 1 ? -s.ℓ : s.ℓ), D))
        b, e, _, _ = locate_element(p.fs, Val(P1), x)
        (b, e) in list || push!(list, (b, e))
    end
    return list
end

"""
    pairing(p::WaveProblem, t::TransferTables, u, K, s::ParticleSupport; var = 1)

`∫ K g_h`, the adjoint pairing `⟨P K, g⟩_w` of a particle kernel `K`
with variable `var` of the state vector `u` (INSPIRAL.md, "The transfer
operator"), over the elements that meet the particle's support
([`support_elements`](@ref)). `K(x)` vanishes outside the support and may
return a number or a static vector, for `K = ρ` and `K = ∇ρ`.

Each element is integrated by the nested rule split at the support, which
is exact up to the inner Gauss rule's order inside it, and the state is
read directly on the host. A `mesh_fold` of `wₙ (P K)ₙ gₙ` over every
node gives the same, at the cost of a pass over the mesh; the tests keep
it as the reference.
"""
function pairing(p::WaveProblem{T,D}, t::TransferTables{T,P1}, u, K,
                 s::ParticleSupport{D}; var::Int=1) where {T,D,P1}
    check_tables(p, t)
    ua = statearray(u, p.fs)
    acc = zero(K(s.z))
    for (b, e) in support_elements(p, s)
        key = blockkey(p.fs, b)
        lo, hi = element_box(block_origin(T, p.fs.forest, key),
                             spacing(T, p.fs.forest, key), e, Val(P1), Val(D))
        g = element_values(ua, b, e, var, Val(P1))
        acc += nested_rule(x -> K(x) * dot(g, element_basis(t, lo, hi, x)), zero(acc), s,
                           t, lo, hi, SVector{0,T}(), Val(D), true)
    end
    return acc
end

"""
    reconstruct(p::WaveProblem, t::TransferTables, u, x; var = 1)

`g_h(x) = Σₙ gₙ φₙ(x)`, variable `var` of the state vector `u` at the
point `x`: the element's polynomial, so that the result is exact for a
`g` that is a polynomial of degree `P` in each dimension and equals the
stored value at a node. On a face between elements the upper element's
polynomial is used.
"""
function reconstruct(p::WaveProblem{T,D}, t::TransferTables{T,P1}, u, x;
                     var::Int=1) where {T,D,P1}
    check_tables(p, t)
    xs = SVector{D,T}(Tuple(x))
    b, e, lo, hi = locate_element(p.fs, Val(P1), xs)
    g = element_values(statearray(u, p.fs), b, e, var, Val(P1))
    return dot(g, element_basis(t, lo, hi, xs))
end

"""
    reconstruct_gradient(p::WaveProblem, t::TransferTables, u, x; var = 1)

`∇g_h(x)`, the gradient of [`reconstruct`](@ref)'s polynomial, as an
`SVector{D}`.
"""
function reconstruct_gradient(p::WaveProblem{T,D}, t::TransferTables{T,P1}, u, x;
                              var::Int=1) where {T,D,P1}
    check_tables(p, t)
    xs = SVector{D,T}(Tuple(x))
    b, e, lo, hi = locate_element(p.fs, Val(P1), xs)
    g = element_values(statearray(u, p.fs), b, e, var, Val(P1))
    return SVector(ntuple(d -> dot(g, element_basis(t, lo, hi, xs, d)), Val(D)))
end

"""
    ball_rule(T, n_r, n_θ; precision = 512)

A cubature for the unit ball in three dimensions: `n_r` Gauss–Legendre
points in the radius, with the volume element `r²` in the weights, times
a product rule on the sphere of `n_θ` Gauss–Legendre points in `cos θ`
and `2n_θ` equally spaced points in `φ`. Returns the points as
`SVector{3,T}`s and the weights. It integrates `r^k Y_lm` exactly for
`k ≤ 2n_r - 3` (`k` even, or any `k` with the `r²` included) and
`l ≤ 2n_θ - 1`. For the averages over a particle's ball in its rest
frame (INSPIRAL.md, "Quadrature"), scaled by `ℓ`.
"""
function ball_rule(::Type{T}, n_r::Integer, n_θ::Integer; precision::Integer=512) where {T}
    xr, wr = gauss_nodes_weights(n_r; precision=precision)
    xμ, wμ = gauss_nodes_weights(n_θ; precision=precision)
    n_φ = 2n_θ
    points = SVector{3,T}[]
    weights = T[]
    setprecision(BigFloat, precision) do
        for i in 1:n_r, j in 1:n_θ, k in 1:n_φ
            r = (1 + xr[i]) / 2
            sθ = sqrt(1 - xμ[j]^2)
            φ = 2big(π) * (k - 1) / n_φ
            push!(points, SVector{3,T}(r * sθ * cos(φ), r * sθ * sin(φ), r * xμ[j]))
            push!(weights, T(wr[i] / 2 * r^2 * wμ[j] * 2big(π) / n_φ))
        end
    end
    return points, weights
end

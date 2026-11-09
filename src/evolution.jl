# The evolution system: the wave equation in 2nd-order form, with every
# spatial derivative the OLDG wide derivative of `element.jl`.
#
#     ∂ₜu = v
#     ∂ₜv = ∇²u = Σ_d D_d (D_d u)          (wave speed c = 1)
#
# Variable 1 is `u`, variable 2 is `v`. The Laplacian is the wide
# derivative applied twice through an auxiliary gradient field — the only
# second derivative that is summation-by-parts compatible with the first
# (arXiv:1604.00075, §II H) — so each right-hand side exchanges ghosts
# twice: once for `u`, then for its gradient, each component `∂_d u` along
# `d` only. See "The right-hand side" in `CODE.md`.
#
# `v` is the paper's *conjugate* variable (its right-hand side has spatial
# derivatives) and is held at one degree less than `u` by truncation
# (§II I): its top Legendre mode is set to zero in the state itself, in the
# initial data and at every stage of the time integrator, through the
# integrator's limiter hooks. The right-hand side does not truncate. See
# "Truncation" in `CODE.md`.

# The wide derivative along dimension `d` of variable `v` of block `b` at
# stored point `c`, whose owned index along `d` is `I`, in units of `1/h`.
# Its window is the node's element plus one point either side, so it starts
# one point before the element's first node.
Base.@propagate_inbounds function wide_derivative(a, c::NTuple{D,Int}, I::Int, v, b, d,
                                                  el::Element{T,P1,PW}) where {D,T,P1,PW}
    i = (I - 1) % P1
    start = c[d] - i - 1
    row = el.stencil[i + 1]
    acc = zero(eltype(a))
    for k in 1:PW
        acc += row[k] * a[Base.setindex(c, start + k - 1, d)..., v, b]
    end
    return acc
end

# `u` alone, from the state into its own field set's working array, for
# the exchange to fill its ghosts. `v` is never read across an element
# face, so it is never scattered: the Laplacian pass reads it from the
# state.
@kernel function scatter_u_kernel!(work, @Const(ua), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    @inbounds work[c..., 1, b] = ua[ntuple(d -> I[d], Val(D))..., 1, b]
end

# Pass 1: `∂_d u` at every owned point, into the working array of the
# `d`-th gradient field set, from which the exchange fills its ghosts.
# That set has a ghost plane along `d` only (`G = e_d`), so its stored
# index is the owned one plus one along `d`. `@inbounds` is an assertion
# that the window stays inside the stored block, which is what `G = 1` of
# `u`'s set guarantees for the first and last element: their windows reach
# exactly one ghost plane. CI-style runs with `--check-bounds=yes` re-check
# it. See "What bounds checking costs" in TreeWave's `CODE.md`.
#
# Both passes take `d` as a `Val` through an `@inline` closure, so that
# each dimension's index arithmetic is compiled with `d` a constant: a
# runtime loop over `d`, or a closure the compiler does not inline, makes
# the kernels two to six times slower (measured on the laptop, 3D, P = 3).
#
# On a geometry ([`Geometry`](@ref)) the `d`-th set holds the flux
# `F_d = A_d ∂_d u` instead, with `A_d = √g g^{dd}` read from the
# coefficient array `coef`, and the Laplacian pass divides by `√g`;
# `coef === nothing` is the flat case, where both helpers are the identity
# and the kernels compile to what they were. See "The metric" in `CODE.md`.
@inline flux(::Nothing, x, inner, b, d) = x
@inline flux(coef, x, inner, b, d) = @inbounds coef[inner..., 1 + d, b] * x
@inline divergence(::Nothing, x, inner, b) = x
@inline divergence(coef, x, inner, b) = @inbounds x / coef[inner..., 1, b]

@inline function gradient_term!(grads, work, c, I, b, el, h, coef, ::Val{d},
                                ::Val{D}) where {d,D}
    cd = ntuple(k -> I[k] + (k == d ? 1 : 0), Val(D))
    inner = ntuple(k -> I[k], Val(D))
    @inbounds grads[d][cd..., 1, b] = flux(coef,
                                           wide_derivative(work, c, I[d], 1, b, d, el) / h,
                                           inner, b, d)
    return nothing
end

@kernel function gradient_kernel!(grads, @Const(work), @Const(spacings), el, coef,
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    @inbounds h = spacings[b]
    ntuple(@inline(d -> gradient_term!(grads, work, c, I, b, el, h, coef, Val(d),
                                       Val(D))), Val(D))
end

# Pass 2: `∂ₜu = v`, and `∂ₜv = Σ_d ∂_d(∂_d u)` from the exchanged
# gradient. The `d`-th term reads `∂_d u` along `d` only, which is why its
# set needs ghosts along `d` only; `v` comes from the state `ua`.
@inline function laplacian_term(grads, I, b, el, ::Val{d}, ::Val{D}) where {d,D}
    cd = ntuple(k -> I[k] + (k == d ? 1 : 0), Val(D))
    return @inbounds wide_derivative(grads[d], cd, I[d], 1, b, d, el)
end

@kernel function laplacian_kernel!(du, @Const(ua), grads, @Const(spacings), el, coef,
                                   ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    terms = ntuple(@inline(d -> laplacian_term(grads, I, b, el, Val(d), Val(D))), Val(D))
    acc = foldl(+, terms; init=zero(eltype(du)))
    @inbounds h = spacings[b]
    @inbounds du[inner..., 1, b] = ua[inner..., 2, b]
    @inbounds du[inner..., 2, b] = divergence(coef, acc / h, inner, b)
end

# The same two passes on a full geometry ([`FullGeometry`](@ref)), whose
# glues may swap axes. A glue carries each variable onto itself, so the
# gradient crosses it as Cartesian components, `V = ∇u`, all `D` of them in
# one set with a ghost plane on every face. `coef` holds `√g` in variable
# 1 and the area vectors `S[d] = √g ∇ξ^d` after it, `S[d][i]` in variable
# `1 + (d-1)D + i`.
#
# Pass 1 forms every `D_e u` at the node, from `u`'s face ghosts along
# `e`, and writes `V_i = Σ_e S[e][i] D_e u / √g`. Pass 2 forms the logical
# flux `F^d = S[d] ⋅ V` at every point of the wide derivative's window
# along `d` and differentiates it: `√g v̇ = Σ_d D_d F^d`. A ghost of an
# element set sits at the same physical point as the block's own face node
# beside it, so the flux there is formed with that node's `S[d]`: the
# component normal to the face of the neighbour's `∇u`, which is the
# neighbour's own `F` across the glue to rounding. No metric is evaluated at
# a ghost. See "The seven-patch ball" in `CODE.md`.
@inline frame_coef(coef, idx, b, d, i, ::Val{D}) where {D} =
    @inbounds coef[idx..., 1 + (d - 1) * D + i, b]

@inline function cart_component(coef, Du, inner, b, ::Val{i}, ::Val{D}) where {i,D}
    terms = ntuple(@inline(e -> frame_coef(coef, inner, b, e, i, Val(D)) * Du[e]), Val(D))
    return foldl(+, terms) / (@inbounds coef[inner..., 1, b])
end

@kernel function cart_gradient_kernel!(gc, @Const(work), @Const(spacings), el, coef,
                                       ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    inner = ntuple(d -> I[d], Val(D))
    @inbounds h = spacings[b]
    Du = ntuple(@inline(e -> (@inbounds wide_derivative(work, c, I[e], 1, b, e, el)) / h),
                Val(D))
    ntuple(@inline(i -> (@inbounds gc[c..., i, b] = cart_component(coef, Du, inner, b,
                                                                   Val(i), Val(D)))),
           Val(D))
end

# `D_d F^d` at owned point `I`, with `F^d = S[d] ⋅ V` formed at each point
# of the window. `o` is the window point's owned index along `d`, `0` and
# `N + 1` for the ghosts, whose `S[d]` is the face node's beside them.
@inline function cart_flux_derivative(gc, coef, I, b, N, el::Element{T,P1,PW}, ::Val{d},
                                      ::Val{D}) where {T,P1,PW,d,D}
    i = (I[d] - 1) % P1
    row = el.stencil[i + 1]
    start = I[d] - i - 1
    acc = zero(eltype(gc))
    for k in 1:PW
        o = start + k - 1
        os = ntuple(q -> q == d ? clamp(o, 1, N) : I[q], Val(D))
        st = ntuple(q -> q == d ? o + 1 : I[q] + 1, Val(D))
        terms = ntuple(@inline(j -> frame_coef(coef, os, b, d, j, Val(D)) *
                                    (@inbounds gc[st..., j, b])), Val(D))
        acc += row[k] * foldl(+, terms)
    end
    return acc
end

@kernel function cart_laplacian_kernel!(du, @Const(ua), @Const(gc), @Const(spacings), el,
                                        coef, ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    N = size(coef, 1)
    terms = ntuple(@inline(d -> cart_flux_derivative(gc, coef, I, b, N, el, Val(d),
                                                     Val(D))), Val(D))
    acc = foldl(+, terms; init=zero(eltype(du)))
    @inbounds h = spacings[b]
    @inbounds du[inner..., 1, b] = ua[inner..., 2, b]
    @inbounds du[inner..., 2, b] = acc / h / coef[inner..., 1, b]
end

# Truncation: `a[…, var, b] ← (Π ⊗ … ⊗ Π) a[…, var, b]` in place, one work
# item per element (`J` is the element's index in its block). The
# one-dimensional projector is `Π = I - topmode ⊗ topdual`, so each
# dimension in turn is a dot product and a rank-one update along every line
# of the element: about `2D` multiply-adds per node, where the tensor
# product as a matrix would cost `(P+1)^D`. The dimensions commute, and
# the order is fixed, so the result does not depend on the thread count.
# `a` is in state layout, with no ghosts.
#
# `tables` is the element, whose `topmode` and `topdual` every line
# shares, or on a geometry a [`MetricTruncation`](@ref
# TreeWaveOLDG.MetricTruncation), whose rank-one pair is the line's own:
# per block, dimension and element index along it.
@inline topmode(el::Element, i, J, d, b) = @inbounds el.topmode[i]
@inline topdual(el::Element, j, J, d, b) = @inbounds el.topdual[j]

@kernel function truncate_kernel!(a, tables, ::Val{P1}, var, ::Val{D}) where {P1,D}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    base = ntuple(d -> (J[d] - 1) * P1, Val(D))
    for d in 1:D
        for K in CartesianIndices(ntuple(k -> k == d ? 1 : P1, Val(D)))
            c = zero(eltype(a))
            for j in 1:P1
                idx = ntuple(k -> base[k] + (k == d ? j : K[k]), Val(D))
                @inbounds c += topdual(tables, j, J, d, b) * a[idx..., var, b]
            end
            for i in 1:P1
                idx = ntuple(k -> base[k] + (k == d ? i : K[k]), Val(D))
                @inbounds a[idx..., var, b] -= topmode(tables, i, J, d, b) * c
            end
        end
    end
end

# The truncation on a full geometry, whose `√g` does not factorize: per
# element, `a ← Ψ G_e⁻¹ Ψᵀ W_e a`, the `√g`-orthogonal projection onto
# `Q_{P-1}`, with `Ψ` the tensor Legendre basis of `Q_{P-1}` at the
# element's nodes, `W_e` the quadrature weights times `√g`, and
# `G_e = Ψᵀ W_e Ψ = U_eᵀ U_e` through its Cholesky factor (see
# [`ElementTruncation`](@ref TreeWaveOLDG.ElementTruncation)). The
# coefficients go through the element's own column of `t.scratch`, which
# no other work item touches, and the two triangular solves run in place
# there.
@kernel function element_truncate_kernel!(a, t, ::Val{P1}, var, ::Val{D},
                                          E) where {P1,D}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    base = ntuple(d -> (J[d] - 1) * P1, Val(D))
    e = 1
    stride = 1
    for d in 1:D
        e += (J[d] - 1) * stride
        stride *= E
    end
    nodes = CartesianIndices(ntuple(_ -> P1, Val(D)))
    M = size(t.modes, 2)
    for q in 1:M                               # c = Ψᵀ W a
        c = zero(eltype(a))
        for (n, K) in enumerate(nodes)
            idx = ntuple(k -> base[k] + K[k], Val(D))
            @inbounds c += t.modes[n, q] * t.weights[n, e, b] * a[idx..., var, b]
        end
        @inbounds t.scratch[q, e, b] = c
    end
    for q in 1:M                               # Uᵀ y = c
        @inbounds c = t.scratch[q, e, b]
        for r in 1:(q - 1)
            @inbounds c -= t.chol[r, q, e, b] * t.scratch[r, e, b]
        end
        @inbounds t.scratch[q, e, b] = c / t.chol[q, q, e, b]
    end
    for q in M:-1:1                            # U z = y
        @inbounds c = t.scratch[q, e, b]
        for r in (q + 1):M
            @inbounds c -= t.chol[q, r, e, b] * t.scratch[r, e, b]
        end
        @inbounds t.scratch[q, e, b] = c / t.chol[q, q, e, b]
    end
    for (n, K) in enumerate(nodes)             # a = Ψ z
        s = zero(eltype(a))
        for q in 1:M
            @inbounds s += t.modes[n, q] * t.scratch[q, e, b]
        end
        idx = ntuple(k -> base[k] + K[k], Val(D))
        @inbounds a[idx..., var, b] = s
    end
end

"""
    WaveProblem(forest, el::Element; truncate=true, backend=CPU(), geometry=nothing)
    WaveProblem(fs::FieldSet, el::Element; truncate=true, backend, geometry=nothing)

Everything the right-hand side and the truncation need, built once:

- `fs`, the state field set (`u`, `v`), which says what the state vector
  is: its layout, the nodes the initial data are set at, and the
  quadrature. It has no ghosts (`G = 0`); nothing exchanges it, but
  `schedule`, its own empty ghost schedule with the element-local
  operators, is what TreeAMR's `regrid!` moves it with.
- `ufs`, `u` alone with one ghost plane on every side (`G = 1`), the
  input of the gradient pass, and its ghost schedule.
- `grads`, one field set per dimension: the `d`-th holds `∂_d u` with a
  ghost plane along `d` only (`G = e_d`), since the Laplacian reads it
  along `d` only, and each has its own schedule, because a schedule
  belongs to a layout.
- the per-block spacings on the kernels' backend, and the element.

That is all the exchange the wide derivative needs and no more: it reads
one point past its element along the dimension it differentiates, so `u`
needs its face ghosts, `∂_d u` the two faces normal to `d`, and `v` none.
Every set is element-centered at the element's order `P`, and the
schedules use TreeAMR's element-local operators, so the forest may be
refined (see [`check_layout`](@ref TreeWaveOLDG.check_layout)): across a
coarse-fine face a fine ghost is the coarse element's polynomial at the
fine nodes, and a coarse ghost the restriction that is its adjoint in the
quadrature, which keeps the discrete energy an exact invariant. An
element set's edge and corner ghosts are never written, and nothing here
reads them. See "The right-hand side" in `CODE.md`.

A forest with reflecting faces (a solid wall, decentered `reflecting` on
every face) makes the problem a reflecting box, and the field sets then
carry the reflection symmetries the wall imposes on the state and its
gradient. `u` and `v` are scalars, so they are `EvenParity` in every
dimension; `∂_d u` is odd under a reflection in `d` and even in the
others. TreeAMR's exchange fills the wall ghosts with these parities,
which is what makes `∂_n u = 0` (a Neumann or solid wall) hold discretely;
see "Reflecting walls" in `CODE.md`.

The first form builds the state set over `forest`. The second takes an
existing one, so that a problem can be rebuilt after `regrid!` has
replaced the set's storage on a new mesh: the state set and its data are
kept, and `ufs`, `grads`, their schedules and the spacings are built
afresh for the forest as it now is. The set must hold the two variables
of an element-centered set at the element's order, and `backend` is its
own, which is the default. [`adaptive_blast`](@ref) rebuilds its problem
this way after every regrid that changes the mesh.

`truncate` selects the paper's scheme (`true`: `v` held at degree `P - 1`)
or the plain one (`false`: both at degree `P`). The default is the paper.
It takes effect through [`initial_state`](@ref) and through
[`truncate_stage!`](@ref), which the integrator has to be given as its
limiter; [`wave_integrator`](@ref) installs it. "Truncation" in `CODE.md`
has why, and what each scheme costs.

`geometry`, a [`Geometry`](@ref) such as the [`PolarAnnulus`](@ref) of
[`annulus_forest`](@ref), maps the forest's logical coordinates to
physical space, and the problem is then the wave equation in that metric:
`√g ∂ₜv = Σ_d D_d F_d` with the flux `F_d = √g g^{dd} D_d u`, which the
gradient sets hold in place of `∂_d u`. The problem keeps the metric's
coefficients at the nodes, the truncation's per-line projections, which
are orthogonal in the `√g`-weighted inner product, and the largest
logical wave speed, which divides the time step. The forest must be
uniform and on the CPU; see "Multi-patch forests" in `CODE.md`. Without
a geometry the problem is the flat one, on a brick or on any forest of
patches, in logical coordinates.

On a [`FullGeometry`](@ref), such as the [`SevenPatchBall`](@ref) of
[`sevenpatch_forest`](@ref), the metric is general, and the glues may swap
axes, which a gradient set per dimension cannot cross. There `grads` is
empty, and `cartgrad`, one set of `D` variables with a ghost plane on
every face, holds `∇u` in Cartesian components, from which the Laplacian
pass forms the logical flux `F^d = √g ∇ξ^d ⋅ ∇u`; the truncation is a
`√g`-orthogonal projection per element. `boundary` gives the Dirichlet
data on the outer faces, as a callback `boundary(x, t) -> (u, ∂₁u, …,
∂_D u)` of the physical position and the time, which must be `isbits`:
the exchange sets `u`'s outer ghosts and the gradient's from it.
[`wave_rhs!`](@ref) passes its `t`. Or `boundary` is a
[`RadiativeBoundary`](@ref), which lets waves leave: its ghosts read the
interior and are written after each exchange. A full geometry with outer
faces needs one of the two, and nothing else takes either; see "The
seven-patch ball" in `CODE.md`.
"""
struct WaveProblem{T,D,F,S,U,GS,US,GSS,V,EL,M,CG,CS,B}
    fs::F
    schedule::S                  # the state set's, for `regrid!`'s transfer
    ufs::U
    grads::GS                    # NTuple{D}: `∂_d u`, ghosts along `d` only
    uschedule::US
    gradschedules::GSS
    spacings::V                  # per block, wherever the kernels run
    element::EL
    truncate::Bool
    metric::M                    # `nothing`, or a `Metric` on a geometry
    cartgrad::CG                 # on a full geometry `∇u`, Cartesian, `G = 1`
    cartschedule::CS
    boundary::B                  # on a full geometry Dirichlet data or
                                 # `RadiativeFaces`; or `nothing`
    valD::Val{D}
end

# What a geometry adds to the problem: its coefficients at the nodes, the
# truncation's per-line projections, and the largest logical wave speed.
struct Metric{G,C,A,TT,T}
    geometry::G
    cfs::C                       # `√g, A₁ … A_D` at the nodes, `G = 0`
    coef::A                      # its working array, which the kernels read
    truncation::TT
    speed::T                     # max over nodes and d of √(A_d/√g) = √g^{dd}
end

metric_coef(p) = p.metric === nothing ? nothing : p.metric.coef

"""
    MetricTruncation

The truncation on a geometry: per block `b`, dimension `d` and element
index `e` along `d`, the rank-one pair `modes[d][:, e, b]` and
`duals[d][:, e, b]` of `Π = I - mode ⊗ dual`, orthogonal in the inner
product weighted by the factor of `√g` along `d` (see
[`jacobian_factor`](@ref TreeWaveOLDG.jacobian_factor)). Their tensor
product is the `√g`-orthogonal projection onto `Q_{P-1}`.
"""
struct MetricTruncation{M}
    modes::M                     # NTuple{D} of (P+1, E, nblocks) arrays
    duals::M
end

@inline topmode(t::MetricTruncation, i, J, d, b) = @inbounds t.modes[d][i, J[d], b]
@inline topdual(t::MetricTruncation, j, J, d, b) = @inbounds t.duals[d][j, J[d], b]

# The callback that fills the coefficient set: `(√g, A₁ … A_D)`.
struct MetricTerms{G}
    geometry::G
end

@inline function (m::MetricTerms)(x, p)
    J, A = geometry_terms(m.geometry, x, p)
    return (J, A...)
end

# The degree-`P` polynomial at the nodes that is orthogonal to every
# polynomial of lower degree in the inner product `Σ ŵᵢ aᵢ fᵢ gᵢ`: `L_P`
# less its projection on `L₀ … L_{P-1}`. With `a` constant it is `L_P`.
function projector_mode(Lall, ŵ, a, P)
    P1 = P + 1
    inner(k, l) = sum(ŵ[i] * a[i] * Lall[i][k + 1] * Lall[i][l + 1] for i in 1:P1)
    G = [inner(k, l) for k in 0:(P - 1), l in 0:(P - 1)]
    rhs = [inner(P, k) for k in 0:(P - 1)]
    c = G \ rhs
    return [Lall[i][P + 1] - sum(c[k + 1] * Lall[i][k + 1] for k in 0:(P - 1))
            for i in 1:P1]
end

# The truncation's per-line projections on `geo`, built in `BigFloat`
# from the Gauss–Lobatto nodes and the factors of `√g` there, and rounded
# to `T` once ("Precision" in `CODE.md`).
function metric_truncation(geo, forest::Forest{D}, el::Element{T,P1},
                           backend) where {D,T,P1}
    P = P1 - 1
    E = forest.N ÷ P1
    origins = block_origins(forest, T)
    spacings = block_spacings(forest, T)
    patches = block_patches(forest)
    nb = length(spacings)
    modes = ntuple(_ -> Array{T}(undef, P1, E, nb), D)
    duals = ntuple(_ -> Array{T}(undef, P1, E, nb), D)
    ξ, ŵ = gll_nodes_weights(P)
    setprecision(BigFloat, 512) do
        Lall = [legendre_all(P, x) for x in ξ]
        offsets = [(P + 1) * (1 + x) / 2 for x in ξ]
        offsets[1], offsets[P1] = 0, P1
        for b in 1:nb, d in 1:D, e in 1:E
            a = [jacobian_factor(geo, Val(d),
                                 big(origins[b][d]) +
                                 big(spacings[b]) * ((e - 1) * P1 + offsets[i]),
                                 Int(patches[b])) for i in 1:P1]
            φ = projector_mode(Lall, ŵ, a, P)
            γ = sum(ŵ .* a .* φ .^ 2)
            modes[d][:, e, b] .= T.(φ)
            duals[d][:, e, b] .= T.(ŵ .* a .* φ ./ γ)
        end
    end
    return MetricTruncation(map(m -> to_backend(backend, m), modes),
                            map(m -> to_backend(backend, m), duals))
end

# What every geometry refuses: a forest it does not map, a refined one,
# and a device.
function check_metric_forest(geo, forest, backend)
    check_geometry(geo, forest)
    lv = unique(level.(forest.leaves))
    length(lv) == 1 || throw(ArgumentError(
        "a geometry takes a uniform forest, and this one has leaves on levels " *
        "$(sort(lv)): the coarse-fine coupling keeps the energy in the flat " *
        "weights, not the √g-weighted ones (\"Multi-patch forests\" in CODE.md)"))
    backend isa CPU || throw(ArgumentError(
        "a geometry runs on the CPU only, and the problem was asked for $backend: " *
        "its truncation tables are not adapted to a device"))
    return nothing
end

function build_metric(geo::Geometry, forest::Forest{D}, el::Element{T},
                      backend) where {D,T}
    check_metric_forest(geo, forest, backend)
    cfs = wave_fieldset(T, forest, 1 + D, ntuple(_ -> 0, D),
                        fill(EvenParity, 1 + D), order(el), backend)
    fill_by_coordinates!(WithPatch(AllVariables(MetricTerms(geo))), cfs)
    c = cfs.work
    speed = zero(T)
    for b in axes(c, D + 2), I in CartesianIndices(ntuple(_ -> forest.N, D)), d in 1:D
        speed = max(speed, sqrt(c[Tuple(I)..., 1 + d, b] / c[Tuple(I)..., 1, b]))
    end
    return Metric(geo, cfs, c, metric_truncation(geo, forest, el, backend), speed)
end

# What a full geometry adds: `√g` and the area vectors at the nodes, the
# per-element truncation, and the largest logical wave speed.
struct FullMetric{G,C,A,TT,T}
    geometry::G
    cfs::C                       # `√g, S[1]…, S[D]…` at the nodes, `G = 0`
    coef::A
    truncation::TT
    speed::T                     # max over nodes of √λ_max(g^{de})
end

"""
    ElementTruncation

The truncation on a full geometry, whose `√g` does not factorize, so that
no tensor product of one-dimensional projections is `√g`-orthogonal: per
element `e` of block `b`, the projection `Ψ G_e⁻¹ Ψᵀ W_e` onto `Q_{P-1}`.
`modes = Ψ` is the tensor Legendre basis of `Q_{P-1}` at the element's
nodes (`(P+1)^D × P^D`, the same for every element); `weights[:, e, b]`
is `W_e`, the element's quadrature weights on `[-1, 1]^D` times `√g`; and
`chol[:, :, e, b]` is the upper Cholesky factor `U_e` of the Gram matrix
`G_e = Ψᵀ W_e Ψ = U_eᵀ U_e`. Built in `BigFloat` and rounded once, the Gram
matrix by sum factorization over the dimensions. `scratch` holds each
element's `P^D` coefficients through the two triangular solves.
"""
struct ElementTruncation{A,B,C,S}
    modes::A                     # (n, m)
    weights::B                   # (n, E^D, nblocks)
    chol::C                      # (m, m, E^D, nblocks)
    scratch::S                   # (m, E^D, nblocks)
end

# The volume element alone, for the truncation's weights; a geometry may
# give a cheaper form than its frame.
@inline geometry_volume(geo::FullGeometry, x, p) = first(geometry_frame(geo, x, p))

# `G = Ψᵀ diag(w) Ψ` for the tensor basis `Ψ[n, q] = ∏_d L[n_d, q_d]`, by
# sum factorization: the sum over the nodes one dimension at a time,
# pairing each dimension's two basis indices, `pair[k, q + P(r-1)] =
# L[k, q] L[k, r]`. About `(P+1)^D P²` operations for the first dimension,
# where the dense product costs `(P+1)^D P^{2D}`.
function gram_sumfact(w::AbstractVector, pair::AbstractMatrix, P1, P, D)
    A = w
    R = 1
    for d in D:-1:1
        A3 = reshape(A, P1^(d - 1), P1, R)
        B = similar(A, P1^(d - 1), P^2, R)
        for j in 1:R, p in 1:(P^2), i in 1:(P1^(d - 1))
            B[i, p, j] = sum(A3[i, k, j] * pair[k, p] for k in 1:P1)
        end
        A = vec(B)
        R *= P^2
    end
    basis = CartesianIndices(ntuple(_ -> P, D))
    pairindex(q, r) = 1 + sum((q[d] - 1 + P * (r[d] - 1)) * (P^2)^(d - 1) for d in 1:D)
    return [A[pairindex(q, r)] for q in vec(basis), r in vec(basis)]
end

function element_truncation(geo, forest::Forest{D}, el::Element{T,P1},
                            backend) where {D,T,P1}
    P = P1 - 1
    E = forest.N ÷ P1
    origins = block_origins(forest, T)
    spacings = block_spacings(forest, T)
    patches = block_patches(forest)
    nb = length(spacings)
    nodes = CartesianIndices(ntuple(_ -> P1, D))
    basis = CartesianIndices(ntuple(_ -> P, D))
    n, m = length(nodes), length(basis)
    modes = Array{T}(undef, n, m)
    weights = Array{T}(undef, n, E^D, nb)
    chol = zeros(T, m, m, E^D, nb)
    ξ, ŵ = gll_nodes_weights(P)
    setprecision(BigFloat, 512) do
        Lall = [legendre_all(P, x) for x in ξ]
        offsets = [(P + 1) * (1 + x) / 2 for x in ξ]
        offsets[1], offsets[P1] = 0, P1
        Ψ = [prod(Lall[K[d]][Q[d]] for d in 1:D) for K in vec(nodes), Q in vec(basis)]
        pair = [Lall[k][(p - 1) % P + 1] * Lall[k][(p - 1) ÷ P + 1]
                for k in 1:P1, p in 1:(P^2)]
        ŵn = [prod(ŵ[K[d]] for d in 1:D) for K in vec(nodes)]
        modes .= T.(Ψ)
        for b in 1:nb, (e, Je) in enumerate(CartesianIndices(ntuple(_ -> E, D)))
            w = [ŵn[i] *
                 geometry_volume(geo,
                                 ntuple(d -> big(origins[b][d]) +
                                             big(spacings[b]) *
                                             ((Je[d] - 1) * P1 + offsets[K[d]]), D),
                                 Int(patches[b]))
                 for (i, K) in enumerate(vec(nodes))]
            G = gram_sumfact(w, pair, P1, P, D)
            weights[:, e, b] .= T.(w)
            chol[:, :, e, b] .= T.(cholesky(Symmetric(G)).U)
        end
    end
    return ElementTruncation(to_backend(backend, modes), to_backend(backend, weights),
                             to_backend(backend, chol),
                             to_backend(backend, zeros(T, m, E^D, nb)))
end

# The callback that fills a full geometry's coefficient set:
# `(√g, S[1]…, S[D]…)`.
struct FrameTerms{G}
    geometry::G
end

@inline flatframe(S::NTuple{D,NTuple{D,T}}) where {D,T} =
    ntuple(k -> S[(k - 1) ÷ D + 1][(k - 1) % D + 1], Val(D * D))

@inline function (m::FrameTerms)(x, p)
    J, S = geometry_frame(m.geometry, x, p)
    return (J, flatframe(S)...)
end

function build_metric(geo::FullGeometry, forest::Forest{D}, el::Element{T},
                      backend) where {D,T}
    check_metric_forest(geo, forest, backend)
    TreeAMR.hasreflecting(forest) && throw(ArgumentError(
        "a full geometry carries the gradient in Cartesian components, and the " *
        "forest has reflecting faces, whose mirror flips a logical component, not a " *
        "Cartesian one: give the outer faces Dirichlet data instead (`boundary`)"))
    cfs = wave_fieldset(T, forest, 1 + D * D, ntuple(_ -> 0, D),
                        fill(EvenParity, 1 + D * D), order(el), backend)
    fill_by_coordinates!(WithPatch(AllVariables(FrameTerms(geo))), cfs)
    c = cfs.work
    speed = zero(T)
    for b in axes(c, D + 2), I in CartesianIndices(ntuple(_ -> forest.N, D))
        J = c[Tuple(I)..., 1, b]
        S = ntuple(d -> ntuple(i -> c[Tuple(I)..., 1 + (d - 1) * D + i, b], D), D)
        g = SMatrix{D,D,T}(ntuple(k -> dot(S[(k - 1) % D + 1], S[(k - 1) ÷ D + 1]) / J^2,
                                  D * D))
        speed = max(speed, sqrt(maximum(eigvals(Symmetric(g)))))
    end
    return FullMetric(geo, cfs, c, element_truncation(geo, forest, el, backend), speed)
end

# The reflection symmetry a solid wall puts on each variable: both state
# variables are even scalars, and `∂_d u` is odd under a mirror in `d` and
# even in the others. Deriving it from the forest's faces means the caller
# only has to ask for a reflecting forest; the parity follows from the
# physics.
state_statesparity(::Val{D}) where {D} = [EvenParity, EvenParity]
state_gradparity(::Val{D}, d) where {D} =
    [ntuple(k -> (k == d ? OddParity : EvenParity), D)]

# An element-centered field set of order `P` over `forest` that carries
# `parity` only where the forest reflects; TreeAMR refuses a parity
# elsewhere.
function wave_fieldset(::Type{T}, forest::Forest{D}, nvars, G, parity, P,
                       backend) where {T,D}
    if TreeAMR.hasreflecting(forest)
        return FieldSet{T}(forest, nvars; G=G, centering=elementcentered(D), P=P,
                           parity=parity, backend=backend)
    else
        return FieldSet{T}(forest, nvars; G=G, centering=elementcentered(D), P=P,
                           backend=backend)
    end
end

function WaveProblem(forest::Forest{D}, el::Element{T}; truncate::Bool=true,
                     backend::Backend=CPU(), geometry=nothing,
                     boundary=nothing) where {D,T}
    check_layout(forest, el)
    fs = wave_fieldset(T, forest, 2, ntuple(_ -> 0, D), state_statesparity(Val(D)),
                       order(el), backend)
    return WaveProblem(fs, el; truncate=truncate, backend=backend, geometry=geometry,
                       boundary=boundary)
end

function WaveProblem(fs::FieldSet{T,D}, el::Element{T}; truncate::Bool=true,
                     backend::Backend=get_backend(fs.work),
                     geometry::Union{Nothing,Geometry}=nothing,
                     boundary=nothing) where {T,D}
    full = geometry isa FullGeometry
    boundary isa RadiativeBoundary && !full && throw(ArgumentError(
        "a radiative boundary forms the normal flux from the Cartesian gradient, which " *
        "a full geometry carries: pass a `FullGeometry` (`geometry`)"))
    boundary === nothing || full || throw(ArgumentError(
        "Dirichlet data (`boundary`) are given in Cartesian components and taken by a " *
        "full geometry only: without one the gradient sets hold logical components, " *
        "and the walls are TreeAMR's reflecting faces"))
    !full || boundary !== nothing || !geometry_boundary(geometry) || throw(ArgumentError(
        "the geometry has outer faces, and no data were given for them: pass " *
        "`boundary = (x, t) -> (u, ∂₁u, …, ∂_D u)`, the Dirichlet data at the " *
        "physical position x and time t, or `boundary = RadiativeBoundary()`"))
    forest = fs.forest
    check_layout(forest, el)
    check_element_set(fs, el)
    fs.nvars == 2 || throw(ArgumentError(
        "the state set holds $(fs.nvars) variables, and the wave equation's state is " *
        "two, u and v: build it with nvars = 2"))
    typeof(backend) == typeof(get_backend(fs.work)) || throw(ArgumentError(
        "the state set lives on $(get_backend(fs.work)), and the problem was asked " *
        "for $backend: the kernels read the state set's layout and the problem's " *
        "own sets together, so they must share a backend; omit `backend` to take " *
        "the set's"))
    P = order(el)
    ufs = wave_fieldset(T, forest, 1, ntuple(_ -> 1, D), [EvenParity], P, backend)
    grads = full ? () :
            ntuple(d -> wave_fieldset(T, forest, 1, ntuple(k -> Int(k == d), D),
                                      state_gradparity(Val(D), d), P, backend), Val(D))
    cartgrad = full ? wave_fieldset(T, forest, D, ntuple(_ -> 1, D),
                                    fill(EvenParity, D), P, backend) : nothing
    # Every operator of an element set follows from its `P`: injection
    # normal to a face, and tangentially the coarse element's polynomial
    # for a fine ghost and its adjoint restriction for a coarse one. The
    # state set has no ghosts, so its schedule fills nothing; `regrid!`
    # takes its operators for the transfer, which refines by the same
    # polynomial and coarsens by the exact `L²` projection.
    ops = Operators(family=ElementLocal)
    schedule = GhostSchedule(fs, ops)
    uschedule = GhostSchedule(ufs, ops)
    gradschedules = map(g -> GhostSchedule(g, ops), grads)
    cartschedule = cartgrad === nothing ? nothing : GhostSchedule(cartgrad, ops)
    spacings = to_backend(backend, block_spacings(forest, T))
    metric = geometry === nothing ? nothing : build_metric(geometry, forest, el, backend)
    boundary isa RadiativeBoundary &&
        (boundary = radiative_faces(boundary, geometry, uschedule, T, backend))
    return WaveProblem{T,D,typeof(fs),typeof(schedule),typeof(ufs),typeof(grads),
                       typeof(uschedule),typeof(gradschedules),typeof(spacings),
                       typeof(el),typeof(metric),typeof(cartgrad),typeof(cartschedule),
                       typeof(boundary)}(fs, schedule, ufs, grads, uschedule,
                                         gradschedules, spacings, el, truncate, metric,
                                         cartgrad, cartschedule, boundary, Val(D))
end

# The Dirichlet data on a full geometry's outer faces, as the two
# exchanges' boundary callbacks: `u` alone, and the `D` Cartesian
# components of its gradient, from `data(x, t) -> (u, ∂₁u, …, ∂_D u)` at
# the ghost's physical position. An element set's outer ghost sits at its
# face node, on the outer face.
struct DirichletValues{W,G,B,T}
    geometry::G
    data::B
    t::T
end

DirichletValues{W}(geo::G, data::B, t::T) where {W,G,B,T} =
    DirichletValues{W,G,B,T}(geo, data, t)

@inline (d::DirichletValues{:u})(x, p) =
    (first(d.data(physical_coordinates(d.geometry, x, p), d.t)),)
@inline (d::DirichletValues{:grad})(x, p) =
    Base.tail(d.data(physical_coordinates(d.geometry, x, p), d.t))

"""
    RadiativeBoundary(; κ=1, inverse_r=true)

The radiative, maximally dissipative condition on a full geometry's outer
faces, as [`WaveProblem`](@ref)'s `boundary`:

    n̂ ⋅ ∇u = -(κ ∂ₜu + u/R)

with `n̂` the outward unit normal and `R` the geometry's
[`outer_radius`](@ref TreeWaveOLDG.outer_radius); `inverse_r = false`
drops the `u/R` term, and then no radius is needed. `κ = 1` with the
`u/R` term is Sommerfeld's condition, which an outgoing monopole
`f(t - r)/r` satisfies exactly on the sphere `r = R`; `κ = 0` without it
is a wall that holds the normal derivative at zero.

The condition is imposed through the face's ghosts, which read the
interior: `u`'s ghost is the face node's own `u`, and the gradient's ghost
is chosen so that the central flux the wide derivative forms there is
`F* = S[k] ⋅ ∇u = -σ |S[k]| (κ v + u/R)` (`σ = ±1` the face's side). The
energy then changes at the rate `-Σ w |S[k]| (κ v² + u v/R)` over the face
nodes, so `E + ½ Σ w |S[k]| u²/R` never grows for `κ ≥ 0`; see
[`wave_energy`](@ref) and "The outer boundary" in `CODE.md`.
"""
struct RadiativeBoundary{T}
    κ::T
    inverse_r::Bool
    function RadiativeBoundary{T}(κ, inverse_r::Bool) where {T}
        κ >= 0 || throw(ArgumentError(
            "κ = $κ is negative, and a radiative boundary with κ < 0 feeds energy " *
            "in: take κ ≥ 0, κ = 1 for Sommerfeld's condition"))
        return new{T}(T(κ), inverse_r)
    end
end

RadiativeBoundary(; κ=1, inverse_r::Bool=true) =
    RadiativeBoundary{float(typeof(κ))}(κ, inverse_r)

"""
    outer_radius(geometry)

The radius of a full geometry's outer sphere, which the `u/R` term of a
[`RadiativeBoundary`](@ref) reads. [`SevenPatchBall`](@ref) gives its
`R1`.
"""
outer_radius(geo::FullGeometry) = throw(ArgumentError(
    "the geometry $(nameof(typeof(geo))) defines no outer radius, which the u/R term " *
    "of a radiative boundary needs: define `outer_radius` for it, or pass " *
    "`RadiativeBoundary(; inverse_r = false)`"))

# A radiative boundary as the problem holds it: `κ`, `1/R` (zero without
# the `u/R` term), and the outer faces as `(block, d, σ)`, one per block
# face of the domain's boundary, on the problem's backend.
struct RadiativeFaces{T,V}
    κ::T
    invR::T
    faces::V
end

# The outer faces from TreeAMR's own list of the ghost regions that face
# outside the domain: the regions across a face, whose direction has one
# nonzero component. An element set's edge and corner ghosts are never
# read, so their regions are left out.
function radiative_faces(rb::RadiativeBoundary, geo, schedule, ::Type{T},
                         backend) where {T}
    invR = rb.inverse_r ? inv(T(outer_radius(geo))) : zero(T)
    faces = NTuple{3,Int32}[]
    for r in schedule.boundaries
        δ = r.direction
        count(!iszero, δ) == 1 || continue
        d = findfirst(!iszero, δ)
        push!(faces, (Int32(r.block), Int32(d), Int32(δ[d])))
    end
    unique!(sort!(faces))
    return RadiativeFaces(T(rb.κ), invR, to_backend(backend, faces))
end

# The owned index of the face node beside the ghost at tangential position
# `I` (`D - 1` indices, the face's dimensions in order) of face `(d, σ)`.
@inline face_index(I, d, σ, N, ::Val{D}) where {D} =
    ntuple(q -> q < d ? I[q] : q == d ? (σ > 0 ? N : 1) : I[q - 1], Val(D))

radiative_u!(p::WaveProblem, boundary) = nothing

function radiative_u!(p::WaveProblem{T,D}, rf::RadiativeFaces) where {T,D}
    nf = length(rf.faces)
    nf == 0 && return nothing
    backend = get_backend(p.ufs.work)
    N = p.fs.forest.N
    TreeAMR.launch_by_owner!(radiative_u_kernel!, backend, p.ufs.work, rf.faces, N,
                             p.valD; ndrange=(ntuple(_ -> N, D - 1)..., nf))
    synchronize(backend)
    return nothing
end

# `u`'s ghost across an outer face is the face node's own value, so that
# the wide derivative's face correction vanishes there and `u̇`'s ghost is
# the face node's `v`. `u`'s set has `G = 1` on every face.
@kernel function radiative_u_kernel!(work, @Const(faces), N, ::Val{D}) where {D}
    J = @index(Global, NTuple)
    @inbounds b, d, σ = faces[J[D]]
    o = face_index(J, d, σ, N, Val(D))
    s = ntuple(q -> o[q] + 1, Val(D))
    g = Base.setindex(s, σ > 0 ? N + 2 : 1, d)
    @inbounds work[g..., 1, b] = work[s..., 1, b]
end

radiative_grad!(p::WaveProblem, boundary, ua) = nothing

function radiative_grad!(p::WaveProblem{T,D}, rf::RadiativeFaces, ua) where {T,D}
    nf = length(rf.faces)
    nf == 0 && return nothing
    backend = get_backend(p.cartgrad.work)
    N = p.fs.forest.N
    TreeAMR.launch_by_owner!(radiative_grad_kernel!, backend, p.cartgrad.work, ua,
                             p.metric.coef, rf.faces, rf.κ, rf.invR, N, p.valD;
                             ndrange=(ntuple(_ -> N, D - 1)..., nf))
    synchronize(backend)
    return nothing
end

# The gradient's ghost across an outer face: the face node's `V` with its
# component along `S[d]` changed so that `S[d] ⋅ V_g = 2F* - F`, where `F`
# is the face node's flux `S[d] ⋅ V`. The Laplacian forms the ghost's flux
# with the face node's `S[d]` and averages it with `F`, so the face's flux
# is `F*`. Only that component is read across the face.
@kernel function radiative_grad_kernel!(gc, @Const(ua), @Const(coef), @Const(faces), κ,
                                        invR, N, ::Val{D}) where {D}
    J = @index(Global, NTuple)
    @inbounds b, d, σ = faces[J[D]]
    o = face_index(J, d, σ, N, Val(D))
    s = ntuple(q -> o[q] + 1, Val(D))
    g = Base.setindex(s, σ > 0 ? N + 2 : 1, d)
    S = ntuple(i -> frame_coef(coef, o, b, d, i, Val(D)), Val(D))
    V = ntuple(i -> (@inbounds gc[s..., i, b]), Val(D))
    S2 = sum(S .* S)
    F = sum(S .* V)
    @inbounds Fstar = -σ * sqrt(S2) * (κ * ua[o..., 2, b] + invR * ua[o..., 1, b])
    c = 2 * (Fstar - F) / S2
    for i in 1:D
        @inbounds gc[g..., i, b] = V[i] + c * S[i]
    end
end

# The radiative boundary's own energy, `½ Σ w |S[d]| u²/R` over the outer
# faces' nodes, with `w` the face's quadrature weight: what makes
# `E + boundary_energy` non-increasing under the `u/R` term. Zero for any
# other boundary.
boundary_energy(p::WaveProblem, boundary, u) = zero(eltype(u))

function boundary_energy(p::WaveProblem{T,D}, rf::RadiativeFaces, u) where {T,D}
    iszero(rf.invR) && return zero(T)
    ua = Array(statearray(u, p.fs))
    coef = Array(p.metric.coef)
    faces = Array(rf.faces)
    forest = p.fs.forest
    N = forest.N
    el = p.element
    P1 = order(el) + 1
    total = zero(T)
    for (b, d, σ) in faces
        h = spacing(T, forest, blockkey(p.fs, Int(b)))
        for I in CartesianIndices(ntuple(_ -> N, D - 1))
            o = face_index(Tuple(I), d, σ, N, Val(D))
            w = prod(q -> q == d ? one(T) : h * P1 / 2 * el.weights[(o[q] - 1) % P1 + 1],
                     1:D)
            S = ntuple(i -> coef[o..., 1 + (d - 1) * D + i, b], D)
            total += w * sqrt(sum(abs2, S)) * ua[o..., 1, b]^2
        end
    end
    return total * rf.invR / 2
end

# One exchange of `fs`: as it always was when there are no Dirichlet data,
# and otherwise with the data at time `t` on the outer faces.
function exchange!(fs, schedule, p::WaveProblem{T}, t, ::Val{W}) where {T,W}
    p.boundary === nothing && return fill_ghosts!(fs, schedule)
    # The radiative ghosts read the interior, and are written after the
    # exchange by `radiative_u!` and `radiative_grad!`.
    p.boundary isa RadiativeFaces && return fill_ghosts!(fs, schedule)
    t === nothing && throw(ArgumentError(
        "the problem has Dirichlet data, which depend on the time, and none was " *
        "given: pass `t`"))
    hook = boundary_by_coordinates(WithPatch(AllVariables(
        DirichletValues{W}(p.metric.geometry, p.boundary, T(t)))))
    return fill_ghosts!(fs, schedule; boundary=hook)
end

# `u`'s exchange, pass 1 and the gradients' exchanges, shared by the RHS
# and the energy. `t` is the time, which only Dirichlet data read.
function fill_gradient!(p::WaveProblem, u, t=nothing)
    ua = statearray(u, p.fs)
    map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, ua, p.valD, Val(p.ufs.G))
    exchange!(p.ufs, p.uschedule, p, t, Val(:u))
    radiative_u!(p, p.boundary)
    gradient_pass!(p, p.metric, t, ua)
    return nothing
end

function gradient_pass!(p::WaveProblem, metric, t, ua)
    map_blocks!(gradient_kernel!, p.ufs, map(g -> g.work, p.grads), p.ufs.work,
                p.spacings, p.element, metric_coef(p), p.valD, Val(p.ufs.G))
    foreach(fill_ghosts!, p.grads, p.gradschedules)
    return nothing
end

function gradient_pass!(p::WaveProblem, metric::FullMetric, t, ua)
    map_blocks!(cart_gradient_kernel!, p.ufs, p.cartgrad.work, p.ufs.work, p.spacings,
                p.element, metric.coef, p.valD, Val(p.ufs.G))
    exchange!(p.cartgrad, p.cartschedule, p, t, Val(:grad))
    radiative_grad!(p, p.boundary, ua)
    return nothing
end

laplacian_pass!(du, u, p::WaveProblem, metric) =
    map_blocks!(laplacian_kernel!, p.fs, statearray(du, p.fs), statearray(u, p.fs),
                map(g -> g.work, p.grads), p.spacings, p.element, metric_coef(p), p.valD)

laplacian_pass!(du, u, p::WaveProblem, metric::FullMetric) =
    map_blocks!(cart_laplacian_kernel!, p.fs, statearray(du, p.fs), statearray(u, p.fs),
                p.cartgrad.work, p.spacings, p.element, metric.coef, p.valD)

"""
    wave_rhs!(du, u, p::WaveProblem, t)

The right-hand side, in the in-place form `f(du, u, p, t)`. `1 + D`
exchanges and three passes:

    u → ufs; fill_ghosts!(ufs)                  # u's face neighbours
    gradient pass; fill_ghosts!(grads[d]) ∀d    # ∂_d u's neighbours along d
    Laplacian pass                              # du, with v from u

It does not truncate: truncation acts on the state, through
[`truncate_stage!`](@ref). `u` is never written, and nothing is carried
between calls.

On a full geometry the gradient exchange is one, of `∇u` in Cartesian
components, so the count is two; `t` sets the Dirichlet data, if any.
"""
function wave_rhs!(du, u, p::WaveProblem, t)
    fill_gradient!(p, u, t)
    laplacian_pass!(du, u, p, p.metric)
    return nothing
end

"""
    truncate_state!(u, p::WaveProblem)

Set the top Legendre mode of `v` to zero in every element of the state
vector `u`, in place: `v ← (Π ⊗ … ⊗ Π) v`. `u`'s own variable is left
alone. Does nothing when `p` does not truncate.

One work item per element applies `Π` one dimension at a time, as a
rank-one update, on the thread that owns the element's block.
"""
function truncate_state!(u, p::WaveProblem{T,D}) where {T,D}
    p.truncate || return u
    ua = statearray(u, p.fs)
    E = p.fs.forest.N ÷ (order(p.element) + 1)
    backend = get_backend(ua)
    # `launch_by_owner!` is TreeAMR's internal launch that `map_blocks!`
    # goes through; an element-shaped `ndrange` keeps the block axis last,
    # so each block runs on the thread that owns it there too.
    tables = p.metric === nothing ? p.element : p.metric.truncation
    launch_truncation!(backend, ua, tables, p, E)
    synchronize(backend)
    return u
end

launch_truncation!(backend, ua, tables, p::WaveProblem{T,D}, E) where {T,D} =
    TreeAMR.launch_by_owner!(truncate_kernel!, backend, ua, tables,
                             Val(order(p.element) + 1), 2, p.valD;
                             ndrange=(ntuple(_ -> E, D)..., nblocks(p.fs)))

launch_truncation!(backend, ua, tables::ElementTruncation, p::WaveProblem{T,D},
                   E) where {T,D} =
    TreeAMR.launch_by_owner!(element_truncate_kernel!, backend, ua, tables,
                             Val(order(p.element) + 1), 2, p.valD, E;
                             ndrange=(ntuple(_ -> E, D)..., nblocks(p.fs)))

"""
    truncate_stage!(u, integrator, p::WaveProblem, t)

[`truncate_state!`](@ref) in the form of a limiter, the hook an
integrator calls on each stage value and on each step's result so that a
state can be put back where it is valid — TreeHydro resets its atmosphere
through the same hooks. [`wave_integrator`](@ref) passes it as both of
IMEXRungeKutta's:

    init(prob, RK4(); dt, stage_limiter = truncate_stage!,
         step_limiter = truncate_stage!)

IMEXRungeKutta calls the stage limiter on every stage value the
right-hand side reads except the first, which is the step's starting
state `uⁿ`, and the step limiter on the step's result. That result is
the next step's `uⁿ`, and [`initial_state`](@ref) truncates the first
one, so every state the right-hand side sees and every step's result has
`v` truncated, at one truncation per right-hand side.
"""
function truncate_stage!(u, integrator, p::WaveProblem, t)
    truncate_state!(u, p)
    return nothing
end

"""
    initial_state(p::WaveProblem, f)

The state vector for initial data `f(x, v) -> value` at the nodes, with
`v` truncated when `p` truncates: the integrator's limiters truncate each
stage and each step, but not the state it starts from, and the first
right-hand side is evaluated there.
"""
function initial_state(p::WaveProblem, f)
    fill_at_nodes!(f, p.fs, p.element)
    u0 = statevector(p.fs)
    gather!(u0, p.fs)
    return truncate_state!(u0, p)
end

"""
    wave_energy(p::WaveProblem, u)

The discrete energy `½ Σ w (v² + Σ_d (D_d u)²)`, a quadrature over the
nodes with the element weights, each block's scaled by its own spacing.
Summation by parts makes it an exact invariant of the semi-discrete
system on a periodic or reflecting box, uniform or refined (across a
coarse-fine face the coarse ghost is the adjoint of the fine one in this
inner product), and truncating `v` does not change that (the projector is
self-adjoint in it), so what a run loses is the time integrator's.

On a geometry it is `½ Σ w (√g v² + Σ_d F_d²/A_d)`, with `F_d = A_d D_d u`
the flux the gradient sets hold: `½ ∫ (v² + |∇u|²) dV` in the physical
metric. It is an exact invariant there too, across glued faces and
walls, with the truncation orthogonal in the `√g`-weighted inner product.

On a full geometry it is `½ Σ w √g (v² + |∇u|²)`, with `∇u` in Cartesian
components; it is invariant to rounding across the glued faces, and
changes through the outer faces by the flux the Dirichlet data let
through. `t` is the time of `u`, which those data read.

Under a [`RadiativeBoundary`](@ref) with its `u/R` term the energy
includes the boundary's own, `½ Σ w |S[k]| u²/R` over the outer faces'
nodes, and then never grows: its rate is `-κ Σ w |S[k]| v²` there.
"""
function wave_energy(p::WaveProblem{T,D}, u; t=nothing) where {T,D}
    fill_gradient!(p, u, t)
    if p.metric === nothing
        kinetic, _ = quadrature_parts(p.fs, p.element, u, 2, 2)
        potential = sum(g -> first(quadrature_parts(g, p.element, nothing, 2, 1)),
                        p.grads)
    elseif p.metric isa FullMetric
        geo = p.metric.geometry
        kinetic, _ = metric_parts(p.fs, p.element, geo, u, 2, 2, 0)
        potential, _ = metric_parts(p.cartgrad, p.element, geo, nothing, 2, 1:D, 0)
    else
        geo = p.metric.geometry
        kinetic, _ = metric_parts(p.fs, p.element, geo, u, 2, 2, 0)
        potential = sum(d -> first(metric_parts(p.grads[d], p.element, geo, nothing, 2,
                                                1, d)), 1:D)
    end
    return (kinetic + potential) / 2 + boundary_energy(p, p.boundary, u)
end

"""
    wave_norm(p::WaveProblem, u; p=2, vars=1:2)

The `p`-norm of the state vector `u` of `p`, normalized by the volume: on
a geometry weighted by `√g`, so in the physical volume, and otherwise
[`quadrature_norm`](@ref). `p = Inf` is the largest value at any node.
"""
function wave_norm(prob::WaveProblem{T,D}, u::AbstractVector; p::Real=2,
                   vars=1:2) where {T,D}
    prob.metric === nothing &&
        return quadrature_norm(prob.fs, prob.element, u; p=p, vars=vars)
    R = real(float(T))
    isinf(p) && return mesh_mapreduce(abs, max, zero(R), prob.fs, u; vars=vars)
    isinteger(p) && p >= 1 || throw(ArgumentError(
        "wave_norm takes an integer p ≥ 1 or Inf, got p = $p"))
    total, volume = metric_parts(prob.fs, prob.element, prob.metric.geometry, u, Int(p),
                                 vars, 0)
    volume == 0 && return zero(R)
    return (total / volume)^inv(R(p))
end

"""
    oldg_timestep(p::WaveProblem; cfl)

`cfl · hₑ / (P+1)²`, the DG time step (arXiv:1604.00075, eq. 64), for the
smallest element width `hₑ = (P+1)h`, the finest level's on a refined
forest. That is `cfl · h / (P+1)` in the mesh's spacing. On a geometry
`hₑ` is logical, and the step is divided by the largest logical wave
speed at any node: `√g^{dd}` for a diagonal metric, and the square root of
the largest eigenvalue of `g^{de}` on a full geometry. RK4 is stable up to `cfl ≈ 5` in one
dimension, measured for every `P` from 1 to 6, and the limit falls as `1/√D`; see "Time step"
in `CODE.md`.
"""
function oldg_timestep(p::WaveProblem; cfl)
    h = minimum_spacing(p.fs.forest)
    dt = cfl * h / (order(p.element) + 1)
    return p.metric === nothing ? dt : dt / p.metric.speed
end

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

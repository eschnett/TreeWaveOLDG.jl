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
"""
struct WaveProblem{T,D,F,S,U,GS,US,GSS,V,EL,M}
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

function build_metric(geo::Geometry, forest::Forest{D}, el::Element{T},
                      backend) where {D,T}
    check_geometry(geo, forest)
    lv = unique(level.(forest.leaves))
    length(lv) == 1 || throw(ArgumentError(
        "a geometry takes a uniform forest, and this one has leaves on levels " *
        "$(sort(lv)): the coarse-fine coupling keeps the energy in the flat " *
        "weights, not the √g-weighted ones (\"Multi-patch forests\" in CODE.md)"))
    backend isa CPU || throw(ArgumentError(
        "a geometry runs on the CPU only, and the problem was asked for $backend: " *
        "its truncation tables are not adapted to a device"))
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
                     backend::Backend=CPU(), geometry=nothing) where {D,T}
    check_layout(forest, el)
    fs = wave_fieldset(T, forest, 2, ntuple(_ -> 0, D), state_statesparity(Val(D)),
                       order(el), backend)
    return WaveProblem(fs, el; truncate=truncate, backend=backend, geometry=geometry)
end

function WaveProblem(fs::FieldSet{T,D}, el::Element{T}; truncate::Bool=true,
                     backend::Backend=get_backend(fs.work),
                     geometry::Union{Nothing,Geometry}=nothing) where {T,D}
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
    grads = ntuple(d -> wave_fieldset(T, forest, 1, ntuple(k -> Int(k == d), D),
                                      state_gradparity(Val(D), d), P, backend), Val(D))
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
    spacings = to_backend(backend, block_spacings(forest, T))
    metric = geometry === nothing ? nothing : build_metric(geometry, forest, el, backend)
    return WaveProblem{T,D,typeof(fs),typeof(schedule),typeof(ufs),typeof(grads),
                       typeof(uschedule),typeof(gradschedules),typeof(spacings),
                       typeof(el),typeof(metric)}(fs, schedule, ufs, grads, uschedule,
                                                  gradschedules, spacings, el, truncate,
                                                  metric, Val(D))
end

# `u`'s exchange, pass 1 and the gradients' exchanges, shared by the RHS
# and the energy.
function fill_gradient!(p::WaveProblem, u)
    map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, statearray(u, p.fs), p.valD,
                Val(p.ufs.G))
    fill_ghosts!(p.ufs, p.uschedule)
    map_blocks!(gradient_kernel!, p.ufs, map(g -> g.work, p.grads), p.ufs.work,
                p.spacings, p.element, metric_coef(p), p.valD, Val(p.ufs.G))
    foreach(fill_ghosts!, p.grads, p.gradschedules)
    return nothing
end

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
"""
function wave_rhs!(du, u, p::WaveProblem, t)
    fill_gradient!(p, u)
    map_blocks!(laplacian_kernel!, p.fs, statearray(du, p.fs), statearray(u, p.fs),
                map(g -> g.work, p.grads), p.spacings, p.element, metric_coef(p), p.valD)
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
    TreeAMR.launch_by_owner!(truncate_kernel!, backend, ua, tables,
                             Val(order(p.element) + 1), 2, p.valD;
                             ndrange=(ntuple(_ -> E, D)..., nblocks(p.fs)))
    synchronize(backend)
    return u
end

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
"""
function wave_energy(p::WaveProblem{T,D}, u) where {T,D}
    fill_gradient!(p, u)
    if p.metric === nothing
        kinetic, _ = quadrature_parts(p.fs, p.element, u, 2, 2)
        potential = sum(g -> first(quadrature_parts(g, p.element, nothing, 2, 1)),
                        p.grads)
    else
        geo = p.metric.geometry
        kinetic, _ = metric_parts(p.fs, p.element, geo, u, 2, 2, 0)
        potential = sum(d -> first(metric_parts(p.grads[d], p.element, geo, nothing, 2,
                                                1, d)), 1:D)
    end
    return (kinetic + potential) / 2
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
speed `√g^{dd}` at any node. RK4 is stable up to `cfl ≈ 5` in one
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

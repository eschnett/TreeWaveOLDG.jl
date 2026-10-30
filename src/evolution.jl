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
@inline function gradient_term!(grads, work, c, I, b, el, h, ::Val{d},
                                ::Val{D}) where {d,D}
    cd = ntuple(k -> I[k] + (k == d ? 1 : 0), Val(D))
    @inbounds grads[d][cd..., 1, b] = wide_derivative(work, c, I[d], 1, b, d, el) / h
    return nothing
end

@kernel function gradient_kernel!(grads, @Const(work), @Const(spacings), el,
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    @inbounds h = spacings[b]
    ntuple(@inline(d -> gradient_term!(grads, work, c, I, b, el, h, Val(d), Val(D))),
           Val(D))
end

# Pass 2: `∂ₜu = v`, and `∂ₜv = Σ_d ∂_d(∂_d u)` from the exchanged
# gradient. The `d`-th term reads `∂_d u` along `d` only, which is why its
# set needs ghosts along `d` only; `v` comes from the state `ua`.
@inline function laplacian_term(grads, I, b, el, ::Val{d}, ::Val{D}) where {d,D}
    cd = ntuple(k -> I[k] + (k == d ? 1 : 0), Val(D))
    return @inbounds wide_derivative(grads[d], cd, I[d], 1, b, d, el)
end

@kernel function laplacian_kernel!(du, @Const(ua), grads, @Const(spacings), el,
                                   ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    terms = ntuple(@inline(d -> laplacian_term(grads, I, b, el, Val(d), Val(D))), Val(D))
    acc = foldl(+, terms; init=zero(eltype(du)))
    @inbounds h = spacings[b]
    @inbounds du[inner..., 1, b] = ua[inner..., 2, b]
    @inbounds du[inner..., 2, b] = acc / h
end

# Truncation: `a[…, var, b] ← (Π ⊗ … ⊗ Π) a[…, var, b]` in place, one work
# item per element (`J` is the element's index in its block). The
# one-dimensional projector is `Π = I - topmode ⊗ topdual`, so each
# dimension in turn is a dot product and a rank-one update along every line
# of the element: about `2D` multiply-adds per node, where the tensor
# product as a matrix would cost `(P+1)^D`. The dimensions commute, and
# the order is fixed, so the result does not depend on the thread count.
# `a` is in state layout, with no ghosts.
@kernel function truncate_kernel!(a, el::Element{T,P1}, var, ::Val{D}) where {T,P1,D}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    base = ntuple(d -> (J[d] - 1) * P1, Val(D))
    for d in 1:D
        for K in CartesianIndices(ntuple(k -> k == d ? 1 : P1, Val(D)))
            c = zero(eltype(a))
            for j in 1:P1
                idx = ntuple(k -> base[k] + (k == d ? j : K[k]), Val(D))
                @inbounds c += el.topdual[j] * a[idx..., var, b]
            end
            for i in 1:P1
                idx = ntuple(k -> base[k] + (k == d ? i : K[k]), Val(D))
                @inbounds a[idx..., var, b] -= el.topmode[i] * c
            end
        end
    end
end

"""
    WaveProblem(forest, el::Element; truncate=true, backend=CPU())

Everything the right-hand side and the truncation need, built once:

- `fs`, the state field set (`u`, `v`), which says what the state vector
  is: its layout, the nodes the initial data are set at, and the
  quadrature. It has no ghosts (`G = 0`); nothing exchanges it.
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
The schedules' operators never run on a uniform forest, which is the
only kind this accepts (see [`check_layout`](@ref TreeWaveOLDG.check_layout)).
See "The right-hand side" in `CODE.md`.

A forest with reflecting faces (a solid wall, decentered `reflecting` on
every face) makes the problem a reflecting box, and the field sets then
carry the reflection symmetries the wall imposes on the state and its
gradient. `u` and `v` are scalars, so they are `EvenParity` in every
dimension; `∂_d u` is odd under a reflection in `d` and even in the
others. TreeAMR's exchange fills the wall ghosts with these parities,
which is what makes `∂_n u = 0` (a Neumann or solid wall) hold discretely;
see "Reflecting walls" in `CODE.md`.

`truncate` selects the paper's scheme (`true`: `v` held at degree `P - 1`)
or the plain one (`false`: both at degree `P`). The default is the paper.
It takes effect through [`initial_state`](@ref) and through
[`truncate_stage!`](@ref), which the integrator has to be given as its
limiter; [`wave_integrator`](@ref) installs it. "Truncation" in `CODE.md`
has why, and what each scheme costs.
"""
struct WaveProblem{T,D,F,U,GS,US,GSS,V,EL}
    fs::F
    ufs::U
    grads::GS                    # NTuple{D}: `∂_d u`, ghosts along `d` only
    uschedule::US
    gradschedules::GSS
    spacings::V                  # per block, wherever the kernels run
    element::EL
    truncate::Bool
    valD::Val{D}
end

# The reflection symmetry a solid wall puts on each variable: both state
# variables are even scalars, and `∂_d u` is odd under a mirror in `d` and
# even in the others. Deriving it from the forest's faces means the caller
# only has to ask for a reflecting forest; the parity follows from the
# physics.
state_statesparity(::Val{D}) where {D} = [EvenParity, EvenParity]
state_gradparity(::Val{D}, d) where {D} =
    [ntuple(k -> (k == d ? OddParity : EvenParity), D)]

# A field set over `forest` that carries `parity` only where the forest
# reflects; TreeAMR refuses a parity elsewhere.
function wave_fieldset(::Type{T}, forest::Forest{D}, nvars, G, parity,
                       backend) where {T,D}
    if TreeAMR.hasreflecting(forest)
        return FieldSet{T}(forest, nvars; G=G, centering=cellcentered(D),
                           parity=parity, backend=backend)
    else
        return FieldSet{T}(forest, nvars; G=G, centering=cellcentered(D),
                           backend=backend)
    end
end

function WaveProblem(forest::Forest{D}, el::Element{T}; truncate::Bool=true,
                     backend::Backend=CPU()) where {D,T}
    check_layout(forest, el)
    fs = wave_fieldset(T, forest, 2, ntuple(_ -> 0, D), state_statesparity(Val(D)),
                       backend)
    ufs = wave_fieldset(T, forest, 1, ntuple(_ -> 1, D), [EvenParity], backend)
    grads = ntuple(d -> wave_fieldset(T, forest, 1, ntuple(k -> Int(k == d), D),
                                      state_gradparity(Val(D), d), backend), Val(D))
    # The operators are a formality: they say how to fill a coarse-fine
    # ghost, and a uniform forest has none. Conservative order 1 is the one
    # family TreeAMR accepts for a ghost-free dimension, which the
    # gradients have.
    ops = Operators(family=Conservative, prolongation=1, restriction=2)
    uschedule = GhostSchedule(ufs, ops)
    gradschedules = map(g -> GhostSchedule(g, ops), grads)
    spacings = to_backend(backend, block_spacings(forest, T))
    return WaveProblem{T,D,typeof(fs),typeof(ufs),typeof(grads),typeof(uschedule),
                       typeof(gradschedules),typeof(spacings),
                       typeof(el)}(fs, ufs, grads, uschedule, gradschedules, spacings,
                                   el, truncate, Val(D))
end

# `u`'s exchange, pass 1 and the gradients' exchanges, shared by the RHS
# and the energy.
function fill_gradient!(p::WaveProblem, u)
    map_blocks!(scatter_u_kernel!, p.ufs, p.ufs.work, statearray(u, p.fs), p.valD,
                Val(p.ufs.G))
    fill_ghosts!(p.ufs, p.uschedule)
    map_blocks!(gradient_kernel!, p.ufs, map(g -> g.work, p.grads), p.ufs.work,
                p.spacings, p.element, p.valD, Val(p.ufs.G))
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
                map(g -> g.work, p.grads), p.spacings, p.element, p.valD)
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
    TreeAMR.launch_by_owner!(truncate_kernel!, backend, ua, p.element, 2, p.valD;
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
nodes with the element weights. Summation by parts makes it an exact
invariant of the semi-discrete system on a periodic box, and truncating
`v` does not change that (the projector is self-adjoint in this inner
product), so what a run loses is the time integrator's.
"""
function wave_energy(p::WaveProblem{T,D}, u) where {T,D}
    fill_gradient!(p, u)
    kinetic, _ = quadrature_parts(p.fs, p.element, u, 2, 2)
    potential = sum(g -> first(quadrature_parts(g, p.element, nothing, 2, 1)), p.grads)
    return (kinetic + potential) / 2
end

"""
    oldg_timestep(p::WaveProblem; cfl)

`cfl · hₑ / (P+1)²`, the DG time step (arXiv:1604.00075, eq. 64), for the
smallest element width `hₑ = (P+1)h`. That is `cfl · h / (P+1)` in the
mesh's spacing. RK4 is stable up to `cfl ≈ 5` in one dimension, measured
for every `P` from 1 to 6, and the limit falls as `1/√D`; see "Time step"
in `CODE.md`.
"""
function oldg_timestep(p::WaveProblem; cfl)
    h = minimum_spacing(p.fs.forest)
    return cfl * h / (order(p.element) + 1)
end

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

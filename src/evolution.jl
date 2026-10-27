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
# twice: once for `u`, once for its gradient. See "The right-hand side" in
# `CODE.md`.
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

# Pass 1: `∂_d u` at every owned point, into the gradient field set's
# working array, from which the exchange fills its ghosts. `@inbounds` is
# an assertion that the window stays inside the stored block, which is
# what `G = 1` guarantees for the first and last element: their windows
# reach exactly one ghost plane. CI-style runs with `--check-bounds=yes`
# re-check it. See "What bounds checking costs" in TreeWave's `CODE.md`.
@kernel function gradient_kernel!(grad, @Const(work), @Const(spacings), el,
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block), owned
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    @inbounds h = spacings[b]
    for d in 1:D
        @inbounds grad[c..., d, b] = wide_derivative(work, c, I[d], 1, b, d, el) / h
    end
end

# Pass 2: `∂ₜu = v`, and `∂ₜv = Σ_d ∂_d(∂_d u)` from the exchanged gradient.
@kernel function laplacian_kernel!(du, @Const(work), @Const(grad), @Const(spacings),
                                   el, ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G[d], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        @inbounds acc += wide_derivative(grad, c, I[d], d, b, d, el)
    end
    @inbounds h = spacings[b]
    @inbounds du[inner..., 1, b] = work[c..., 2, b]
    @inbounds du[inner..., 2, b] = acc / h
end

# Truncation: `dst[…, var, b] = (Π ⊗ … ⊗ Π) src`, the tensor product of the
# one-dimensional projector over the node's element. It reads the whole
# element, so `src` is a copy and not `dst` itself. `src` and `dst` are in
# state layout, with no ghosts.
@kernel function truncate_kernel!(dst, @Const(src), el::Element{T,P1}, var,
                                  ::Val{D}) where {T,P1,D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    i = ntuple(d -> (I[d] - 1) % P1, Val(D))
    start = ntuple(d -> I[d] - i[d], Val(D))
    acc = zero(eltype(dst))
    for J in CartesianIndices(ntuple(_ -> P1, Val(D)))
        w = one(acc)
        for d in 1:D
            @inbounds w *= el.truncate[i[d] + 1][J[d]]
        end
        @inbounds acc += w * src[ntuple(d -> start[d] + J[d] - 1, Val(D))..., b]
    end
    @inbounds dst[ntuple(d -> I[d], Val(D))..., var, b] = acc
end

"""
    WaveProblem(forest, el::Element; truncate=true, backend=CPU())

Everything the right-hand side and the truncation need, built once: the
state field set (`u`, `v`), the auxiliary gradient field set (`∂_d u`, one
variable per dimension), the one ghost schedule both share, the per-block
spacings on the kernels' backend, the scratch array truncation projects
from, and the element.

Both field sets are cell-centered with `G = 1` whatever the order: the
wide derivative reads one point past its element and no further. They
share a layout, and a schedule belongs to a layout, so one schedule
serves both. Its operators never run on a uniform forest, which is the
only kind this accepts (see [`check_layout`](@ref TreeWaveOLDG.check_layout)).

A forest with reflecting faces (a solid wall, decentered `reflecting` on
every face) makes the problem a reflecting box, and the field sets then
carry the reflection symmetries the wall imposes on the state and its
gradient. `u` and `v` are scalars, so they are `EvenParity` in every
dimension; the `d`-component of the gradient `∂_d u` is odd under a
reflection in `d` and even in the others. TreeAMR's exchange fills the
wall ghosts with these parities, which is what makes `∂_n u = 0` (a
Neumann or solid wall) hold discretely; see "Reflecting walls" in
`CODE.md`.

`truncate` selects the paper's scheme (`true`: `v` held at degree `P - 1`)
or the plain one (`false`: both at degree `P`). The default is the paper.
It takes effect through [`initial_state`](@ref) and through
[`truncate_stage!`](@ref), which the integrator has to be given as its
limiter; "Truncation" in `CODE.md` has why, and what each scheme costs.
"""
struct WaveProblem{T,D,F,S,V,A,EL}
    fs::F
    grad::F
    schedule::S
    spacings::V                  # per block, wherever the kernels run
    lap::A                       # (N, …, N, nblocks) copy of `v` to truncate from
    element::EL
    truncate::Bool
    valD::Val{D}
end

# The reflection symmetry a solid wall puts on each variable of the two
# field sets: both state variables are even scalars, and the `d`-th
# component of the gradient is odd under a mirror in `d` and even in the
# others. Deriving it from the forest's faces means the caller only has to
# ask for a reflecting forest; the parity follows from the physics.
state_statesparity(::Val{D}) where {D} = [EvenParity, EvenParity]
function state_gradparity(::Val{D}) where {D}
    return [ntuple(d -> (d == j ? OddParity : EvenParity), D) for j in 1:D]
end

function WaveProblem(forest::Forest{D}, el::Element{T}; truncate::Bool=true,
                     backend::Backend=CPU()) where {D,T}
    check_layout(forest, el)
    if TreeAMR.hasreflecting(forest)
        fs = FieldSet{T}(forest, 2; G=1, centering=cellcentered(D),
                         parity=state_statesparity(Val(D)), backend=backend)
        grad = FieldSet{T}(forest, D; G=1, centering=cellcentered(D),
                          parity=state_gradparity(Val(D)), backend=backend)
    else
        fs = FieldSet{T}(forest, 2; G=1, centering=cellcentered(D), backend=backend)
        grad = FieldSet{T}(forest, D; G=1, centering=cellcentered(D), backend=backend)
    end
    # The operators are a formality: they say how to fill a coarse-fine
    # ghost, and a uniform forest has none.
    schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
    spacings = to_backend(backend, block_spacings(forest, T))
    lap = allocate(backend, T, (ntuple(_ -> forest.N, D)..., nblocks(fs)))
    return WaveProblem{T,D,typeof(fs),typeof(schedule),typeof(spacings),
                       typeof(lap),typeof(el)}(fs, grad, schedule, spacings, lap, el,
                                               truncate, Val(D))
end

# Pass 1 and the exchange after it, shared by the RHS and the energy.
function fill_gradient!(p::WaveProblem, u)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(gradient_kernel!, p.fs, p.grad.work, p.fs.work, p.spacings,
                p.element, p.valD, Val(p.fs.G))
    fill_ghosts!(p.grad, p.schedule)
    return nothing
end

"""
    wave_rhs!(du, u, p::WaveProblem, t)

The right-hand side, in the SciML in-place form. Two exchanges and two
passes:

    scatter!(fs, u); fill_ghosts!(fs)      # u's face neighbours
    gradient pass; fill_ghosts!(grad)      # ∂_d u's face neighbours
    Laplacian pass                         # du

It does not truncate: truncation acts on the state, through
[`truncate_stage!`](@ref). `u` is never written, and nothing is carried
between calls.
"""
function wave_rhs!(du, u, p::WaveProblem, t)
    fill_gradient!(p, u)
    map_blocks!(laplacian_kernel!, p.fs, statearray(du, p.fs), p.fs.work,
                p.grad.work, p.spacings, p.element, p.valD, Val(p.fs.G))
    return nothing
end

"""
    truncate_state!(u, p::WaveProblem)

Set the top Legendre mode of `v` to zero in every element of the state
vector `u`, in place: `v ← (Π ⊗ … ⊗ Π) v`. `u`'s own variable is left
alone. Does nothing when `p` does not truncate.
"""
function truncate_state!(u, p::WaveProblem{T,D}) where {T,D}
    p.truncate || return u
    ua = statearray(u, p.fs)
    p.lap .= selectdim(ua, D + 1, 2)
    map_blocks!(truncate_kernel!, p.fs, ua, p.lap, p.element, 2, p.valD)
    return u
end

"""
    truncate_stage!(u, integrator, p::WaveProblem, t)

[`truncate_state!`](@ref) in the form of an OrdinaryDiffEq limiter, the
hook an integrator calls on each stage value and on each step's result
so that a state can be put back where it is valid — TreeHydro resets its
atmosphere through the same hooks. Pass it as both:

    RK4(stage_limiter! = truncate_stage!, step_limiter! = truncate_stage!)

RK4 calls the stage limiter on its three intermediate stage values and on
the step's result, so for RK4 the step limiter repeats the last
projection, which leaves the state unchanged to roundoff. It is passed
anyway so that a method that does not limit its result still truncates
it.
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
    kinetic, _ = quadrature_parts(p.fs, p.element, nothing, 2, 2)
    potential, _ = quadrature_parts(p.grad, p.element, nothing, 2, 1:D)
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

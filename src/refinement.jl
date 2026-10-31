# The refinement criterion: a per-element indicator, reduced to a
# per-block verdict for TreeAMR's `regrid!`.
#
# The split is TreeAMR's, as in TreeWave: the mesh takes a flag per block,
# the application reduces its own indicator to that flag, and the mesh
# dilates where the criterion fired by the buffer. What differs is the
# indicator. TreeWave's Löhner indicator is a second difference over three
# cells; an element here already holds a polynomial, and how well it
# resolves the data is how much of the data is in its last Legendre mode.
# For data the element resolves, the coefficients fall off geometrically,
# and the last one is what is left; for smooth data the coefficient of
# `L_m` falls as `hₑ^m`, so a fixed threshold makes refinement stop on its
# own.
#
# Which mode is the last one depends on the scheme. Without truncation it
# is `u`'s top mode, `P`. With it, `∂ₜu = v` and `v` has no mode of degree
# `P` in any dimension, so every coefficient of `u` that is of degree `P`
# along a line is constant in time: it is the initial data's (and what a
# regrid puts there), and it does not follow the wave. Measured on the
# blast wave, the top mode of a level-1 block that the ring had crossed
# was `7.03e-6` at `t = 0.1` and at `t = 0.4`, while the ring there had an
# amplitude of `0.14`; an indicator on it left the ring on coarse blocks
# and kept the start's refinement for good. The last mode that moves with
# the solution is `P - 1`, so the truncated scheme reads that one.
#
# Threading: TreeAMR calls `flag_blocks`' callback concurrently, one block
# per thread. So the indicator is not computed inside it: one kernel,
# launched by block owner, writes every element's amplitude, and the
# callback only reads its block's slice of that array.

# The amplitude of one Legendre mode of variable `var` along every line of
# one element, the largest over the element's lines in every dimension,
# into `τ[e1, …, eD, b]`. `dual` is the mode's row of `Element.duals`:
# along a line the element's polynomial is `Σ_k c_k L_k`, and `c_m = Σ_j
# dual[j] u_j` exactly, by the discrete orthogonality of the Legendre
# modes under the Gauss–Lobatto quadrature. For the top mode this is the
# dot product the truncation kernel forms with `topdual`. `|L_m| ≤ 1` on
# the element, with `|L_m(±1)| = 1`, so `|c_m|` is the mode's amplitude in
# the units of `u`. `G` is the set's ghost width, which offsets the owned
# nodes in the working array.
@kernel function mode_kernel!(τ, @Const(a), dual::NTuple{P1}, var, ::Val{D},
                              ::Val{G}) where {P1,D,G}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    base = ntuple(d -> (J[d] - 1) * P1 + G[d], Val(D))
    m = zero(eltype(τ))
    for d in 1:D
        for K in CartesianIndices(ntuple(k -> k == d ? 1 : P1, Val(D)))
            c = zero(eltype(τ))
            for j in 1:P1
                idx = ntuple(k -> base[k] + (k == d ? j : K[k]), Val(D))
                @inbounds c += dual[j] * a[idx..., var, b]
            end
            m = max(m, abs(c))
        end
    end
    @inbounds τ[ntuple(d -> J[d], Val(D))..., b] = m
end

"""
    modal_indicator(fs, el; mode, var=1) -> Array

The refinement indicator of every element of every block of the field
set `fs`: the largest amplitude of the Legendre mode `L_mode` of variable
`var` along any line of the element, in any dimension, divided by the
largest `|value|` of that variable over the whole mesh. An array
`τ[e1, …, eD, b]` on the host, with `E = N/(P+1)` elements per dimension
and one slice per block of this rank.

`mode` is the last Legendre mode that moves with the solution, and the
caller has to say which: `P - 1` for the paper's scheme, which truncates
`v` (with it every coefficient of `u` of degree `P` along a line is
constant in time, so the top mode would show the initial data and not
the wave), and `P` for the plain one. Mode `0`, an element's mean, says
nothing about resolution and is refused, and so is the truncated scheme
at `P = 1`.

For data of degree below `mode` along every line it vanishes, since the
mode is discretely orthogonal to them. For smooth data it falls as
`hₑ^mode`, the size of the element's Legendre coefficient of that degree,
so a fixed threshold makes refinement terminate where the data are
resolved; for data that are not, it is of the order of the data.

The reference amplitude is global, a `mesh_mapreduce`, so the call is
collective under MPI and the indicator does not depend on how the blocks
are split between ranks. A global scale rather than the element's own is
what keeps a negligible tail of the data from scoring like a feature:
TreeWave measured a scale-free Löhner indicator refining the whole
domain on the same blast wave (see `field_scales` in TreeWave). A field
that vanishes everywhere has an indicator of zero.

One kernel over every element, launched by block owner, then one copy to
the host, so it runs on any backend and the flagging that reads it
([`refine_flags`](@ref)) needs no field data.
"""
function modal_indicator(fs::FieldSet{T,D}, el::Element{T,P1}; mode::Integer,
                         var::Integer=1) where {T,D,P1}
    check_element_set(fs, el)
    check_mode(mode, P1 - 1)
    1 <= var <= fs.nvars || throw(ArgumentError(
        "var = $var is not a variable of this field set, which has $(fs.nvars)"))
    R = real(float(T))
    peak = mesh_mapreduce(abs, max, zero(R), fs; vars=var)
    E = fs.forest.N ÷ P1
    backend = get_backend(fs.work)
    τdev = allocate(backend, R, ntuple(_ -> E, D)..., nblocks(fs))
    TreeAMR.launch_by_owner!(mode_kernel!, backend, τdev, fs.work, el.duals[mode + 1],
                             Int(var), Val(D), Val(fs.G); ndrange=size(τdev))
    synchronize(backend)
    τ = Array(τdev)
    iszero(peak) || (τ ./= peak)
    return τ
end

function check_mode(mode, P)
    1 <= mode <= P || throw(ArgumentError(
        "mode = $mode is not a Legendre mode an element of order P = $P can be " *
        "refined by: take P - 1 = $(P - 1) for the scheme that truncates v, whose " *
        "u keeps its top mode constant in time, and P = $P for the plain one. " *
        "Mode 0 is the element's mean, which says nothing about resolution, so " *
        "the truncated scheme at P = 1 has no mode to refine by"))
    return nothing
end

"""
    refine_mark(τ, el, b, k; refine_tol, coarsen_tol, maxlevel_cap)

The regrid mark for block `b` with key `k`, from the block's slice of the
[`modal_indicator`](@ref) `τ`, in the form TreeAMR's `flag_blocks`
accepts: a bare `RegridFlag` or a `(flag, box)` pair. It is TreeWave's
`refine_mark` with the element as the unit:

    some element above refine_tol, level < cap  ->  (Refine, box)
    some element above coarsen_tol              ->  (Keep, box)
    none, level > 0                             ->  Coarsen
    otherwise                                   ->  Keep

with `box` the bounding box of the nodes of the elements above
`coarsen_tol`, in the block's owned indices `1:N`, as TreeAMR's flag
boxes are given. A box is what makes a block a source of the regrid
buffer.

The two thresholds mean different things. `refine_tol` is "the element
does not resolve its data": go finer. `coarsen_tol` is "there is
something here at all". The box is keyed on `coarsen_tol` and not on
`refine_tol` because a block refined to the cap has stopped firing
`refine_tol` there, and still has to ask for its own level around
itself, `(Keep, box)`, so that the margin travels with the feature. A
quiet block stays a bare `Keep` or `Coarsen`, since a box would recruit
its neighbours. The gap between the two thresholds is the dead band that
stops a block from flipping on alternate regrids. TreeWave's
`refine_mark` has the measurements behind each of these.

`maxlevel_cap` is a cap: at calibrated thresholds the indicator stops
refinement on its own before it binds.
"""
function refine_mark(τ::AbstractArray, el::Element{T,P1}, b::Integer,
                     k::MortonKey{D}; refine_tol, coarsen_tol,
                     maxlevel_cap::Integer) where {T,P1,D}
    check_thresholds(refine_tol, coarsen_tol)
    E = size(τ, 1)
    fire = false
    boxed = false
    lo = ntuple(_ -> E, Val(D))
    hi = ntuple(_ -> 1, Val(D))
    for e in CartesianIndices(ntuple(_ -> E, Val(D)))
        t = τ[e, b]
        fire |= t > refine_tol
        t > coarsen_tol || continue
        boxed = true
        lo = ntuple(d -> min(lo[d], e[d]), Val(D))
        hi = ntuple(d -> max(hi[d], e[d]), Val(D))
    end
    # Element `e` holds the owned nodes `(e - 1)(P + 1) + 1 … e(P + 1)`.
    box = boxed ? ntuple(d -> ((lo[d] - 1) * P1 + 1):(hi[d] * P1), Val(D)) : nothing
    if fire && level(k) < maxlevel_cap
        return (Refine, box)
    elseif boxed
        return (Keep, box)
    elseif level(k) > 0
        return Coarsen
    else
        return Keep
    end
end

function check_thresholds(refine_tol, coarsen_tol)
    coarsen_tol < refine_tol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie below refine_tol ($refine_tol): the " *
        "gap between them is the dead band that stops blocks from flipping between " *
        "levels on alternate regrids"))
    return nothing
end

"""
    refine_flags(fs, el; mode, refine_tol, coarsen_tol, maxlevel_cap, var=1)

[`refine_mark`](@ref) for every block of `fs`, as the flag vector TreeAMR's
`regrid!` takes, through `flag_blocks`. The indicator is
[`modal_indicator`](@ref) of Legendre mode `mode` of variable `var`, `u`
by default: `mode = P - 1` for the scheme that truncates `v`, and `P` for
the plain one. It is computed once for the whole pass before the
flagging, which TreeAMR runs concurrently over blocks; each callback then
only reads its own block's slice. The indicator's reference amplitude is
a global reduction, so the call is collective under MPI.

`fs` holds the state at the nodes: the state set after `scatter!`, or
after `fill_at_nodes!` in TreeAMR's `adapt_to_initial_data!`, whose
`flags` keyword takes `fs -> refine_flags(fs, el; …)`. No ghosts are
read, so none need filling.
"""
function refine_flags(fs::FieldSet, el::Element; mode::Integer, refine_tol, coarsen_tol,
                      maxlevel_cap::Integer, var::Integer=1)
    check_thresholds(refine_tol, coarsen_tol)
    τ = modal_indicator(fs, el; mode=mode, var=var)
    return flag_blocks(fs.forest) do b, k
        refine_mark(τ, el, b, k; refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                    maxlevel_cap=maxlevel_cap)
    end
end

"""
    refinement_buffer(forest, maxlevel_cap, travel)

The regrid buffer, in points, that covers a feature moving `travel` in
physical units between one regrid and the next, measured at the spacing
of level `maxlevel_cap`: `⌈travel/h⌉ + 1`. TreeWave's rule, unchanged.

The width is physics, feature speed times the regrid cadence, so it is
the application's to choose. It must exceed the motion it covers, hence
the `+ 1`: TreeAMR measured a margin narrower than the travel per
interval to be slightly worse than none. The spacing is the cap's rather
than the forest's current finest, which is coarse while the hierarchy is
still being built.

Recruitment reaches one ring of neighbours, so TreeAMR caps the buffer
at `N`: the feature may not move more than one finest-level block between
regrids. This refuses a wider one, naming that constraint.
"""
function refinement_buffer(forest::Forest, maxlevel_cap::Integer, travel::Real)
    h = spacing(forest, maxlevel_cap)
    points = ceilint(travel / h) + 1
    points <= forest.N || throw(ArgumentError(
        "a feature travelling $travel between regrids needs a $points-point margin " *
        "at level $maxlevel_cap (h = $h), which exceeds the block width N = " *
        "$(forest.N). Recruitment reaches one ring of neighbours, so the travel " *
        "must stay under one finest-level block width, $(forest.N * h): regrid " *
        "more often."))
    return points
end

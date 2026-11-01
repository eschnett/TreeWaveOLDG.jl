# The refinement criterion: a per-element indicator, reduced to a
# per-block verdict for TreeAMR's `regrid!`. See "Adaptive refinement" in
# `CODE.md`.
#
# The split is TreeAMR's, as in TreeWave: the mesh takes a flag per block,
# the application reduces its own indicator to that flag, and the mesh
# dilates where the criterion fired by the buffer. What differs is the
# indicator. TreeWave's Löhner indicator is a second difference over three
# cells; an element here already holds a polynomial, and how well it
# resolves the data is how much of the data is in its last Legendre modes.
# For smooth data the coefficient of `L_m` falls as `(hₑ/ℓ)^m`, with `ℓ`
# the data's length scale, so a fixed threshold makes refinement stop on
# its own.
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
# the solution is `M = P - 1`, so the truncated scheme reads that one.
#
# One mode alone is not enough before the mesh resolves the data: on an
# element wider than the feature the coefficients no longer fall with the
# degree, and any one of them can cross zero. The blast wave's Gaussian,
# centred on an element corner, scores `2.8e-2` in mode 2 at `hₑ = 1.56σ`
# against `5.7e-2` at half that width, so a threshold between the two
# passes the coarser mesh as resolved. So the indicator takes every mode
# that moves, `m = 1 … M`, each as the estimate it gives of the mode-`M`
# amplitude, `|c_m|^(M/m)`, since `|c_m| ≈ (hₑ/ℓ)^m`, and the largest of
# them. On resolved data that is close to `|c_M|` (within a factor 1.1 at
# `P = 3` and 2.4 at `P = 4` on the Gaussian); on unresolved data it would
# take every mode crossing zero at once to pass. `CODE.md` has the
# alternatives measured against it.
#
# Threading: TreeAMR calls `flag_blocks`' callback concurrently, one block
# per thread. So the indicator is not computed inside it: one kernel,
# launched by block owner, writes every element's amplitude, and the
# callback only reads its block's slice of that array.

# The scaled amplitudes of the Legendre modes `duals` of variable `var`
# along every line of one element, the largest over the element's lines
# in every dimension and over the modes, into `τ[e1, …, eD, b]`. `duals[q]`
# is a row of `Element.duals`: along a line the element's polynomial is
# `Σ_k c_k L_k`, and `c_m = Σ_j dual[j] u_j` exactly, by the discrete
# orthogonality of the Legendre modes under the Gauss–Lobatto quadrature.
# For the top mode this is the dot product the truncation kernel forms
# with `topdual`. `|L_m| ≤ 1` on the element, with `|L_m(±1)| = 1`, so
# `|c_m|` is the mode's amplitude in the units of `u`. It enters as
# `post (weights[q] |c_m|)^pows[q]`, the weight including one over the
# global peak. `G` is the set's ghost width, which offsets the owned nodes
# in the working array.
@kernel function mode_kernel!(τ, @Const(a), duals::NTuple{Q,NTuple{P1}}, weights::NTuple{Q},
                              pows::NTuple{Q}, post, var, ::Val{D},
                              ::Val{G}) where {Q,P1,D,G}
    J = @index(Global, NTuple)                 # (e1..eD, block)
    b = J[D + 1]
    base = ntuple(d -> (J[d] - 1) * P1 + G[d], Val(D))
    m = zero(eltype(τ))
    for d in 1:D
        for K in CartesianIndices(ntuple(k -> k == d ? 1 : P1, Val(D)))
            for q in 1:Q
                dual = duals[q]
                c = zero(eltype(τ))
                for j in 1:P1
                    idx = ntuple(k -> base[k] + (k == d ? j : K[k]), Val(D))
                    @inbounds c += dual[j] * a[idx..., var, b]
                end
                m = max(m, (weights[q] * abs(c))^pows[q])
            end
        end
    end
    @inbounds τ[ntuple(d -> J[d], Val(D))..., b] = post * m
end

# One kernel over every element of `fs`: the largest over the modes
# `modes` of `post (weights[q] |c_m|/A)^pows[q]`, with `A` the global peak
# of variable `var`; the result on the host.
function scaled_modes(fs::FieldSet{T,D}, el::Element{T,P1}, modes::NTuple{Q,Int},
                      weights::NTuple{Q}, pows::NTuple{Q}, post,
                      var::Integer) where {T,D,P1,Q}
    1 <= var <= fs.nvars || throw(ArgumentError(
        "var = $var is not a variable of this field set, which has $(fs.nvars)"))
    R = real(float(T))
    peak = mesh_mapreduce(abs, max, zero(R), fs; vars=var)
    # A field that vanishes everywhere scores zero rather than NaN.
    scale = iszero(peak) ? zero(R) : inv(peak)
    E = fs.forest.N ÷ P1
    backend = get_backend(fs.work)
    τ = allocate(backend, R, ntuple(_ -> E, D)..., nblocks(fs))
    TreeAMR.launch_by_owner!(mode_kernel!, backend, τ, fs.work,
                             map(m -> el.duals[m + 1], modes),
                             map(w -> R(w) * scale, weights), map(R, pows), R(post),
                             Int(var), Val(D), Val(fs.G); ndrange=size(τ))
    synchronize(backend)
    return Array(τ)
end

"""
    modal_indicator(fs, el; mode, var=1) -> Array

The refinement indicator of every element of every block of the field
set `fs`, an array `τ[e1, …, eD, b]` on the host, with `E = N/(P+1)`
elements per dimension and one slice per block of this rank. With
`c_m` the coefficient of the Legendre mode `L_m` of variable `var` along
a line of the element, and `A` the largest `|value|` of that variable
over the whole mesh, it is

    τ = max over the element's lines in every dimension, and m = 1 … mode,
        of (|c_m|/A)^(mode/m).

`mode` is the last Legendre mode that moves with the solution, and the
caller has to say which: `P - 1` for the paper's scheme, which truncates
`v` (with it every coefficient of `u` of degree `P` along a line is
constant in time, so the top mode would show the initial data and not
the wave), and `P` for the plain one. Mode `0`, an element's mean, says
nothing about resolution and is refused, and so is the truncated scheme
at `P = 1`.

Each mode enters as the estimate it gives of mode `mode`'s amplitude:
for smooth data `|c_m| ≈ (hₑ/ℓ)^m`, with `ℓ` the data's length scale, so
`|c_m|^(mode/m)` falls as `hₑ^mode` for every `m`, and so does `τ`. A
fixed threshold therefore makes refinement terminate where the data are
resolved. The lower modes are there for data that are not: on an
element wider than the feature the coefficients no longer fall with the
degree, and a single one can cross zero, which a single-mode indicator
reads as resolved (measured on the blast wave's Gaussian: see "Adaptive
refinement" in `CODE.md`). `τ` vanishes on data that are constant along
every line of the element, and only there. [`mode_amplitude`](@ref) is
one mode's amplitude alone.

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
    M = Int(mode)
    modes = ntuple(identity, M)
    return scaled_modes(fs, el, modes, map(doublefactorial, modes), map(m -> M // m, modes),
                        1 // doublefactorial(M), var)
end

"""
    mode_amplitude(fs, el; mode, var=1) -> Array

The amplitude of the single Legendre mode `L_mode` of variable `var`,
the largest `|c_mode|` along any line of each element in any dimension,
relative to the largest `|value|` of that variable over the mesh, in the
layout of [`modal_indicator`](@ref), which is the largest of these over
the modes `1 … mode`, each raised to `mode/m`. It vanishes on data of
degree below `mode` along every line, since the mode is discretely
orthogonal to them, and for smooth data it falls as `hₑ^mode`.
Collective under MPI, like the indicator. Any mode `0 … P` is accepted.
"""
function mode_amplitude(fs::FieldSet{T,D}, el::Element{T,P1}; mode::Integer,
                        var::Integer=1) where {T,D,P1}
    check_element_set(fs, el)
    0 <= mode <= P1 - 1 || throw(ArgumentError(
        "mode = $mode is not a Legendre mode of an element of order P = $(P1 - 1)"))
    return scaled_modes(fs, el, (Int(mode),), (1,), (1,), 1, var)
end

# `(2m - 1)!! = 1 · 3 · 5 ⋯ (2m - 1)`: a smooth `f` on an element of width
# `hₑ` has the Legendre coefficient `c_m ≈ (hₑ/2)^m f⁽ᵐ⁾/(2m - 1)!!`.
doublefactorial(m) = prod(1:2:(2m - 1); init=1)

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
[`modal_indicator`](@ref) up to Legendre mode `mode` of variable `var`,
`u` by default: `mode = P - 1` for the scheme that truncates `v`, and
`P` for the plain one. It is computed once for the whole pass before the
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

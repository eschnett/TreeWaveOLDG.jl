# TreeWaveOLDG design

The scalar wave equation in 2nd-order form, discretized with the
operator-based local discontinuous Galerkin method of Miller and
Schnetter (OLDG, [arXiv:1604.00075](https://arxiv.org/abs/1604.00075);
"the paper" below), on a [TreeAMR](https://github.com/eschnett/TreeAMR.jl)
mesh. It is [TreeWave](https://github.com/eschnett/TreeWave.jl) with the
finite-difference stencil replaced by the OLDG one. It began as the
*unrefined prototype* that TreeAMR's `CODE.md` proposed under
"Element-local DG stencils", to judge the method on a uniform forest
before TreeAMR grew what a refined forest needs. TreeAMR 0.2.7 has it,
the element centering (its milestone M16), and the field now runs on
static refined forests too; see [Refined forests](#refined-forests).

**Status.** Periodic or reflecting boxes, uniform or with a static
refinement of any number of levels, in any number of dimensions, any order `P ≥ 1`,
two initial conditions (a standing sine mode and a 2D radial blast wave)
and a viewer for the second, on the CPU, stepped by IMEXRungeKutta's RK4
with its stage arithmetic run by block owner. On a refined forest the
discrete energy stays an exact invariant and the L2 error converges at
order `P`; in `D ≥ 2` the L∞ error loses part of an order at the
coarse-fine faces **(open)**. The forest can also follow the solution:
a modal indicator, a regrid driver in chunks and an adaptive blast wave,
which matches the uniform run at its finest spacing to `0.3%` at 61–90%
of its work; see [Adaptive refinement](#adaptive-refinement). The
first parts of `INSPIRAL.md`'s inspiral exist: the analytic particle
(`INSPIRAL-PLAN.md`'s M1), its field, the field's time derivative, the
effective source and the window, the transfer between particles and
grid (M2), the projection of a source onto the nodes and its adjoint, and
the coupled right-hand side for particles on a prescribed path (M3), in
which a particle at rest or in uniform motion keeps its remainder; see
[Particles](#particles). The particles run on uniform forests and on
static nested boxes whose finest level holds the source's ball (M3b,
measured on Symmetry, with the layout M4 to M7 run on); see [Nested
boxes](#nested-boxes). A particle on a prescribed path with known
acceleration, the slow circle and hyperbolic motion, runs against the
retarded field of a point charge (M4): the circle's tangential force is
radiation reaction and its radial force zero, both converging, since
the particle field carries the `t′²` term of `Φ_C`'s lab slice; and
since the source is written and the force read through the averaging
over shared nodes, face crossings stay within radiation reaction from
`d/12` on, where the transfer alone gave impulses of tens to hundreds
of times it. M4's gate passes; see [Prescribed
paths](#prescribed-paths) and [The averaging](#the-averaging). The suite passes, also with
`--check-bounds=yes` and with four threads.

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [The equation](#the-equation)
- [The method](#the-method)
  - [The reference element](#the-reference-element)
  - [The wide derivative](#the-wide-derivative)
  - [Second derivatives](#second-derivatives)
  - [Summation by parts and the energy](#summation-by-parts-and-the-energy)
- [The layout](#the-layout)
  - [Positions and quadrature](#positions-and-quadrature)
- [The right-hand side](#the-right-hand-side)
- [Reflecting walls](#reflecting-walls)
- [Refined forests](#refined-forests)
  - [The coupling](#the-coupling)
  - [The energy](#the-energy)
  - [Accuracy](#accuracy)
  - [What a refined forest does not run](#what-a-refined-forest-does-not-run)
- [Adaptive refinement](#adaptive-refinement)
  - [The indicator](#the-indicator)
  - [The verdict](#the-verdict)
  - [The driver](#the-driver)
- [Truncation](#truncation)
- [Time step](#time-step)
- [The blast wave](#the-blast-wave)
- [The viewer](#the-viewer)
- [Precision](#precision)
- [Threads and devices](#threads-and-devices)
  - [The integrator](#the-integrator)
- [Particles](#particles)
  - [The particle](#the-particle)
  - [The near field](#the-near-field)
  - [The split](#the-split)
  - [The window](#the-window)
  - [The transfer](#the-transfer)
  - [The averaging](#the-averaging)
  - [Quadrature](#quadrature)
  - [The right-hand side with particles](#the-right-hand-side-with-particles)
  - [Nested boxes](#nested-boxes)
  - [Prescribed paths](#prescribed-paths)
- [File layout](#file-layout)
- [Tests](#tests)
- [Measured results](#measured-results)
  - [Convergence](#convergence)
  - [Reflecting walls](#reflecting-walls-1)
  - [Refined forests](#refined-forests-1)
  - [The RK4 limit](#the-rk4-limit)
  - [The blast wave's error](#the-blast-waves-error)
  - [Adaptive refinement](#adaptive-refinement-1)
  - [Threads on Symmetry](#threads-on-symmetry)
  - [The source bracket](#the-source-bracket)
  - [The transfer's cost](#the-transfers-cost)
  - [Uniform motion](#uniform-motion)
  - [Nested boxes](#nested-boxes-1)
  - [Prescribed paths](#prescribed-paths-1)
- [What carried over from TreeWave, and what did not](#what-carried-over-from-treewave-and-what-did-not)
- [Possible extensions](#possible-extensions)
- [Open questions](#open-questions)

## Goals

- Implement the paper's derivative operator on TreeAMR's mesh, through
  TreeAMR's public API, with no mesh machinery here.
- Measure what the method does on the wave equation: its order, its
  energy behaviour, its time step, and what truncation does, so that
  the TreeAMR work a refined forest needs is judged on numbers.
- Keep TreeWave's properties where they cost nothing: generic in the
  floating-point type, KernelAbstractions kernels launched through
  `map_blocks!`, a right-hand side that is a pure function of the state.

## Scope and non-goals

- **Refined forests, static or adaptive, for the field.** A forest may
  be uniform or refined: `wave_forest(…; refined)` makes a static
  one, of two levels or of nested regions level by level, and the kernels, the exchange and the norms take any
  forest TreeAMR's element centering accepts. `adaptive_blast` regrids
  between chunks of a run, by a modal indicator; the cycle is the blast
  wave's, and another problem would write its own from the same parts
  (`refine_flags`, `regrid!`, `WaveProblem(fs, el)`). See [Refined
  forests](#refined-forests) and [Adaptive
  refinement](#adaptive-refinement).
- **Particles on the finest level only.** The forest may be refined, as
  the nested boxes of [Nested boxes](#nested-boxes) are, but
  `InspiralProblem`, `project_source!` and the pairing refuse a source's
  ball or a particle's support that reaches a coarser block, because the
  transfer measures its distances in the finest element width; see
  [What a refined forest does not run](#what-a-refined-forest-does-not-run).
- **Periodic or reflecting boxes.** Every driver builds a fully periodic
  forest by default, and passes no boundary hook to `fill_ghosts!`. A
  box with `reflecting=true` puts a solid (Neumann) wall on every face
  as a TreeAMR reflection symmetry, which the exchange implements by parity-
  signed mirrored transfers rather than a boundary hook; see [Reflecting
  walls](#reflecting-walls). Refinement may touch a wall. An *outflow* or
  partly reflecting box would need the paper's boundary treatment (its
  `Ω₀` and `Ω_{K+1}` singleton elements), which is not implemented.
- **Two initial conditions**, the standing sine mode, on a uniform or
  refined forest, and the 2D radial blast wave, on a uniform or an
  adaptive one. TreeWave's travelling pulse, which exercised a moving
  refined region, is not carried over: the blast wave's ring does that.
- **One `P` per run**, the same in every element. `P` adaptivity is out
  of scope in TreeAMR too (`P` is the field set's, `N` the forest's).
- **No mesh machinery here.** The element centering, its positions and
  quadrature weights, and the coarse-fine operators are TreeAMR's; a
  change to them belongs upstream.
- **No filter.** The paper's tunable modal filter (its eq. 76) is not
  implemented; its tests ran without it, with truncation only.

## The equation

    ∂ₜu = v
    ∂ₜv = ∇²u                    (wave speed c = 1)

Variable 1 is `u`, variable 2 is `v`. This is the paper's eq. (62)
exactly, and TreeWave's system. `u` is the paper's *primary* variable
(no spatial derivative in its right-hand side) and `v` its *conjugate*
one.

## The method

### The reference element

An element of order `P` holds `P + 1` Gauss–Lobatto–Legendre nodes
`ξ₀ = -1 < … < ξ_P = 1` with weights `ŵᵢ`, the paper's choice of
Legendre test functions and collocation points. `Element(T, P)` in
[`src/element.jl`](src/element.jl) builds everything the kernels read
about it as tuples: the nodes, the weights, each node's offset within
its element in units of the mesh spacing, the wide derivative, and the
truncation projector.

The mesh spacing `h` is TreeAMR's, block extent over `N`. An element is
`P + 1` cells wide, `hₑ = (P+1)h`, and every table is written per `h`,
so the kernels divide by the same per-block spacing TreeWave's does.

### The wide derivative

The paper's operator (eq. 57) is `D = d - ½(b - F)`: the element's own
pseudospectral derivative `d`, minus half the boundary operator `b`,
plus half the fetch operator `F`. The central fetch `ξ = 1` is the only
one that is summation by parts. In nodal form it is a `(P+1) × (P+3)`
matrix: row `i` is node `i`, and its columns are nodes `-1 … P+1`, the
element's own nodes with the neighbours' adjacent face nodes either side.
So it is a stencil whose window is the node's element plus one point
either side, with coefficients chosen by the node's place in the
element.

Per `h`, the element's derivative is `2 d̂/(P+1)`, with `d̂` the
derivative on `[-1, 1]` in its closed form for Gauss–Lobatto nodes. The
face terms add `P/2` times the jump `u₀ - u₋₁` to the first row and
`u_{P+1} - u_P` to the last, because `1/(hₑ ŵ₀) = P/(2h)`. With them the
diagonal vanishes, the neighbours reach only the face rows, and `D` is
odd under `x → -x`, as the paper says (§II F). For `P = 2, 3, 4` the
tables reproduce the paper's Figs. 5–7.

### Second derivatives

`∂²u = D(Du)` through an auxiliary first derivative, the paper's §II H:
the only second derivative it found that is summation by parts with the
first. The Laplacian is `Σ_d D_d(D_d u)`; the wave equation needs no
mixed derivatives. The price is a second exchange per right-hand side,
since `D_d u` at a neighbour's face node is needed before the second
application; see [The right-hand side](#the-right-hand-side).

### Summation by parts and the energy

With the quadrature weights `w` (the element weights scaled by
`hₑ/2`), `wD + (wD)ᵀ = 0` on a periodic line. So the discrete energy

    E = ½ Σ w (v² + Σ_d (D_d u)²)

is an exact invariant of the semi-discrete system, with or without
truncation, because the projector is self-adjoint in the same inner
product. What a run loses is the time integrator's. `wave_energy`
computes it. On a refined forest the same holds, with each block's
weights scaled by its own spacing, because the coarse-fine coupling is
built to keep `wD` skew; see [The energy](#the-energy).

## The layout

The node-duplicated layout is TreeAMR's **element-centered** field set,
`centering = elementcentered(D)` with `P` the element's order, and
`G = 1`. A block of `N = E(P+1)` points per dimension holds `E` elements
and owns all `N`, both face nodes of every element, so the state layout
is the usual `N^D` per variable. At `G = 1` the same-level exchange puts
the neighbouring block's first owned node, its low face node, into the
high ghost plane, and its last owned node into the low one: exactly the
outside value the wide derivative reads. Faces between elements inside a
block need no exchange. Across a coarse-fine face the ghost is still the
neighbour's face-adjacent plane along the face's normal, and TreeAMR's
element-local operators (`Operators(family = ElementLocal)`) map it
across the face; see [The coupling](#the-coupling). TreeAMR fills an
element set's faces only, never an edge or corner ghost, which OLDG
does not read.

`E` must be even, so `N` is a multiple of `2(P+1)`: a block of a refined
forest covers half its parent, and has to cover whole elements of it,
which every coarse-fine operator reads. TreeAMR's element centering
requires it of every forest, uniform or not. `check_layout` and
`wave_forest` refuse anything else with the reason, and `WaveProblem`
also refuses a field set whose `P` is not the element's.

### Positions and quadrature

The positions and the quadrature are TreeAMR's: for an element set,
`coordinates` and `fill_by_coordinates!` place each point at its
Gauss–Lobatto node, and `volume_weighted_norm` and the folds weight it
by the element's tensor quadrature. Element `q = (I-1) ÷ (P+1)` and node
`i = (I-1) % (P+1)` of owned point `I` sit at
`origin + h·(q(P+1) + offsets[i])`, with the block's own origin and
spacing, and the offsets of the two face nodes are the integers `0` and
`P+1`, so the two face nodes at one position get bit-identical
coordinates. TreeAMR builds the offsets in `BigFloat` as `Element` does,
so its positions are this package's bit for bit (a test).

The functions in [`src/layout.jl`](src/layout.jl) keep this package's
names, take the `Element`, and refuse a set that is not element-centered
at its order:

- `node_coordinates(fs, el, b, I)`: TreeAMR's `coordinates` at owned
  point `I` (over `1:N`, not a stored index).
- `fill_at_nodes!(f, fs, el)`: TreeAMR's `fill_by_coordinates!`.
- `quadrature_norm(fs, el, u; p, vars)`: the `p`-norm under the
  tensor-product quadrature, each block's weights scaled by its own
  spacing, normalized by the volume. Over all variables it is
  `volume_weighted_norm` to rounding; it keeps a fold of its own, through
  TreeAMR's `mesh_fold`, because it selects variables. Equal weights
  would over-count the nodes clustered near faces.

`wave_energy` sums through the same fold, and `ball_norm` reads each
node's position from it.

## The right-hand side

`wave_rhs!` in [`src/evolution.jl`](src/evolution.jl) does

    u → ufs; fill_ghosts!(ufs)                  # u's face neighbours
    gradient pass; fill_ghosts!(grads[d]) ∀d    # ∂_d u's neighbours along d
    Laplacian pass                              # du, with v from u

The wide derivative along `d` reads one point past its element along
`d` and nowhere else, so the exchange carries exactly that, in three
kinds of field set, all element-centered at the element's order:

- `fs`, the state (`u`, `v`), with no ghosts (`G = 0`). It says what the
  state vector is: its layout, the nodes the initial data are set at
  (`initial_state` fills it and gathers), and the quadrature. Nothing
  exchanges it.
- `ufs`, `u` alone with `G = 1`, filled from the state by a copy kernel
  and then exchanged. `v` is never read across a face, so it is never
  scattered: the Laplacian pass reads it from the state.
- `grads`, one set per dimension: the `d`-th holds `∂_d u` with a ghost
  plane along `d` only (`G = e_d`), since the Laplacian reads it along
  `d` only. Its exchange fills the two faces normal to `d`, and no edges
  or corners.

Each set has its own schedule, since a schedule belongs to a layout. The
schedules' operators are `Operators(family = ElementLocal)`, which take
everything from the set's `P` ([The coupling](#the-coupling)). So a
right-hand side makes `1 + D` exchanges, each of one stage. In 3D at
`N = 16` they fill `1536 + 3 · 512` ghost points per block, faces only,
where one exchange of `u, v` and one of a `D`-variable gradient set with
`G = 1` filled every face, edge and corner, `5 · 1736`, and `v` is no
longer copied. Measured on the
laptop (3D, `P = 3`, 512 blocks of `16³`, one thread) the two
exchanges took 33 ms of a 65 ms right-hand side, and the `1 + D` took
7–12 ms of 26 (the laptop was not idle, and runs differ that much),
measured with cell-centered sets, when `u`'s exchange also filled its
edges and corners, `1736` points.

`WaveProblem` holds the three kinds of set, their schedules, the
per-block spacings on the kernels' backend and the element. As in
TreeWave, `u` is never written and nothing is carried between calls; the
tests check both. The right-hand side does not truncate; see
[Truncation](#truncation).

The window of a node in the first or last element of a block reaches
exactly one ghost plane along the dimension it differentiates, which is
the ghost plane its set has and what the kernels' `@inbounds` asserts; a
run with `--check-bounds=yes` re-checks it (TreeWave's `CODE.md`, "What
bounds checking costs", has why the annotation is worth having).

The gradient and Laplacian kernels take `d` as a `Val`, through an
`@inline` closure over `ntuple(…, Val(D))`, so that each dimension's
index arithmetic is compiled with `d` a constant. A runtime loop over
`d` made them about 2.5 times slower than that on the laptop, and the same
closure without `@inline`, which the compiler did not inline, six times.
With `d` constant both are faster than the one-set kernels were (6.9 ms
against 13.6 for the gradient pass at one thread), although they now
write and read `D` separate arrays.

## Reflecting walls

A solid (Neumann, `∂_n u = 0`) wall is not a boundary *hook*: the
OLDG wide derivative and the face term at the wall come from the solution's
own symmetry, so the wall is declared as a reflection symmetry on the mesh,
and the exchange fills the wall's ghosts by a parity-signed mirrored
transfer. `wave_forest(…; reflecting=true)` marks every face `reflecting`
on a non-periodic forest, and `WaveProblem` reads it off the forest and
gives each field set the parities the wall imposes:

- the state `u`, `v` are scalars, **even** under every mirror;
- the `d`-th component of the gradient `∂_d u` is **odd** under a
  mirror in `d` and even in the others.

The parity of a variable says whether it changes sign when mirrored across a
wall, which is physics (a scalar stays, a normal vector flips) and so
cannot be guessed by the mesh; that is why TreeAMR requires a `parity`
per variable exactly where the forest reflects (storage.jl, `parities_of`).

The scheme is then the same kernels run on the periodic box: the gradient
pass writes `∂_d u` in the interior, the exchange mirrors it odd-wise
into the wall ghosts, and the face term `(P/2)(w₁ - w₀)` at the
wall — the wide derivative's boundary row `d - ½(b - F)` — reads the
mirrored ghost, which carries the symmetries through. The even mirror of
`u` at the wall is what makes `∂_n u` vanish: the boundary gradient
error falls at the discretization's order. The reflected standing mode of
[`reflect_exact`](@ref), `u = cos(ωt) ∏ cos(mπx_d/L)`, has
`∂_n u = 0` on every wall and converges at order `P`; `v`'s ghosts
need no special boundary term because `v` is even and is filled the same
way `u`'s are.

Summation by parts is preserved — the wall is an exact discrete symmetry, so
the discrete energy stays an exact invariant of the semi-discrete system,
conserved to the integrator's own loss exactly as on the periodic box. The
reflected mode is the `sin` mode of a periodic box of side `2L` folded
in half, so `reflect_omega` is the same `mπ√D/L` with `m` the mode
count across the full box.

## Refined forests

`wave_forest(T, Val(D), P, E; refined)` makes the static two-level
forest TreeWave's `sinewave.jl` makes: the blocks whose centres lie in a
region are refined once, and the forest is balanced. `refined = true`
takes the middle half of the box, `(L/4, 3L/4)^D`, which holds a block
centre from `roots = 3` on; a tuple of `D` intervals gives any region,
so that refinement can reach a wall of a reflecting box. A vector of
such regions, one per level above the root level, refines level by
level: region `k` refines the leaves of level `k - 1` whose centres lie
in it, and the forest is balanced after each; `nested_cubes` builds the
nested boxes of the inspiral that way ([Nested boxes](#nested-boxes)).
`wave_errors` and `reflect_errors` pass it through. Nothing else is particular to a
refined forest: the kernels read each block's own spacing, TreeAMR's
element-local operators fill the coarse-fine ghosts, the quadrature
weights each block by its own spacing, and the time step is the finest
level's ([Time step](#time-step)).

The design was settled with TreeAMR's element centering, its milestone
M16; TreeAMR's `CODE.md` ("Centerings", "Operators") has the operators
and the argument for them from the mesh's side.

### The coupling

`D_d = d - ½(b - F)` per element, and `F` reads the neighbour's face node
across each element face, at a block face the ghost. Across a
coarse-fine face normal to `d`, with `M_c` and `M_f` the face
quadratures (the tensor Gauss–Lobatto weights of the tangential
dimensions, each side's own), TreeAMR's `ElementLocal` operators make
the ghost:

- **normal to the face**, for every kind, the neighbour's face-adjacent
  owned plane: injection. The ghost sits at the face, on the other
  side, and so does that plane;
- **tangentially, a fine ghost from coarse**: `I`, the coarse element's
  degree-`P` polynomial at the fine nodes, exact to degree `P`;
- **tangentially, a coarse ghost from fine**: `R = M_c⁻¹ Iᵀ M_f`, over the
  `2(P+1)` fine nodes that cover a coarse element. It conserves `Σ w u`,
  and it is exact to degree `P - 1`, not `P`.

In 1D the face is a point and all three are injection, so a 1D
two-level forest is the uniform scheme on a non-uniform element mesh.
The same `I` and `R` serve `u`'s exchange and each gradient's, under a
mirror too. TreeAMR fills an element set's faces only, in one stage;
the right-hand side reads no edge or corner ghost, which a test checks
by poisoning them with `NaN`. So `D_d` is exact on data of degree `P`
along `d` and `P - 1` across it everywhere, and of degree `P` across it
at the fine nodes but not at the coarse ones (a test).

### The energy

The discrete energy `E = ½ Σ w (v² + Σ_d (D_d u)²)` is an exact
invariant of the semi-discrete system on a refined forest, as on a
uniform one, with or without truncation. The uniform prototype left
this open; the coarse ghost was chosen to make it so:

- The own-element part `W(d - ½b)` is skew by the Gauss–Lobatto
  summation-by-parts property, element by element, whatever the
  element's width.
- What is left are the coupling blocks across faces, and `WD` is skew
  exactly when, across every face, the block by which the coarse side
  reads the fine values is minus the transpose of the block by which
  the fine side reads the coarse ones: `W_c D_cf = -(W_f D_fc)ᵀ`.
- Normal to the face, a face node's weight `hₑŵ₀/2 = h/P` times the face
  term's coefficient `P/(2h)` is `½` on either side, whatever `h`, so the
  condition reduces to the tangential one, `M_c R = Iᵀ M_f`. That is how
  `R` is defined: the coarse ghost is the adjoint of the fine one in the
  quadrature.

Then `⟨g, D_d u⟩_w = -⟨D_d g, u⟩_w` for any `u` and `g` on the whole
forest, the Laplacian `Σ_d D_d D_d` is self-adjoint and negative
semi-definite in the same inner product, and `dE/dt = 0`; the truncation
projector is self-adjoint per element and changes nothing. Measured, the
skew-adjointness residual is about `1e-18` relative to its terms in 1D
to 3D, periodic and reflecting, and over a run the energy is lost to RK4
only, at its rate ([Refined forests](#refined-forests-1) under
"Measured results").

Any other coarse ghost breaks this. The exact `L²` projection in `R`'s
place, which keeps degree `P`, gives a relative rate `(dE/dt)/E` of
`3e-7` to `1e-6` on a rough state in 2D at `P = 3`, and the standing mode
then *gains* energy, `3e-5` of it over a quarter period at `P = 2`
(`roots = 8`); this was measured by an experimental TreeAMR, not in the
package.

### Accuracy

Over the box, the standing mode's L2 error converges at order `P` on a
two-level forest, as on a uniform one, in 1D and 2D for `P = 2, 3, 4`,
and lies between the uniform coarse and fine runs'. That bracket says
little, since the coarse part of the box sets the norm: with the middle
half refined, a 2D error is `0.86–0.92` of the coarse run's, where the
coarse three quarters alone would give `√(3/4) = 0.87`. The claim that
refinement helps is the error *inside* the refined region, which lies
between the uniform runs' there in L2, in 1D and 2D, and in L∞ in 1D.
Tables under [Refined forests](#refined-forests-1).

A uniform forest is not always the upper bound. At a full period the
uniform runs at `P = 3` converge faster than order `P`, by a
cancellation a two-level forest does not keep. In 1D, where the coupling
is plain injection, the two-level error then still converges at order
`P` or better, `3.66` and `3.83`, but is `1.2` to `4` times the uniform
coarse run's, at one and four periods and `roots = 4` to `16`.

**A larger error in the coarse part (1D).** At `P = 2` the L∞ error of
the two-level forest can exceed the uniform coarse run's, `0.310`
against `0.275` at `roots = 4`. It is in the coarse part, in `v`, and it
is not a degree loss, which 1D cannot have. The coarse part's L2 error
is `1.11`, `1.15` and `1.13` times the uniform coarse run's there at
`roots = 4, 8, 16`: a fixed fraction, converging at order `P` with the
rest, and the L∞ ratio over the box is `1.13`, `1.04`, `0.96` and `1.21`
at `roots = 4` to `32`. The standing mode is not an eigenmode of the
two-level operator; the coarse-fine faces scatter the mismatch between
the two levels' discrete solutions back into the coarse part, and that
mismatch is the size of the coarse error. At `P = 3` and `4` the
fraction is between `0.91` and `1.01`. The L∞ norm, being local, follows
the coarse part. Refining the middle half does what it should, where it
should: inside it the L2 error is close to the uniform fine run's.

**The coarse ghost's degree loss in `D ≥ 2` (open).** `R` is exact to
degree `P - 1` tangentially, and in 2D that costs the L∞ error part of
an order, while the L2 error over the box keeps order `P` at a quarter
period:

- With the middle half refined, the L∞ error converges at `2.3–2.7`
  instead of 3 at `P = 3` and at `3.5–3.8` at `P = 4` from `roots = 8`
  on, and at `1.3–1.8` at `P = 2` from `roots = 16` on. Its maximum
  moves into the refined region, at fine nodes, and there it leaves the
  bracket: `9.6e-4` against the uniform coarse run's `5.8e-4` at
  `P = 3`, `roots = 16`.
- A refined stripe, with coarse-fine faces but no corners of the
  refined region, shows the same, so it is the faces.
- With the exact `L²` projection in `R`'s place (the experimental TreeAMR
  above) the same runs converge at order `P` in L∞ too, `2.03`, `3.06`
  and `3.90` in the stripe's fine part at the finest pair; the energy is
  then not conserved.
- Where the regular error is small, it shows in L2 too. At a full
  period, where the uniform run at `P = 3` converges faster than order
  `P` (4.1 and 5.1 at the finer pairs), the two-level forest's L2 error
  converges at `2.92, 2.65, 2.55` and is 45 times the uniform coarse
  run's at `roots = 32`; with the exact projection it converges at
  `3.7` and is 5 times. After four periods, at `P = 3` and `roots = 8`
  with a refined stripe, the L2 error is `2.0e-4` against the uniform
  coarse run's `9.5e-5` (the exact projection's: `9.3e-5`).

So the energy-compatible ghost buys an exact invariant and costs, in
`D ≥ 2`, about half an order of the local error at the coarse-fine
faces, which then travels. Erik decided on 2026-10-31 to keep `R` for
the invariant. The two alternatives, the exact projection (order back,
energy growing) and an order-preserving pair through a finer mortar,
are recorded in TreeAMR's `CODE.md` ("What the element centering leaves
open"), where either would be built. In 1D nothing of it shows, nor for
the reflecting box's `m = 1` mode below.

**Walls.** On a reflecting box with the corner `(0, L/2)^D` refined, so
that coarse-fine faces meet the walls, the Neumann mode converges at
order `P` in L2, and in 2D in L∞ too, the L2 error in the refined
corner lies between the uniform runs', and the energy is lost to RK4
only. That mode, `cos(πx_d)`, vanishes on the coarse-fine faces at
`x_d = 1/2`, which hides the degree loss; with `m = 2` the L∞ error at
`P = 3` converges at `2.38` and `2.34`, as in the periodic box.

**Time step.** RK4's limit on a two-level forest, at the finest spacing,
is the uniform forest's to within 2% in 1D and 2D and 7% above it in 3D
at `P = 1` ([The RK4 limit](#the-rk4-limit)): the coarse-fine faces add
no stiffer mode, and `oldg_timestep`'s finest-level step is safe.

### What a refined forest does not run

- **The particles off the finest level.** The transfer's distances
  (`rnear`, `rfar`, the graded bands) are in units of the finest element
  width, its near and graded elements are classified in it, and
  `project_near!` divides by its weights, so on a coarser element every
  one of them would be off by the ratio of the spacings, and the
  quadrature silently too coarse. `InspiralProblem`, `project_source!`
  and the pairing therefore refuse a source's ball or a particle's
  support that reaches a block coarser than the finest; elsewhere the
  forest may be refined ([Nested boxes](#nested-boxes)).
- **Point interpolation.** TreeAMR's `interpolate` refuses an element
  set, which needs a basis that maps a position to its element's
  polynomial (**(open)** in TreeAMR); nothing here uses it.
- **`test/reference.jl`'s global arrays**, which are uniform-only; the
  refined tests use quadrature inner products instead.

## Adaptive refinement

The forest follows the solution: [`src/refinement.jl`](src/refinement.jl)
holds the criterion, and `adaptive_blast` in
[`src/blast.jl`](src/blast.jl) is TreeWave's `track_blast` cycle for
OLDG. The split is TreeAMR's, as in TreeWave: the application reduces
its own indicator to one flag per block, and `regrid!` balances the
forest, dilates the flagged boxes by the buffer and moves the state set.
TreeAMR refines an element set by the coarse element's polynomial and
coarsens it by the exact `L²` projection, so a regrid keeps the
Gauss–Lobatto integral `Σ w u` of every variable (a test, to `1e-13`).
Measurements are under [Adaptive refinement](#adaptive-refinement-1) in
"Measured results".

### The indicator

`modal_indicator(fs, el; mode)` scores each element by its Legendre
coefficients along every line, in every dimension, relative to the
global peak `A = max|u|`:

    τ = max over lines and m = 1 … mode of ((2m - 1)!! |c_m| / A)^(mode/m) / (2mode - 1)!!

For smooth data `c_m ≈ (hₑ/2)^m f⁽ᵐ⁾/(2m - 1)!!`, so each term is the
estimate mode `m` gives of mode `mode`'s amplitude, all of them `hₑ^mode`;
on a single Fourier mode they agree, and `τ` is `|c_mode|` itself (to
`1e-12`, a test). The single mode's amplitude is `mode_amplitude`.

**Which mode.** Without truncation the last mode is `u`'s top one, `P`.
With truncation `∂ₜu = v` and `v` has no mode of degree `P` in any
dimension, so every coefficient of `u` of degree `P` along a line is
constant in time: it is the initial data's, or what the last regrid put
there, and does not follow the wave. On the blast wave the top mode of a
level-1 block the ring had crossed was `7.03e-6` at `t = 0.1` and at
`t = 0.4`, with the ring's amplitude there `0.14`; refining on it left
the ring on coarse blocks and kept the start's refinement for good. The
truncated scheme therefore reads `mode = P - 1`, the last mode that
moves, and the plain one `mode = P`. Mode 0, the mean, is refused, so
the truncated scheme cannot adapt at `P = 1`. The mode is the caller's
to give, with no default.

**Why every mode up to it.** A single coefficient is not monotone in the
spacing before the mesh resolves the data: on an element wider than the
feature the coefficients stop falling with the degree, and one of them
can cross zero. The blast wave's Gaussian, centred on an element corner,
scores `2.8e-2` in mode 2 at `hₑ = 1.56σ` and `5.7e-2` at `0.78σ`; at
`refine_tol = 3e-2` the single mode passed the root mesh as resolved,
and the run's error reached 7.4. With the lower modes in, it would take
every mode crossing zero at once: over `hₑ/σ` from 1 to 6 the indicator
never falls below its value at `hₑ = σ`, and at `refine_tol = 3e-2` the
initial adaptation never stops on a mesh coarser than `0.42σ` at the
centre at `P = 3`, where the single mode stopped at `1.56σ` to `3.12σ`
in 10 of 180 cases (`σ = 0.04` to `0.12`, `roots = 2` to 8, modes 2
and 3). On resolved data the two agree, and so do the meshes they build.
The weights are what makes that so: without them `c_1^mode` overstates
smooth data by `(2mode - 1)!!` (measured 3.0, 15 and 105 for modes 2, 3,
4) and refines a sine wave three to a hundred times too early. The
alternatives measured against this are in "Measured results" and in
`HISTORY.md`.

The reference amplitude is the global peak at the time of the call, a
`mesh_mapreduce`, as TreeWave's `field_scales` is refreshed at every
regrid: a scale-free indicator scores a negligible tail like a feature
(TreeWave's measurement), and a peak frozen at `t = 0` measured worse
here: the ring's score falls as it spreads, and blocks around it coarsen. `u` alone is read; `v` starts at zero.

### The verdict

`refine_mark` is TreeWave's, with the element as the unit: `(Refine,
box)` where an element is above `refine_tol` below the cap, `(Keep,
box)` where one is above `coarsen_tol`, `Coarsen` otherwise above level
0. The box covers the nodes of the elements above `coarsen_tol`, so that
a block at the cap still recruits its neighbours to its level around the
feature. `refine_flags` computes the indicator once, by block owner, and
hands `flag_blocks`, which calls back concurrently, a callback that only
reads its block's slice. `refinement_buffer` is TreeWave's,
`⌈chunk/h_cap⌉ + 1` points, refused above `N`.

**The thresholds.** Matching the uniform run at the cap's spacing takes
two conditions, both measured:

- `refine_tol` low enough that the initial adaptation reaches the cap:
  `1e-2` for the blast wave at `P = 3` and at `P = 4` with cap 2, `1e-3`
  at `P = 4` with cap 3. The indicator falls as `hₑ^(P-1)`, so the value
  depends on `P` and on how far the cap is below the root level.
- `coarsen_tol` three orders below it, far wider than TreeWave's factor
  of 4. A block whose data are above the run's error but below
  `coarsen_tol` sits coarser than the cap and recruits no neighbours, and
  the ring then reaches data already evolved there, which a later regrid
  refines. **Refining evolved data costs accuracy.** At `P = 4`, cap 3,
  `refine_tol = 3e-3`, the start stopped at level 2, and as the global
  peak decayed the centre went to level 3 at `t = 0.075`: the transfer
  left `v`'s error at the nodes as it was, `8.3e-3`, and one chunk later
  it was `0.124`, with the energy up `7.6e-6`; the same mesh started from
  exact data reached `1.9e-3`. That is consistent with the evolved data's
  jumps across element faces, which a finer element weights by `1/hₑ` in
  its face terms and which the energy-conserving scheme never damps. The
  run ended 290 times the uniform run's error.

The recommended values, `refine_tol` as above and `coarsen_tol =
refine_tol/1000`, keep L∞ within `1.003` and L2 within `1.002` of the
uniform run at the cap for `P = 3, 4` and caps 2 and 3, at 61–79% of its
node-steps to `t = 0.2` and 79–90% to `t = 0.4`, by when the ring fills
the unit box. A factor 10 less between the thresholds already costs up
to 1.5 in L∞ and 1.6 in L2 at cap 3. That is a far better error than TreeWave's 1.41 at 80%
of the points, and a smaller saving: the narrow band is what keeps the
error.

**The wake.** Behind the ring the 2D solution has a real wake (no
Huygens principle): the exact solution at `t = 0.4` scores `2.1e-3` at
level 2 there at `P = 3`, and `1.2e-4` at `P = 4`, both above the
recommended `coarsen_tol`. So the wake stays at the cap, at cap 2 and 3
alike, and the indicator is right to keep it; it coarsens only at
`coarsen_tol ≥ 1e-3`, in runs that are 3–8 times off the uniform one.
The indicator's lower modes change nothing there, since on resolved data
it is the single mode.

### The driver

`adaptive_blast(T; P, E, roots, maxlevel_cap, refine_tol, coarsen_tol,
chunk, …)` has no defaults for the thresholds, the cap or the chunk:

1. TreeAMR's `adapt_to_initial_data!` builds the initial hierarchy,
   flagging and re-sampling the initial data on each new mesh, and the
   initial `v` is truncated.
2. The run goes in chunks of `chunk`, each a fresh `wave_integrator` with
   the step from the current finest spacing: IMEXRungeKutta has no
   `reinit!`, and the state vector's length changes with the mesh.
3. After every chunk but the last: `scatter!` the state into the state
   set, flag, `regrid!` with the buffer, and if the mesh changed, rebuild
   the problem with `WaveProblem(fs, el)` (the other sets and schedules
   belong to the old layout), `gather!`, and `truncate_state!`.

The truncation after the regrid is needed: refining keeps `v` in
`Q_{P-1}`, but coarsening projects two fine polynomials onto one of
degree `P` and brings `v`'s top mode back (a test, above `1e-3`).

**The chunk and the buffer.** The chunk barely matters at the derived
buffer: from `0.0125` to `0.05` at cap 2 and `0.00625` to `0.025` at cap
3 the error stays within `0.993–1.012` of the uniform run's, and shorter
chunks only regrid more often to the same effect, most of their regrids
leaving the mesh alone. The longest chunk the buffer allows is the
cheapest: `0.05` at cap 2 (`buffer = 8 = N` at `P = 3`, 9 of `N = 10` at
`P = 4`) and `0.025` at cap 3. No buffer is fatal (1.2 to 78 times the
error); 2 to 4 points cost up to 1.10.

**Energy at regrids.** Within a chunk the energy is the semi-discrete
system's invariant, and only RK4 loses any (`7.1e-10` per chunk of
`0.05` at `P = 3`, cap 2). A regrid changes the operator and so the
energy; at the recommended thresholds by at most `3e-11` of it, and a
regrid that leaves the mesh alone leaves the state bit for bit. Runs
that refine evolved data jump by `1e-10` to `2e-3`, and the size of the
jump does not order them by their error, so it is no diagnostic.

## Truncation

The paper holds each conjugate variable one degree below the primary
ones (§II I): at every integrator substep the coefficient of the `P`-th
Legendre mode of `v` is set to zero. That is done here literally, in
the state, through the integrator's limiter hooks: the API an integrator
offers for putting a state back into its valid region, which TreeHydro
uses to reset its atmosphere. `truncate_stage!` is the limiter, and
`truncate_state!` the projection it applies. `wave_integrator` installs
it as both of IMEXRungeKutta's hooks:

    init(prob, RK4(); dt, stage_limiter = truncate_stage!,
         step_limiter = truncate_stage!, partition = state_partition(fs, u0))

IMEXRungeKutta calls the stage limiter on every stage value the
right-hand side reads except the first, which is the step's starting
state `uⁿ`, and the step limiter on the step's result, which is the next
step's `uⁿ`. The limiters do not reach the state the integrator starts
from, so `initial_state` truncates the initial `v` itself. So every state
the right-hand side sees and every step's result has `v` truncated, at
one truncation per right-hand side: four of each per RK4 step. The
drivers (`wave_errors`, `uniform_blast`) go through `wave_integrator`, so
a run cannot leave the limiters out.

The right-hand side computes `∂ₜv = ∇²u` without projecting it, and the
scheme does not rely on the stages' linearity to keep `v` truncated: the
mode is set to zero wherever it is evaluated.
`evolution_tests.jl` checks every stage and every step, and checks that
RK4 without the limiters leaves the mode in, so it is the limiter that
removes it.

The one-dimensional projector is `Π = I - |L_P⟩⟨L_P|/γ` in the
quadrature inner product, where Legendre modes of different degree are
discretely orthogonal. In several dimensions the projector is the tensor
product `Π ⊗ … ⊗ Π`, which removes every mode whose degree is `P` in
*any* dimension, so `v` lives in `Q_{P-1}`. The paper's text describes
one dimension only.

The kernel applies it in place, one work item per element, one
dimension at a time, and as a rank-one update:
`Π = I - topmode ⊗ topdual`, with `topmode = L_P(ξᵢ)` and
`topdual = ŵⱼ L_P(ξⱼ)/γ` (both in `Element`, from `BigFloat`). Along
each line of the element that is a dot product and an update, about
`2D` multiply-adds per node, where the tensor product as a matrix costs
`(P+1)^D` (64 at `P = 3` in 3D) and needed a copy of `v` to read from.
The dimensions commute and run in a fixed order, so the result does not
depend on the thread count, and it equals the matrix form to roundoff
(a test). On the laptop (3D, `P = 3`, 512 blocks of `16³`, one thread)
a truncation took 41 ms before, more than half a right-hand side, and
takes 3 ms. The copy it replaced was a serial broadcast, and the first
write to its scratch array, which so lay on one NUMA domain.

`truncate=false` gives the plain scheme with both variables at degree
`P`. What each costs is under [Convergence](#convergence). In short,
truncation gives a clean order `P` for `u` and `v` alike. Without it,
`u` is more accurate and often converges faster, but `v` converges
irregularly and more slowly. The default is the paper's scheme.

## Time step

`oldg_timestep(p; cfl)` is `cfl · hₑ/(P+1)²`, the paper's DG step
(eq. 64), which is `cfl · h/(P+1)` in the mesh's spacing. The integrator
takes `⌈(t1 - t0)/dt⌉` equal steps, so the step it takes is at most
this. Measured, RK4
is stable up to `cfl ≈ 5` in one dimension for every `P` from 1 to 6,
and the limit falls as `1/√D` (the Laplacian's spectral radius adds over
dimensions); see [The RK4 limit](#the-rk4-limit). The drivers default to
`cfl = 1/2`. On a refined forest `hₑ` is the finest level's, and the
limit there is the uniform forest's at that spacing, to a few per cent.

RK4's error is fourth order in `dt ∝ h`, so beyond `P = 4` it eventually
dominates the spatial error. A convergence study at higher order needs a
smaller `cfl` or a higher-order `alg`. At `P = 5` and `cfl = 1/2` the
rates over the measured resolutions are still 5.

## The blast wave

[`src/blast.jl`](src/blast.jl) is TreeWave's radial blast wave in 2D: a
Gaussian peak `exp(-(r/σ)²)` at rest in the middle of the periodic unit
box, `σ = 0.08`, evolved to `t = 0.4`. It splits, the ingoing half passes
through the origin, and a single ring expands, losing amplitude to
geometric spreading with a wake behind it. On a uniform forest it is the
case whose feature is aligned with nothing, so it crosses element and
block faces at every angle; that is what it adds to the sine mode.

The exact solution is TreeWave's: the inverse Hankel transform of the
Gaussian, `J₀(kr)` tabulated once on an `(r, k)` grid in `Float64`
(`blast_reference`), contracted against `cos(kt)` and `sin(kt)` for each
time (`blast_radial_table`), and summed over the periodic images of the
source (`blast_exact`), since a ring that reaches the corners of the box
is as close to an image as to the source. Its quadrature error, about
`1e-6`, is below every error measured against it; `blast_tests.jl` checks
it against the initial data at `t = 0`.

`uniform_blast` runs it with the same truncating RK4 as `wave_errors`,
and takes an `observer(p, t, u, ue)` called at `nsnapshots` times with
the state and the exact state at the same nodes. `adaptive_blast` runs
the same case on a forest that follows the ring ([Adaptive
refinement](#adaptive-refinement)), with the same observer. What
TreeWave's version has and this one does not: the indicator's own
scales and its panel, the coverage measure, and the backend, centering
and type options of its viewer.

## The viewer

`bin/visualize2d.jl` draws the blast wave, in an environment of its own
(`bin/Project.toml`, with CairoMakie and SixelTerm) so that CairoMakie is
not a dependency of the package, as in TreeWave:

    julia --project=bin bin/visualize2d.jl [--P=4] [--E=2] [--roots=16] [--plain]
    julia --project=bin bin/visualize2d.jl --adaptive [--roots=4] [--cap=2]

It writes `bin/output/blast_2d.png` (`blast_2d_plain.png` with
`--plain`, `blast_2d_adaptive.png` with `--adaptive`) and shows it
inline on a terminal. `--adaptive` runs `adaptive_blast` from 4 × 4 root
blocks up to level 2 at the recommended thresholds (`1e-2`, `1e-5`,
chunk `0.05`), the same finest spacing as the uniform default, and
colours the block faces by level, TreeWave's colours; at `P = 4` its
error of `u` at `t = 0.4` is the uniform run's, `1.79e-4` in L∞, at 88%
of its node-steps. One figure, three parts:

- a filmstrip of `u` at four times. Each block is one heatmap on cells
  around its nodes: within an element a node's cell runs to the
  midpoints with its neighbours, and a face node's cell ends at the face,
  so the two nodes of a face each get the half-cell on their own side and
  the cells tile the block. Element faces are drawn faintly and block
  faces on top;
- `u` against the minimum-image radius at every node of the final
  frame, face nodes and interior nodes apart, against the exact
  free-space profile out to `L/2`. A jump across a face shows as a face
  node off the curve; past `L/2` the ring meets its own periodic images
  and the exact solution stops being radial too;
- the L∞ and L2 errors of `u` against the exact ring, and the energy
  lost, over time; with `--adaptive` its modulus, since a regrid may
  raise it.

The run comes from `uniform_blast`'s or `adaptive_blast`'s observer, so
the script has no time-stepping loop of its own. CI renders both schemes and keeps the
figures.

## Precision

The nodes are irrational from `P = 3` on, so TreeWave's rule of exact
rational literals cannot hold for the tables. `Element(T, P)` builds
nodes, weights, derivative and projector in 512-bit `BigFloat` and
rounds each to `T` once. Each type's table is the correctly rounded
`BigFloat` one, not the `Float64` one converted. The nodes are
symmetrized before rounding, so `ξᵢ == -ξ_{P-i}` holds bit for bit.

`Float32` runs, with the same step count as `Float64` and an error that
agrees to `1e-3` at a coarse mesh. A MultiFloats type is not tested: the
sine mode needs `sin` and `cos`, as in TreeWave.

## Threads and devices

Every kernel writes one output point per work item, or one element per
work item in place for truncation, and reads only what no other work
item writes, so the answer does not depend on the thread count. The
folds are TreeAMR's, which are bit-identical across thread counts on the
CPU, and so is the integrator's stage arithmetic (below). The suite
passes with `--threads=4`; there is no subprocess test comparing thread
counts bit for bit, as TreeWave has **(open)**.

Every pass over the mesh runs each block on the thread that owns it,
TreeAMR's `threadchunks` partition: the kernels through `map_blocks!`,
the truncation through `TreeAMR.launch_by_owner!` (internal, the launch
`map_blocks!` uses, with an element-shaped `ndrange` whose last axis is
still the block), the exchanges by TreeAMR's own ownership rule, and the
stage arithmetic through `state_partition`. Each array's pages are first
written by the same partition, so with the threads pinned
(`JULIA_EXCLUSIVE=1`) they lie on the NUMA domain of the core that uses
them. No pass is serial any more: the scratch copy of `v` that truncation
used to fill by a broadcast is gone.

The element is a tuple of tuples and so travels as a kernel argument,
and the gradient sets' arrays as a tuple. The spacings and origins
follow the field sets' backend (`to_backend`). Nothing has been run on a
device yet **(open)**. The truncation kernel reads its element through
runtime tuple indices, which a GPU may spill to local memory.

### The integrator

The time integrator is IMEXRungeKutta's explicit `RK4()`, through
`wave_integrator` in [`src/stepping.jl`](src/stepping.jl), with the
limiters of [Truncation](#truncation) and the partition
`state_partition(fs, u0)`. That is TreeHydro's helper: thread `c` owns
the state entries of the blocks in chunk `c` of `threadchunks(nblocks)`,
which is one range, since a block's `u` and `v` are contiguous. On a
device array it is `nothing`, IMEXRungeKutta's broadcast path. By owner
and broadcast give the same bits (a test), whatever the thread count.

It replaced OrdinaryDiffEq's `RK4`, for three reasons:

- OrdinaryDiffEq forms its stage vectors with serial broadcasts, which
  were 79% of a 64-thread step in TreeWave ("What it buys, and what caps
  it" there). IMEXRungeKutta forms each combination in one pass, each
  block's entries on its owner's thread.
- OrdinaryDiffEq allocates and first-writes its cache serially, so its
  pages lie on one NUMA domain; IMEXRungeKutta first-touches its scratch
  through the partition.
- With a step limiter, OrdinaryDiffEq's RK4 evaluates the right-hand
  side once more after the limiter has changed the result, to refill
  its first-same-as-last slot: five right-hand sides and five
  truncations per step. IMEXRungeKutta's makes four of each.

On the laptop (3D, `P = 3`, 512 blocks of `16³`) the switch alone took
a step from 497 to 385 ms at one thread and from 155 to 120 ms at six,
nearly all of it the fifth evaluation; the stage arithmetic is 5–6 ms
there, since one M3 core already streams most of its memory bandwidth.
On Symmetry's 64 cores the stage arithmetic was 234 ms of a 714 ms step
and is 17 of 72; see [Threads on Symmetry](#threads-on-symmetry).

The drivers step one integrator: `wave_errors` with `solve!`, and
`uniform_blast` with `step!`, calling its observer after the steps
nearest the snapshot times, so that watching a run does not change its
steps.

## Particles

`INSPIRAL.md` designs two inspiralling scalar particles of fixed size
`ℓ ≪ h` on this scheme, and `INSPIRAL-PLAN.md` builds it in milestones.
What exists is M1, the analytic particle, everything about a particle
that needs no mesh; M2, the transfer between particles and grid; M3,
the coupled right-hand side for particles on a prescribed path, which
subtracts the projected source after `wave_rhs!` and leaves `wave_rhs!`
unchanged; M3b's nested boxes, a static refined forest whose finest
level holds the source; and M4's prescribed paths with known
acceleration, the slow circle and hyperbolic motion, with the retarded
field of a point charge to start from and compare with. The particles do
not yet move under their forces (M6a).
Sections of `INSPIRAL.md` are named below without links.

The closed forms are derived in the Wolfram Language and generated into
Julia:

- `derivations/profile.wl`, `nearfield.wl`, `selfforce.wl` and
  `labslice.wl` derive the profile's potentials, the near field and its
  self-force, and the `t′²` term and the lab slice of orders 1–3, and assert
  every identity they rely on, so that a wrong derivation stops the
  script (`check` in `derivations/common.wl` exits with status 1).
- `derivations/generate.wl` writes `src/particle_forms.jl`, which is not
  edited by hand. Each radial function is a scale times a Horner
  polynomial, `ℓ^p` times a polynomial in `σ² = q/ℓ²` inside the
  particle and `r^p` times one in `ℓ²/q` outside, with `q = |y|²` and a
  branch on `σ < 1`. Inside, the generator takes `1 - σ²` as the
  variable instead where that is better conditioned on `[0, 1]`, which
  is the case for `ρ₀ = c(1 - σ²)⁴` alone (condition 1 against `10¹²`
  near the edge). Coefficients are exact rationals, `_q(T, n, d)`,
  rounded once to the model's type, and `π` is `T(π)`, the rule of
  [Precision](#precision). Everything else is generic in its argument
  types, so dual numbers pass through.
- `src/particle.jl` is written by hand on top: the particle's state, the
  boost, and the time derivatives, which are ForwardDiff dual numbers.

The suite regenerates the file and compares it byte for byte when
`TREEWAVEOLDG_WOLFRAMSCRIPT` names a `wolframscript` (CLAUDE.md,
"Commands"), and skips that test otherwise, as on CI.

### The particle

`ParticleModel(; Q, ℓ, M)` is `INSPIRAL.md`'s particle: the profile

    ρ₀(r) = (3465 Q / 512π ℓ³) (1 - σ²)⁴,   σ = r/ℓ < 1

(`C³` at `σ = 1`), its static field `Φ₀` (`∇²Φ₀ = ρ₀`, `-Q/4πr`
outside), and the physical mass `M`. The bare mass `M₀ = M - δm` must
be positive, `δm = 693Q²/4199πℓ ≈ 0.0525 Q²/ℓ`, and the constructor
refuses a particle below that classical radius with the reason.

The radial potentials `B_k = ∫ ρ₀(x′) |x - x′|^k d³x′` for
`k = -1, 1, 3, 5` (`B₋₁ = -4πΦ₀`) are derived from the shell average of
`|x - x′|^k` and asserted to satisfy `∇²B_k = k(k+1)B_{k-2}`, to match
`INSPIRAL.md`'s exterior forms and `B₁′`, and to be even polynomials
inside. Inside the particle, with `σ = r/ℓ`,

    B₃/Q = ℓ³ (33/256 + 231σ²/256 + 693σ⁴/1280 - 33σ⁶/256 + 11σ⁸/256
               - 3σ¹⁰/256 + 7σ¹²/3328 - 3σ¹⁴/16640)
    B₅/Q = ℓ⁵ (99/2048 + 165σ²/256 + 693σ⁴/512 + 99σ⁶/256 - 55σ⁸/1024
               + 3σ¹⁰/256 - 15σ¹²/6656 + σ¹⁴/3328 - 9σ¹⁶/452608)

At `σ = 1` the interior and exterior forms of `B_k` agree to `k + 6`
derivatives (`B₋₁` is `C⁵`, `B₅` is `C¹¹`), two more for each step in
`k`, since `∇²B_k ∝ B_{k-2}`. The moments are `INSPIRAL.md`'s,
`⟨s²⟩ = 3ℓ²/13`, `⟨s⁴⟩ = ℓ⁴/13`, `⟨s⁶⟩ = 7ℓ⁶/221`, and so are
`δm = (1/12π)∬ρρ′/|x - x′|` and `∬ρρ′|x - x′| = 60984Q²ℓ/96577`.

A moving particle carries its rest-frame profile, a Lorentz scalar:
`particle_density(m, x, z, v)` is `ρ₀(|y|)` with `y = Λ(v)(x - z)`, and
`boosted_static` is `Φ_C = Φ₀(|y|)`, which solves
`∇²Φ_C - (v·∇)²Φ_C = ρ` exactly. The boost is written
`Λ(v)r = r + γ²/(γ+1) (v·r) v`, which needs no `v̂` at rest.

`Trajectory(; z, v, a, ȧ, ä, a⃛)` is a particle's state with the first
derivatives of its acceleration, as `SVector{3}`s and `isbits`. It is
the Taylor polynomial in `t` that time derivatives act on:
`trajectory_at(traj, t)` evaluates it, and its derivatives, at a time
that may be a dual number. Holding `a⃛` lets `𝒟Ψ` see the time
derivative of the order-3 term's `ä`; nothing holds `d⁴a/dt⁴`, so the
bracket omits `d⁴a·∂_äΦ⁽³⁾`, which is of order 5.

### The near field

`nearfield.wl` derives orders 1–3 from the retardation series of the
extended particle, by the three steps of `INSPIRAL.md`'s "How the terms
are computed", without using its results: `ρ(t, x′) = ρ₀(s + δ(t))`
expanded in `δ` and in `t` with the orders counted by a bookkeeping
parameter, every integral reduced by the identities
`∫R^k ∂^αρ₀ = ∂^αB_k` and `∫R^k sᵢF = rᵢ∫R^kF - ∂ᵢ∫R^{k+2}F/(k+2)`, and
the `B_k` kept as abstract functions of `q = |r|²` until the end. The
result equals `INSPIRAL.md`'s orders 1–3 inside and outside the
particle, which is the independent check its **(open)** item asked for.
Also asserted: each order's Laplacian is `∂ₜ²` of the lower terms (the
hierarchy `∇²Φₙ = ∂ₜ²Φₙ₋₂`); orders 1–3 hold no `ȧ`; their point limits
are `INSPIRAL.md`'s; and they are the terms `k = 1, 2, 3` of the
hyperbolic-motion field.

Each order is generated as a sum over rotation invariants times radial
functions, `(a·y)` at order 1, `a²` and `(a·y)²` at order 2, `(ä·y)`,
`a²(a·y)` and `(a·y)³` at order 3, together with its Laplacian in the
same form, taken from the hierarchy (`near_field_term`,
`near_field_laplacian`).

**The self-force** of the particle's own terms (`selfforce.wl`), the
force `-∫ρ₀∇Φ⁽ᴺ⁾` and the potential average `(1/Q)∫ρ₀Φ⁽ᴺ⁾` on a
particle at rest:

| order | force | potential average |
|---|---|---|
| 1 | `-δm a`, `δm = 693Q²/4199πℓ` | 0 |
| 2 | 0 | `(2541/193154π) Qℓ a² ≈ 0.00419 Qℓ a²` |
| 3 | `-(2541/96577π) Q²ℓ ä - (15246/482885π) Q²ℓ a² a` | 0 |

The `ä` coefficient, `≈ -0.00837 Q²ℓ`, is `INSPIRAL.md`'s
`-(1/24π)∬ρρ′|x - x′|`. The `a² a` coefficient, `≈ -0.01005 Q²ℓ`, was
**(open)** there; it is exactly 6/5 of the `ä` one. On a circular orbit,
where `ä = -Ω²a` and `a² = v²Ω²`, the two together are
`(2541/96577π) Q²ℓ Ω² (1 - 6v²/5) a`. Order 2's potential average,
which `INSPIRAL.md` does not mention, enters the mass
`m = M₀ + QΦ̄ₑ` of "Equations of motion"; it is `ℓa²` small. The
generated `self_inertia`, `self_force_addot`, `self_force_a2a` and
`self_potential_a2` hold these coefficients for M6.

**Moving particles.** `near_field(m, Val(N), x, traj)` is the rest-frame
term at the rest-frame position `y = Λ(v)(x - z)`, with the rest frame's
derivatives of the worldline of `rest_frame_derivatives`: the proper
acceleration `α′ = γ²Λ(v)a` for `a` and `C` for `ä` (**The lab slice**
below). At rest it is the time-symmetric field order by order. Moving,
the particle field adds the lab slice of each order it carries,
`lab_slice(m, Val(N), x, traj)`.

**The `t′²` term of `Φ_C`,** the one part of the re-expansion carried
ahead of M5, decided by the owner on 2026-11-02 after M4's first series
([Prescribed paths](#prescribed-paths)). A lab point at `r = x - z` sits
at rest-frame time `t′ = -γ v·r = -v·y`, where the particle, at rest at
`t′ = 0`, has moved by `α′t′²/2`, so its static field there is
`Φ₀(|y - α′t′²/2|)`, and the lab slice picks up

    T₂ = -½ t′² (α′·∇Φ₀(y)) = -(v·y)² (α′·y) Φ₀′(q),    Φ₀′ = dΦ₀/dq,

which outside the particle is `-(Q/8π) v²(v̂·n̂)²(α′·n̂)`, of order `r⁰`
and proportional to `v²a`: the part `INSPIRAL.md` found the boosted form
missing, and whose force does not converge. `derivations/labslice.wl`
derives it with the extended `Φ₀`, so it is smooth inside the particle,
asserts that it is that `t′²` coefficient and that its Laplacian in `y`
is

    ∇²T₂ = -[2v²(α′·y)Φ₀′ + 4(v·y)(v·α′)Φ₀′ + (v·y)²(α′·y)(8Φ₀″ + ρ₀′)],

with `∇²Φ₀ = ρ₀` differentiated in `q` so that no term of size `Q/ℓ³`
appears. From M4 to M5 the particle field carried `T₂` alone, from order
1 on, gated with the order-1 term: it holds one acceleration, so it
counts as order 1, and `INSPIRAL.md`'s table asks each order carried to
come with its lab-slice corrections; order 0 stays the particle paper's
`Φ_C` alone. Since M5, `T₂` is part of the lab slice of order 1, `Λ₁`,
which the particle field carries with order 1 in the same way, and it is
no longer generated on its own; `labslice.wl` still derives it, as a
check of `Λ₁`. Its own force on the particle at rest is
`(11/60)δm v²α′ + (11/30)δm (v·α′)v`; `Λ₁`'s is zero.

It also removes something at rest, which `INSPIRAL.md` does not say.
The bracket of [The split](#the-split) takes `∂ₜ²`, which sees the field
at neighbouring times, where `v ≠ 0`. `T₂` vanishes at rest, but its
second time derivative does not: without it, along a trajectory through
rest the bracket with orders 1–3 carried keeps `(a·r)²(a·∇Φ₀)` at order
3, which outside the particle is `(Q/4π)(a·n̂)³` plus terms in `(ℓ/r)²`
and behaves like `r⁰`, so that carrying order 3 removed only the `ä`
part of the order-3 residual. With `T₂` carried, a point particle's
bracket has no order-3 part along a trajectory through rest, and an
extended one's keeps only terms in `(ℓ/r)²` (`particle_tests.jl`). Those
are the second time derivative of the field of the slicing density of
**The lab slice** below, which `T₂` alone lacks.

**The lab slice** (M5). `labslice.wl`
derives the whole re-expansion of orders 1–3, `Λ_N`, what the field of
order `N` on the lab's slice adds to the rest-frame term `near_field`.
It is not `T₂` and the `t′` terms of the rest-frame near field alone.
The model's density `ρ₀(|Λ(v)(x - z)|)` is rigid on the lab's slices
("Fields and particles"), not on the rest frame's, and the two differ
inside the particle already at order 1, by `½(v·y)²(α′·∇ρ₀)`. With `T₂`
alone the bracket keeps that difference, `Q(v·y)²(α′·y)ρ₀′`, linear in
the acceleration and of size `Q v²|a|/ℓ²` inside the particle; on `main`
at `v = (0.2, 0.1, 0)` the bracket divided by `|a|` tends to `8.4·10⁻⁴`
as `a → 0` at a point inside, which is that term, where at rest it
tends to zero. So `labslice.wl` takes the lab-sliced density's history
in the instantaneous rest frame: the density at a rest-frame event
`P = (t′, s)` is `ρ₀` of `P`'s distance from the worldline in the rest
frame of the worldline's event `E` that the lab sees simultaneous with
`P`, `τ_E = t′ + v·s - v·Z′(τ_E)`. Its time-symmetric field is the
retardation series of `nearfield.wl` at any `t′`, evaluated at
`t′ = -v·y`, and `Λ_N` is that minus `phiN[N]`. Everything is
polynomial in `v`, which enters only through the slicing.
`INSPIRAL.md`'s formula, which re-expands the rest-frame near field at
`t′` with the particle at its position and velocity then, is the same
with a rest-frame-sliced density.

The worldline in the rest frame is `Z′(τ) = ½α′τ² + ⅙Bτ³ + (1/24)Cτ⁴`
in its coordinate time; these are `nearfield.wl`'s `a`, `ȧ` and `ä`.
`rest_frame_derivatives(traj)` gives them from the lab's `v`, `a`, `ȧ`,
`ä` by inverting `t′(t)` along the boosted worldline as a series,
generically, so that it works on dual numbers. `B` vanishes at rest but
not in general: constant lab acceleration is not constant proper
acceleration.

`Λ_N` and `∇²Λ_N` are generated as sums over monomials in ten
invariants, `v·y`, `α′·y`, `B·y`, `C·y`, `v²`, `v·α′`, `|α′|²`, `v·B`,
`α′·B` and `v·C`, times radial functions (`lab_slice_term`,
`lab_slice_laplacian`): 3, 21 and 74 terms at orders 1, 2, 3, with up to
six powers of `v`, and as many in the Laplacians but one of order 2's,
which vanishes once the `B_k` are their closed forms. They use the
`B_k` up to `k = 11`. Some radial factors vanish as `ℓ → 0`, finite-size
terms. The Laplacians come from the invariant form,
`∇²(fM) = (4qf″ + 6f′ + 4kf′)M + f Σᵢⱼ ∂ᵢ∂ⱼM (uᵢ·uⱼ)` with `k` the
degree of `M` in `y`, and are asserted against `∇²` of the terms.
`lab_slice(m, Val(N), x, traj)` evaluates `Λ_N`. In `Float64` the
generated terms and Laplacians agree with `BigFloat` to `10⁻¹⁴`,
inside, outside and at `σ = 0.999` and `1.001`.

The source evaluates them some `10⁷` times per right-hand side for the
same particle, so the generator emits each order and kind (the term, its
Laplacian) in two steps. The terms are gathered by their monomial in the
four invariants linear in `y` and their power of `ℓ` or `r`; the six
others do not depend on the point, so each group's radial polynomial with
them folded in is one polynomial, whose coefficients
`_lab_coefficients` computes once per particle and step, and
`_lab_evaluate` sums one polynomial per group: 3, 10 and 23 groups at
orders 1–3 on each side of the particle's edge, where there are 3, 21
and 74 terms. `_lab_slice_term` and `_lab_slice_laplacian` chain the
two. The branch on `σ < 1` is taken once per evaluation, and `r`, `ℓ²/q`
or `σ²`, `1 - σ²` are formed once; inside, a group's polynomial is in
`σ²` or `1 - σ²`, whichever is better conditioned for the worst of its
terms. Three things the generated code must keep, each found the hard
way:

- **Running sums.** Julia parses `a + b + …` as one call of `+` with as
  many arguments, which allocates beyond about 30 of them: the order-3
  sum written as one expression allocated even in `Float64`.
- **No inlining.** A `BigFloat` Hessian of the order-3 lab slice with
  every radial function inlined took 178 s to compile, against 4 s in
  `Float64`, and 17 s with them out of line. `_lab_coefficients` and
  `_lab_evaluate` are not inlined; the polynomials inside them are.
- **No copies of the tables.** The bracket reads a coefficient in place
  through `_coefficient(c, i, d)`, which lifts it from its rates there;
  wrapping the tables for every point copied them, about 10 KB per
  evaluation, and order 0 cost ten times what it did.

One evaluation of the source's bracket per point, one particle, `Float64`
on the laptop (`source_terms`, inside and outside the particle):

| order | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| before M5 (`T₂` alone), ns | 8.5 | 48 | 69 | 102 |
| each term on its own, inlined, ns | 17 | 129 | 1044 | 4413 |
| one function per order, ns | 17 | 72 | 380 | 1867 |
| grouped, as now, ns | 15 | 71 | 315 | 1017 |

So carrying the lab slice costs ten times what the source cost at
order 3, and the projection is mostly source evaluations ([The
transfer's cost](#the-transfers-cost)) **(open)**: on Symmetry the
projection costs 8.2 times what it did at order 3 and a step 4.9 times
([Prescribed paths](#prescribed-paths-1) under "Measured results"). The
grouped form
costs more to compile in `BigFloat`: the particle tests take 5m46 on the
laptop, against 1m53 one function per order and 43 s before M5.

`labslice.wl` asserts: the density's displacement against the lab's own
formula at `γ = 5/4`; that the construction at `t′ = 0` and `v = 0` is
`nearfield.wl`'s; `Λ_N = 0` at `v = 0`; `Λ₁ = T₂` plus the field of the
slicing density; the decompositions and Laplacians, inside and outside,
at random exact points; and orders 0–3 of the point limit against the
series of the exact field of uniform acceleration, the Born field
`-Q/4πD` with `D² = g²S²/4 - (X² - t²)`, on the tilted slice
`t′ = -v·y`, at three points and velocities, exactly. That is the
independent check `INSPIRAL-PLAN.md` asks for at `t ≠ 0`. `Λ₁` exerts
no force on the particle at rest: the slicing field's cancels `T₂`'s
`(11/60)δm v²α′ + (11/30)δm (v·α′)v`. So M6a's self-force needs no
order-1 lab-slice term.

The particle field carries `Λ_N` with order `N`, and the near field
takes `C` for `ä`; the bracket follows ([The split](#the-split)). With
them, along any motion, the bracket with orders up to `N` carried is
`O(ε^{N+1})` when the derivatives of the velocity scale as `εᴺ`, inside
the particle and out (`particle_tests.jl`); before, `T₂` alone left
`Q(v·y)²(α′·y)ρ₀′` at `O(ε)` inside it.

### The split

The particle field and its time derivative, `INSPIRAL.md`'s "The
particle field":

- `particle_field(m, x, traj; order)` is
  `Ψ = Φ_C + Σ_{N ≤ order} (Φ⁽ᴺ⁾ + Λ_N)`, the near field and the lab
  slice of each order, `order` from 0 (the particle paper's `Φ_C`
  alone) to 3, so that M4 can measure what each order buys.
- `particle_rate(m, x, traj; order)` is `𝒟Ψ`, a dual-number derivative
  in `t` of `Ψ` along the trajectory's Taylor polynomial.

`source_bracket(m, x, traj; order)` is one particle's `𝒟²Ψ - ∇²Ψ + ρ`.
Near the particle `∇²Ψ`, `(v·∇)²Ψ` and `ρ` are each of size `Q/ℓ³`, and
the bracket is assembled without subtracting them:

- `ρ` cancels against `□_vΦ_C = ∇²Φ_C - (v·∇)²Φ_C` analytically, and
  each near-field term has `∇²_x = ∇²_y + γ²(v·∇_y)²`, where
  `γ²(v·∇_y)² = (v·∇_x)² = (v·∂_z)²`. So the bracket is
  `𝒟²Ψ - (v·∂_z)²Ψ`, the part of `Ψ`'s second time derivative the
  acceleration causes, minus `∇²_y` of the near-field terms and of the
  lab slice, which is their generated closed form. The lab slice obeys
  the same `∇²_x = ∇²_y + (v·∂_z)²`, as every function of `y` at fixed
  `v` and rest-frame derivatives does.
- `Ψ` depends on `x` and `t` only through eleven invariants: `q = |y|²`,
  the four `u·y` for the rest-frame vectors `u = v, α′, B, C`, and their
  six dot products that the lab slice reads, `v²`, `v·α′`, `|α′|²`,
  `v·B`, `α′·B` and `v·C`. With `r = x - z` and `M = Λ(v)² = I + γ²vvᵀ`
  they are `rᵀMr`, `(Λ(v)u)·r` with `Λ(v)v = γv`, and numbers that do not
  depend on `x`, so their dependence on time is through `r`, with
  `ṙ = -v` and `r̈ = -a`, and through `M`, the `Λ(v)u` and the dot
  products. The lab slice reads the dot products only through its
  groups' coefficients, which `ParticleJets` holds with their rates. Of each invariant `I` the bracket needs the rate `İᵤ`
  of uniform motion (`r` moving with `-v`, everything else held) and the
  acceleration's parts `δİ = İ - İᵤ` and `δÏ = Ï - Ïᵤ`, written out term
  by term: for `q`, `δq̇ = rᵀṀr` and `δq̈ = -2aᵀMr - 4vᵀṀr + rᵀM̈r`.
  Then

      𝒟²Ψ - (v·∂_z)²Ψ = Σᵢ Ψ_i δÏᵢ + Σᵢⱼ Ψ_ij δİᵢ (2İᵤ + δİ)ⱼ,

  the coefficient of `ε₁ε₂` in `Ψ(I + ε₁δİ + ε₂(2İᵤ + δİ) + ε₁ε₂δÏ)`, one
  evaluation of `Ψ` on nested dual numbers.

The cancellations left are between terms of size `Qa/ℓ²`, and cost
little (see [The source bracket](#the-source-bracket)). The generator
emits each near-field term and its Laplacian as a function of the
invariants (`_near_field_term`, `_near_field_laplacian`) as well as of
`y`.

`ParticleSource(models, w, trajs; order)` is the effective source for
several particles, a callable `S(x)`,

    S = W Σₐ [𝒟²Ψₐ - ∇²Ψₐ + ρₐ] - 2∇W·∇Ψ̂ - Ψ̂ ∇²W,

zero beyond the window, and `particle_source(models, w, x, trajs; order)`
is the same at one point. It holds each particle's `ParticleJets`, the
`x`-independent `M`, the `Λ(v)u`, the dot products and the lab slice's
coefficients (the term's with their first two time derivatives, the
Laplacian's alone), by dual numbers along the trajectory, once per
source and not per point, and is `isbits`, so that it can be a kernel
argument; with the coefficients it is about 18 KB per particle. In the
window's transition the same evaluation of `Ψ` carries a second outer
partial seeded with `∇W·∇I`, which gives `∇W·∇Ψ`, and its value gives
`Ψ`. Before M5, on the laptop, a point inside the window cost 0.14 µs
for both particles and one in the transition 0.22 µs, of which 11 ns is
`window_terms`, and building the jets 0.8 µs per particle ([The
transfer's cost](#the-transfers-cost)); the lab slice multiplies the
first two by about ten (**The lab slice** in [The near
field](#the-near-field)).

### The window

`Window(centre, R₁, R₂; continuity = m)` is one inside `R₁`, zero
beyond `R₂`, and `1 - S_m((r - R₁)/(R₂ - R₁))` between, with `S_m` the
polynomial step of degree `2m + 1` whose first `m` derivatives vanish at
both ends,

    S_m(ξ) = ξ^{m+1} Σₖ₌₀^m C(m+k, k)(1 - ξ)^k,   S_m′ = (2m+1)C(2m, m) ξ^m (1 - ξ)^m.

So `W` is `C^m` and no smoother. `INSPIRAL.md` asks for `C^{P+1}` or
better, and `m = P + 1` is the choice: degree `2P + 3`, `9` at `P = 3`.
A larger `m` steepens the step's middle, whose slope grows like `√m`,
and costs resolution for smoothness the scheme of order `P` cannot use.
`window_terms(w, x)` gives `W`, `∇W` and `∇²W` in closed form; `W = 1`
holds exactly where the particles are, so the split is exact there.

### The transfer

[`src/transfer.jl`](src/transfer.jl), generic in `D` and tested in 1D,
2D and 3D; the particle physics on top of it is three-dimensional.
`INSPIRAL.md`'s "The transfer operator": within an element `φₙ` is the
tensor-product Lagrange polynomial of node `n` and `wₙ` its quadrature
weight, and

    (P S)ₙ = (1/wₙ) ∫_{element(n)} S φₙ,    ⟨P S, g⟩_w = ∫ S g_h

for every grid function `g`, with `g_h = Σₙ gₙ φₙ` its reconstruction,
discontinuous across faces. It is element-local and needs no exchange.

- `TransferTables(T, P; q = P + 3, graded = ((1, 14), (2, 10), (3, 8),
  (4, 7)), rfar = 9, nsinh = 48, nmid = 16, ninner = 24)` is `isbits`
  and sits beside the `Element`, which it leaves alone. It
  holds the Gauss–Lobatto nodes and weights with their barycentric
  weights, so that `φᵢ(ξ) = λᵢ ∏_{j≠i}(ξ - ξⱼ)` anywhere; the far
  rule's `q` Gauss–Legendre points; the `q × (P+1)` Lagrange table at
  them and the same divided by the node weight, `wᴳₖ φᵢ(ξᴳₖ)/ŵᵢ`, which
  is the far rule's one-dimensional factor of `P` (the element's
  Jacobian cancels between the integral and `wₙ`); and the reference
  Gauss points of the nested rule; and the graded rule, a `GaussRule`
  (points, weights and contraction table) for each `(distance, points)`
  pair of `graded`, with `rfar`, the distance from which the far rule
  applies ([Quadrature](#quadrature)). All built in `BigFloat` and
  rounded once. There is no derivative table at the
  far rule's points, since nothing reads one; derivatives come from the
  product form where they are needed.
- `ParticleSupport(z, v, ℓ)`, or `ParticleSupport(m, traj)`: where a
  particle is for the transfer. Its support is the ellipsoid
  `|Λ(v)(x - z)| ≤ ℓ`.
- `project_source!(du, p, t, S, particles; ball, rnear = 1, factor = -1,
  var = 2)` adds `factor · P S` to variable `var` of `du`, by default
  subtracting it from `v`'s rate as the remainder equations want. `S`
  takes an `SVector{D}` and is a kernel argument, so it is `isbits`;
  `ball = (centre, R)` holds its support, and elements that miss it are
  skipped, and blocks that miss it are left out of the launch. Elements
  at least `rfar·hₑ` from every particle take the far rule in two
  kernels launched through `TreeAMR.launch_by_owner!` over the list of
  the ball's blocks, split evenly over the threads (M4, [Prescribed
  paths](#prescribed-paths)): one over `E·q` points per dimension and
  block evaluates `S` into a scratch array (`projection_scratch`), and
  one per element contracts its `q^D` values one dimension at a time, in
  place, into `P S` at its nodes. That is the one place a kernel is not
  launched through `map_blocks!`; it is written so that it could run on
  a device, and has not been. The others run on the host: those at
  least `rnear·hₑ` from every particle take the graded rule, `q` points
  per dimension by their distance from the nearest particle, evaluated
  and contracted one dimension at a time as the points are visited
  (`tensor_rule`), threaded over elements one band at a time, and the few
  dozen closer ones take the nested rule, threaded over the points of
  each element's outermost line and summed afterwards in the rule's
  order, since the element that holds a particle can be a third of the
  nested work. Each element writes only its own nodes, and every sum is
  in a fixed order, so the result does not depend on the thread count. An
  element near two particles takes its rule from the nearer one, since
  the other is then at least about half their separation away. An
  element that two supports meet is refused.
- `pairing(p, t, u, K, s; var = 1)` is `∫ K g_h = ⟨P K, g⟩_w` for a
  particle kernel `K` that vanishes outside the support, `ρ` or `∇ρ`
  (`K` may return a vector). It visits the elements `support_elements`
  finds, those meeting the box `[z - ℓ, z + ℓ]`, at most `2^D` since
  `2ℓ < hₑ`, from the box's corners through `TreeAMR.locate_point` and
  the element arithmetic of `node_coordinates`, and reads the state on
  the host. A `mesh_fold` of `wₙ (P K)ₙ gₙ` gives the same at the cost of
  a projection and a pass over the mesh; the tests keep it as the
  reference ([The transfer's cost](#the-transfers-cost) has both).
  With `continuous = true` it first replaces each node on an element's
  boundary by the average over the elements that share it, which is how
  the inspiral reads its forces ([The averaging](#the-averaging)).
- `reconstruct(p, t, u, x)` and `reconstruct_gradient` give `g_h(x)` and
  `∇g_h(x)` at any point, for the flux spheres (M6b) and the tests. A
  point on a face between elements takes the upper element, as
  `locate_point` gives a point on a face between blocks to the upper
  block.
- `ball_rule(T, n_r, n_θ)`: Gauss in the radius with the `r²` in the
  weights, times Gauss in `cos θ` and `2n_θ` equal steps in `φ`, for the
  averages over a particle's ball in its rest frame (M3 on).

The transfer does not look for periodic images: the source's ball and
every particle's support must lie inside the box, and `support_elements`
refuses a support that reaches a face. On a refined forest every block
the source's ball meets, and every block that holds part of a particle's
support, must be on the finest level ([Nested boxes](#nested-boxes)).
The graded and nested rules and the pairing run on the CPU.

### The averaging

[`src/transfer.jl`](src/transfer.jl) and
[`src/inspiral.jl`](src/inspiral.jl). The particle's support is far
narrower than an element, so while it straddles a face the pairing
`∫(∇ρ)Φ_r,h` reads the jump `[Φ_r,h]` there, a term of size
`Q[Φ_r,h]/ℓ` that the continuum does not have, and M4 measured it as
impulses of tens to hundreds of times radiation reaction at every face
crossing ([Prescribed paths](#prescribed-paths-1) in "Measured
results"). The paper's averaging post-processing (its eq. 98–100) makes
the reconstruction continuous: every node that elements share takes the
mean over them, two on a face, four on an edge and eight at a corner.
On one level coincident nodes have equal weights, so this averaging `A`
is the `⟨,⟩_w`-orthogonal projection onto continuous fields, and
self-adjoint. The inspiral uses it on both sides of the transfer:

    dv -= A P S                                 # the source, written
    F = ∫(∇ρ)(AΦ_r)_h - v∫ρ(AΠ_r)_h             # the force, read

so that `⟨Π_r, A P S⟩_w = ∫ S (AΠ_r)_h` and the force is
`-∂_z ⟨A P ρ(z), Φ_r⟩_w`, the gradient of the discrete interaction energy
as before, with `A P` in place of `P`. A straddling support reads the
jump of `∇(AΦ_r)_h`, of size `Q[∇Φ_r,h]`, not `Q[Φ_r,h]/ℓ`.

- `SharedNodes(fs, ball)` holds the blocks the source's ball meets, the
  ones `project_source!` writes, and for each the same-level block
  across each upper face among them; it is built once, in
  `InspiralProblem`, since the ball does not move.
- `average_shared!(ua, sn, var, Val(P+1))` averages one dimension at a
  time: each pair of coincident nodes across a face along `d` takes
  their mean, inside a block and across its upper face, threaded over
  blocks. The passes together give the group's mean, the same values
  the pairing reads. A face to a block outside the ball's is left alone,
  so at the ball's outermost blocks a node some of whose sharers are
  outside gets no exact mean; the source has fallen to zero there with
  the window.
- `inspiral_rhs!` averages the source alone, without the wave equation's
  part or a second array: in the ball's blocks it projects the source
  into `du`'s first variable, which `wave_rhs!` has just set to `v`,
  averages it there, adds it to `dv`, and writes `du = v` again.
- `pairing(…; continuous = true)` averages the nodes of each element the
  support meets over all the elements that share them, which it finds
  through `locate_element`; it refuses a neighbour on another level, or
  across the domain's faces, where it would need a periodic image.
- `InspiralProblem(…; continuous = true)` is the default;
  `continuous = false` is the transfer and its adjoint alone, as the
  milestones before M4's decision ran, and `particle_forces(…;
  continuous)` reads either way on either problem. M3's and M3b's
  measurements, and M4's first series, ran without the averaging.
- **(open)** Near the finest level's edge the reading refuses a support
  element with a coarser neighbour, one element further out than the
  transfer's own refusal; the nested boxes keep the particles well
  inside.

### Quadrature

`INSPIRAL.md`'s "Quadrature" proposed tensor Gauss–Legendre far from the
particles and, within `r_near ≈ 2hₑ`, a tensor-product sinh rule
(Johnston and Elliott, 2005) that clusters points about the particle on
the scale `ℓ`. The far rule is as proposed, but only from `r_far = 9hₑ`
on. Between `r_near = hₑ` and there a graded tensor Gauss rule takes its
place, and within `r_near` a nested rule split at the support replaces
the sinh rule.

**The profile's edge.** `ρ₀ ∝ (1 - σ²)⁴` is only `C³` at `σ = 1`, and
across such an edge every Gauss rule converges algebraically, the sinh
transformation included, because it is the edge and not the scale `ℓ`
that limits it. In 1D, a bump of half-width `ε = 10⁻³` against a
Lorentzian of the same width, relative errors of the sinh rule:

| points | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| bump `(1 - x²/ε²)⁴` | 0.12 | 0.05 | 2.3e-3 | 4.8e-5 |
| Lorentzian `1/(x² + ε²)` | 1.1e-3 | 5.1e-6 | 4.8e-12 | 2.2e-23 |

`∇ρ` has the same edge (`C²`), and so, weaker, does the source `S`. So
the near rule puts a breakpoint wherever a line crosses the support.

**The nested rule.** One dimension at a time: the outer coordinates are
fixed and the line along the next dimension is integrated, so each line
can have its own breakpoints and its own scale.

- *Breakpoints.* The support is `rᵀMr ≤ ℓ²` with `r = x - z` and
  `M = I + γ²vvᵀ`. With some coordinates fixed and the rest minimized
  over, the form left on the fixed ones is `|r_K|² + c(v_K·r_K)²` with
  `c = γ²/(1 + γ²|v_free|²)` (Sherman–Morrison), so the extent of the
  support's section along the line is the pair of roots of a quadratic.
  The integral over the remaining dimensions also changes character
  where the section starts to cross an element face in one of them, so
  each of those is free or fixed at either face, and every combination
  adds two roots: at most `2·3^(D-d)` breakpoints at level `d`.
- *Inside the support's extent,* `ninner = 24` Gauss points per segment,
  plain on the innermost line, where the integrand is smooth between
  breakpoints, and under `x = a + (b - a)(1 - cos θ)/2` on outer lines,
  where the integral over the remaining dimensions behaves like a
  half-integer power of the distance to the segment's ends (`(c² - t²)^{9/2}`
  for `ρ₀` in 2D). With 16 points the bump in 3D was off by `2·10⁻¹⁰`.
- *Outside,* the sinh rule toward the particle's centre projected onto
  the segment, `ξ = ξc + ε sinh(μu - η)`, with the scale `β` the
  distance from the particle's centre to the line: through the fixed
  coordinates and to the element in the remaining ones. Outside the
  support `β` is at least about `ℓ`. A segment with `β` at least half
  its length takes `nmid = 16` points, a closer one `nsinh = 48`; both
  are converged at the switch, so it moves the result by roundoff.

A tensor sinh rule has one scale per dimension, and that does not fit
every line of an element: with the particle just outside a face, `1/r²`
needed 64 points per dimension at `ℓ/hₑ = 10⁻³`, and `1/r³` had not
converged at 64. That is why the nested rule is used on every near
element and not only on the at most `2^D` the support meets.

**Point counts.** The points per line that bring the nested rule to
`10⁻¹²` of `max|∫ f φₙ|` on a unit element, for a particle of size `ℓ`
centred just outside a face (`f = 1/r` regularized inside, as the
Coulomb field of a uniform ball, and `1/r²`) and inside the element
(`1/r`):

| `ℓ/hₑ` | `5·10⁻²` | `5·10⁻³` | `5·10⁻⁴` | `5·10⁻⁵` |
|---|---|---|---|---|
| face, `1/r` | 24 | 32 | 32 | 40 |
| face, `1/r²` | 32 | 40 | 64 | 64 |
| inside, `1/r` | 24 | 32 | 32 | 32 |

The count grows like `log(hₑ/ℓ)`, as `INSPIRAL.md` wants, and the
suite checks `16 + 8 log₁₀(hₑ/ℓ)` for the first row. On the first run's
source (`ℓ/hₑ = 8·10⁻³`, orders 1–3), against a reference at 64 points:
an element `ℓ` from the particle needs 48 points to reach `6·10⁻¹³`, one
`0.08hₑ` away 24, and one `0.5hₑ` or more away 16, which is where
`nsinh` and `nmid` come from. Integrated against a polynomial of degree
`P` per dimension, the particle's own `ρ` is exact to `10⁻¹²` from
`ℓ/hₑ = 10⁻¹` to `10⁻⁴`, moving or not, with the support straddling an
element corner, and `Σₙ wₙ (P ρ)ₙ = Q/γ`.

**The graded rule, `r_near` and `r_far`.** Wherever two rules meet,
they must agree to `10⁻¹²`, so that a particle crossing the switch does
nothing visible, and away from the support, which only the nested rule
splits at, a tensor Gauss rule converges geometrically in the element's
distance from the particle: the source is analytic there. So the points
per dimension grow toward the particle instead of the nested rule
spreading out. On the first run's source (`hₑ = d/8`, orders 1–3), the
worst error at a node, relative to `∫|S|` over the element, of `q`
Gauss points per dimension against 32, over 200 elements at each
distance `δ` of their nearest point from the nearer particle, in random
directions and offsets, leaving out elements within `hₑ` of the
window's spheres:

| `δ` | `q = 6` | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| `1hₑ` | 3.5e-6 | 2.4e-7 | 1.7e-8 | 7.4e-11 | 2.5e-13 | **8.9e-16** |
| `2hₑ` | 2.5e-8 | 4.1e-10 | 6.5e-12 | **2.7e-15** | 1.0e-15 | 1.1e-15 |
| `3hₑ` | 2.6e-9 | 2.0e-11 | **1.4e-13** | 2.3e-15 | 3.0e-15 | 2.9e-15 |
| `4hₑ` | 7.2e-11 | **3.5e-13** | 3.2e-15 | 4.4e-15 | 2.9e-15 | 1.3e-15 |
| `5hₑ` | 1.7e-11 | 7.1e-14 | 4.8e-15 | 3.5e-15 | 2.4e-15 | 2.7e-15 |
| `8hₑ` | 2.0e-12 | 1.4e-14 | 1.7e-14 | 1.1e-14 | 7.4e-15 | 6.3e-15 |
| `9hₑ` | **3.7e-13** | 3.9e-16 | 1.1e-16 | 7.9e-17 | 8.5e-17 | 9.5e-17 |

The bold entries are the defaults: `graded = ((1, 14), (2, 10), (3, 8),
(4, 7))`, each pair the distance from which a `q` applies, and
`rfar = 9` for the far rule's `q = P + 3 = 6`, which nothing closer
reaches `10⁻¹²` with. Within `rnear = 1` the nested rule applies;
14 points at `0.5hₑ` give `6·10⁻¹²`, and a smaller `rnear` saves
nothing ([The transfer's cost](#the-transfers-cost)). Placed at random,
elements come out worse than on the axes, where the far rule seemed to
reach `10⁻¹²` near `5hₑ`; `HISTORY.md` has that table. A model `1/r³`
source does with fewer points, so it is the real source's structure on
scales larger than `ℓ` that needs them. The suite checks every band and
`r_far` in directions spread over the sphere, and that a source that does not move projects
the same to `10⁻¹²` with and without a particle on either side of
`r_near` and of an element face.

**The window's edge.** Elements cut by the window's spheres `R₁` and
`R₂` carry a far-rule error of `10⁻⁸`–`3·10⁻⁷` relative at `hₑ = d/8`,
because the window is `C^{P+1}` there and nothing splits at it, and
elements within a few `hₑ` of the spheres, not cut by them, reach a few
`10⁻¹²` with the far rule for the same reason. It does
not move with the particles; how it falls with `hₑ` is not measured, and
the `10⁻¹²` of `INSPIRAL.md` does not hold on those elements **(open)**.

### The right-hand side with particles

[`src/inspiral.jl`](src/inspiral.jl), `INSPIRAL.md`'s "The remainder on
OLDG", "The right-hand side" and "Time integration", as far as they need
no solve (`INSPIRAL-PLAN.md`'s M3). The remainder `(Φ_r, Π_r)` is the
package's `(u, v)`, and

    du = v
    dv = Σ_d D_d(D_d u) - P S

with `wave_rhs!` unchanged and the source subtracted after it.

- **The state** is one plain vector: the field's state vector, `u` and
  `v` interleaved block by block as `statearray` has them, followed by
  `z` and `v` of each particle, six numbers per particle.
  `field_state(u, p)` is the field's part as a view, which every function
  that takes a field state accepts with `p.wave`, as TreeAMR's multi-set
  state vectors are used through `stateview`; `particle_state(u, p)` reads
  the particles. `INSPIRAL-PLAN.md` planned an `ArrayPartition` for
  OrdinaryDiffEq; IMEXRungeKutta's stage arithmetic by block owner takes a
  plain `Array` only, and the particles' entries are added to thread 1's
  ranges of the partition (`inspiral_partition`), so that path needs no
  change there.
- `InspiralProblem(wave, models, window, motion; order = 3, tables,
  rnear = 1)` holds the three-dimensional `WaveProblem`, the particle
  models, the window, the motion, the near-field order, the transfer
  tables and the projection's scratch, which is reused across calls and
  carries nothing between them. It refuses a window's ball or a
  particle's support at `t = 0` that reaches a block coarser than the
  finest ([Nested boxes](#nested-boxes)). The motion is `Prescribed(path)`, with
  `path(t)` the particles' `Trajectory`s at `t`, `isbits`;
  `uniform_motion(z, v)` is one, and `circular_motion` and
  `hyperbolic_motion` are M4's ([Prescribed paths](#prescribed-paths)).
  `Dynamic` is M6a's.
- `inspiral_rhs!(du, u, p, t)`: `wave_rhs!` on the field part,
  `project_source!` of `ParticleSource(models, window, path(t))` over the
  window's ball, subtracted from `dv`, and `ż = v`, `v̇ = a` for each
  particle with `a` from the path. A prescribed motion gives the source
  its trajectories whatever the state; the particles' part of the state
  follows the path to the integrator's error, which the tests check. The
  right-hand side does not truncate, writes nothing of `u`, and is a pure
  function of `(u, t)`, with `t` entering only through the path.
- `inspiral_stage!` truncates the field part and is the stage and step
  limiter; `inspiral_integrator` installs it, with the partition, as
  `wave_integrator` installs `truncate_stage!`.
- `inspiral_initial_state(p; t = 0)`: `Φ_r = (1 - W)Ψ̂` and
  `Π_r = (1 - W)𝒟Ψ̂` through `fill_at_nodes!` at the trajectories of time
  `t`, `Π_r` truncated, and the particles from the path. `Ψ̂` is not
  evaluated where `W = 1`. For a particle in uniform motion
  `(1 - W)Φ_C(z + vt)` is the exact remainder at every `t`, since `Φ_C`
  solves the wave equation with the particle's source, so the same
  function at time `t` is the reference the runs are compared with.
- `particle_forces(p, u, t)`: the remainder's force on each particle,
  `F = -(Q/γ)(∇̄Φ_r + vΠ̄_r) = ∫(∇ρ)Φ_r,h - v∫ρΠ_r,h`, through the pairing
  over the support's elements. It is a diagnostic and does not act on the
  particles; the analytic parts of the force are M6a's.
- `ball_norm(p, u, centre, radius; p = 2, vars = 1:2)`:
  `quadrature_norm` over the nodes within a ball, through `mesh_fold`,
  with each node's position and weight from its own block's origin and
  spacing, which reach the fold with each point, so it holds on a
  refined forest too.

**The box.** The field of a charge is not periodic, and in a periodic
box the initial data `(1 - W)Ψ̂` has kinks at the faces, where `Φ_C` and
its images do not match. The source also has a net charge,
`∫S = ∫ρ = Q/γ`, since `∫∇²(WΨ̂) = 0`, which in a periodic box drives the
mean of `Π_r` down at the rate `Q/(γV)`. Neither reaches a point before
a signal from the faces does, so the solution at distance `r` from the
centre is that of an unbounded domain until the clean time. In the
continuum that is `L/2 - r`; on the grid the faces' error travels at
about twice the speed of light, and it is about `(L/2 - r)/2`. Every
error norm of an inspiral run is taken within the ball that is still
clean. [Uniform motion](#uniform-motion) has when the faces' error
arrives and how large it is, and [Nested boxes](#nested-boxes) moves the
faces out.

### Nested boxes

`INSPIRAL-PLAN.md`'s M3b, decided with the owner on 2026-11-01 in
answer to M3's clean-time gate ([Uniform motion](#uniform-motion)): from
M4 on, the inspiral runs on a static forest of nested cubes about the
centre, fixed before the run and not adaptive. What needs resolution,
the window's ball about the orbit, is known in advance, and a regrid
would change the operator and the energy and refine evolved data
([The verdict](#the-verdict)). The field side is the refined forests'
([Refined forests](#refined-forests)); what M3b adds is the forest with
more than two levels and the particles on its finest level.

- **The forest.** `nested_cubes(Val(D), w₀, levels)` gives the regions
  of `wave_forest(…; roots, L = 2^levels·w₀, refined)`: the cube of level
  `k` has the half-width `2^(levels-1-k)·w₀` about the box's centre, so
  the finest level, `levels - 1`, covers the cube of half-width `w₀`, the
  root level the periodic box of side `2^levels·w₀`, and every level has
  `roots·E` elements across, its element width twice the next finer
  one's. Each region refines only the leaves of the level below whose
  centres lie in it, so region `k` makes level `k` and no other. With `roots` a multiple of 4 every cube's faces lie on faces of
  the blocks of the level below, and each cube sits `roots/4` blocks of
  its level inside the next coarser one, so balancing refines nothing
  more (a test).
- **One time step for every level,** the finest level's
  ([Time step](#time-step)), as on every refined forest. There is no
  local time stepping: the source on the finest level is most of the
  work.
- **The transfer on the finest level.** `hₑ`, and with it `rnear`,
  `rfar` and the graded bands, stays the finest level's, so every block
  the transfer touches must be on that level: every block the source's
  ball meets, and every block that holds part of a particle's support.
  `InspiralProblem` checks the window's ball and the supports at
  `t = 0`; every `project_source!` checks its ball and the supports
  again, and every pairing its support, since the particles move.
  Anything else is refused, naming the block's level and how far it
  reaches into the ball (its nearest point's distance from the ball's
  centre), or the support's centre. The checks are a pass over the
  blocks and `2^D` `locate_point`s per particle. On the finest level the
  projection, the pairing and the initial remainder are, element for
  element, those of a uniform forest at the finest spacing: the same
  element boxes give the same rules and the same sums, to an ulp (a
  test, against a uniform box over the finest cube with the same
  coordinates).
- **Norms and initial data.** `ball_norm` weights each node by its own
  block's spacing and places it by its own block's origin, and
  `inspiral_initial_state` fills `(1 - W)Ψ̂` at each block's own nodes
  through `fill_at_nodes!`, so both hold on any level; a ball inside the
  finest level gives the uniform forest's norm (a test).
- **Scripts.** `bin/uniform_motion.jl --levels=n --w0=w` runs M3's
  particle on `n` nested cubes, the finest of half-width `w` (by default
  M3's box, `w = L/2`), `2w·hdiv` elements across every level, which has
  to be a multiple of `4E`; `bin/nested_waves.jl` sends the exact
  outgoing pulse `[g(r - t) - g(r + t)]/r`, `g(s) = exp(-(s/σ)²)
  cos(2πs/λ)`, `σ = λ/2`, through the levels, the field alone, and
  records the error and the exact solution in the ball of each level's
  cube, `w_k = 2^k·w₀`, and the energy. `bin/symmetry/nested_motion.sh`
  and `bin/symmetry/nested_waves.sh` run them on a Symmetry node.

**What the coarse-fine faces send back.** Measured on Symmetry (under
[Nested boxes](#nested-boxes-1) in "Measured results"):

- **Inside M3's clean time the nested boxes give M3's numbers,** the
  errors in every ball and the force, to the digits printed, at `d/4`,
  `d/8` and `d/12`.
- **The coarse-fine face sends back no start-up error.** M3's periodic
  faces spoil the `R₂` ball from `t ≈ 0.19` on, through the kinks the
  non-periodic data have there; at the coarse-fine face at the same
  place the data are smooth, and the `R₂` ball stays clean to at least
  `t = 6`, thirty times longer. In uniform motion the force stays clean
  until the particle leaves the fixed window's `R₁` ball, at `t ≈ 5.7`
  at `v = 0.2`, which is what then spoils it, not the faces.
- **A particle at rest stays clean until the outer faces' error
  arrives;** the coarse-fine faces send back nothing above the
  discretization error. On four and five levels around M3's box the
  errors and the force agree to every digit until `t ≈ 14`, and on five
  levels they stay at their clean level to `t = 20`, the end of the run:
  the force at or below `2.7·10⁻⁷`, the error in the `R₂` ball at
  `2.8–5.0·10⁻⁷`. On four levels the outer faces, at 24, reach the `R₂`
  sphere at `t ≈ 14` and the particle at `t ≈ 16`: through the coarse
  levels their error travels at about 1.5 times the speed of light,
  where it is twice in M3's uniform box, so the clean time at distance
  `r` is about `(L/2 - r)/1.5`. What arrives is mostly a drift of `Φ_r`,
  which exerts little force: the force stays below `6.5·10⁻⁷` to
  `t = 20`.
- **A physical wave crosses a face resolved at 8 elements per wavelength
  or more with about `10⁻³` of its amplitude reflected,** and one at 4
  elements per wavelength with about a tenth, measured at the centre,
  where the reflections of the cube's six faces meet. The radiation of
  the first run (`λ = 16`) is resolved that well out to a face whose
  coarse side has `hₑ = 2`; start-up junk of `λ = 4` only to `hₑ = 1/2`.
  The coarse ghost's degree loss ([Accuracy](#accuracy)) shows in no
  other way.
- **The energy holds** to `2·10⁻¹⁰` relative over `32768` steps on five
  levels.
- **The source cost as the whole forest** in M3b, though it reads only
  the window's ball: its far kernels ranged over every block, and the
  ball's blocks, a few contiguous runs of the forest's order, belonged to
  a few threads. On four levels at `d/12` it was 0.79 s of a 3.66 s step,
  against 0.29 s in M3's uniform box of the same finest cube. Since M4
  the far kernels range over the ball's blocks only, split evenly over
  the threads, which cut the source by 3.1 to 4.7 on the layout for M4
  to M7 ([Prescribed paths](#prescribed-paths-1) in "Measured results");
  the plan's pass over every block on every call remains, and is small.

**The layout for M4 to M7,** chosen by the owner on 2026-11-02. `w₀ = 4`
about the centre of mass, 64 elements across each level at `hₑ = d/8`
(`E = 4`, `roots = 16`), and five levels: an outer box of side 128 at
`7.5·10⁷` nodes, as `INSPIRAL-PLAN.md` estimated, about 14 GB. A
convergence series scales every level's spacing together, keeping the
cubes. The faces at 8 and 16 send the first run's radiation back below
the discretization error, about `2·10⁻⁶` in the finest cube, the one at
32 at `1.5·10⁻⁵` from `t ≈ 60`, and the outer faces nothing before
`t ≈ 96`; four levels let the outer faces in from `t ≈ 38`. Start-up junk of
`λ = 4` comes back from the face at 16 at `10⁻²` from `t ≈ 28`. The
outer faces' error from the non-periodic data, at 1.5 times the speed of
light, reaches the window's edge at about `t ≈ 40`, which covers M4's
circle (`t ≈ 5` and a time to measure over) several times. M7's five
orbits, `t ≈ 90`, need more: `INSPIRAL-PLAN.md` estimates seven levels,
which is M7's to settle.

### Prescribed paths

`INSPIRAL-PLAN.md`'s M4, `INSPIRAL.md`'s validation step 4: one particle
on a path whose acceleration and its derivatives are known exactly, so
that nothing is solved, on M3b's nested boxes in the layout chosen for
M4 to M7 ([Nested boxes](#nested-boxes)). [`src/prescribed.jl`](src/prescribed.jl):

- `circular_motion(centre, radius, speed; phase, normal)` and
  `hyperbolic_motion(z₀, g; direction)` are `Prescribed` motions, `isbits`,
  whose `Trajectory` holds `a`, `ȧ`, `ä` and `a⃛` in closed form: on the
  circle `ȧ = -Ω²v`, so that `(Q²/12π)ȧ` opposes the motion; in
  hyperbolic motion, at rest at `z₀` at `t = 0`, `X(t) = (√(1 + g²t²) -
  1)/g`, with `a = g`, `ȧ = 0` and `ä = -3g³` at `t = 0` (a test checks
  every derivative against dual numbers).
- `retarded_field(path, k, Q, x, t)` is the retarded field of a point
  charge on particle `k`'s worldline and its time derivative. The
  retarded time `s` is the root of `t - s = |x - z(s)|`, by Newton's method
  from `t - |x - z(t)|`, converged to roundoff of the positions; then
  `Φ = -Q/(4πγ(R - R·v))` with `R = x - z(s)` and `v = v(s)`, the
  normalization of "Fields and particles" (`Q/γ` in the lab), and
  `∂ₜΦ = (dΦ/ds)/(1 - R̂·v)` by a dual number in `s`. A point no past light
  cone reaches is refused: behind a particle in eternal hyperbolic motion
  by more than `1/g`. For uniform motion it is `-Q/4π|y|`, at `t = 0` of
  hyperbolic motion `INSPIRAL.md`'s closed form, and it solves the wave
  equation (tests).
- **The exact remainder as initial data.** `inspiral_initial_state(p;
  t, retarded = W_x)` adds `W_x(Φ_ret - Ψ̂_pt)` (and `W_x(∂ₜΦ_ret -
  𝒟Ψ̂_pt)`) to M3's `(1 - W)Ψ̂`, with `W_x` a `Window` and `Ψ̂_pt` the
  particle field of `point_model(m)`, the same particle at `ℓ = 10⁻¹²`. The
  total field is then `W_xΦ_ref + (1 - W_x)Ψ̂`, with
  `Φ_ref = Φ_ret + Ψ̂ - Ψ̂_pt`: the retarded field of the extended particle
  up to its finite-size terms beyond `order`, of relative size `ℓ²` times
  orders not carried. The difference is taken against the point limit
  because `Φ_ret - Ψ̂_pt` is regular at the particle, the two `-Q/4πr`
  cancelling to roundoff, and `Ψ̂ - Ψ̂_pt` is the finite-size part through
  `order` that `Ψ̂` already carries; within `10⁻⁶` of the particle the
  difference is taken as zero. At `t = 0` of hyperbolic motion it is
  `-(Q/4πr) Σ_{k>order} (-gr/2)^k P_k(cos θ)` (a test). The same function
  at a later `t` is the reference: in the ball where `W_x = 1` the
  continuum remainder is it until a signal from where `W_x < 1` arrives.
- `ball_norm(…; hole = (centre, radius))` leaves out a ball, the one
  about the particle, where the exact remainder holds the structures the
  grid cannot follow.
- **Far kernels over the ball's blocks.** `project_source!`'s two far
  kernels now range over the blocks the source's ball meets, listed per
  call, and the launch splits them evenly over the threads instead of
  giving each block to its owner. On nested boxes the ball's blocks are
  a few percent of the forest, contiguous runs in its order that a few
  threads own, which then did all the far work. Every element is
  computed by the same arithmetic whichever thread runs it, so the
  result does not change (the suite's ulp and thread-count tests pass).

**The two runs.** `bin/prescribed.jl`, one particle of the first run
(`Q² = 0.005`, `ℓ = 10⁻³`), its window `R₁ = 1.25`, `R₂ = 2.5` about the
centre `c`, `P = 3`, RK4 at `cfl = 1/2`, on the five nested cubes of
half-width 4 to 64, the particle off the faces' planes by
`o = (1.23, 0.71, 0.93)·10⁻²` as in M3:

- **The slow circle,** chosen here (`INSPIRAL-PLAN.md`, "Left to settle
  later"): about `c + o` in the `xy` plane, of radius `r_c = 0.5`, the
  first run's, at `v = 0.05`, so `Ω = 0.1`, `|a| = 0.005`,
  `|ȧ| = 5·10⁻⁴`, and the radiation-reaction force
  `(Q²/12π)|ȧ| = 6.63·10⁻⁸`; its relativistic correction is `γ⁴ - 1 =
  0.5%`. The particle starts at `c + o + (r_c, 0, 0)` and stays well
  inside `R₁`. Its force is recorded every step to `t = 8`, a fifth of
  the clean time at the window's edge, 0.8 rad of the orbit, with two to
  eight face crossings depending on `hₑ`. The measuring time, also
  chosen here, is `t = 2` to 8, leaving out 0.25 about each crossing:
  the force is averaged over every step there, since a single step
  scatters about the mean by more than `F_RR` up to `d/8`. Started from
  the retarded field, no junk has to leave; starting at `t = 2` leaves
  out the time in which the evolved remainder near the particle takes
  over from the projected exact one, which the reading below measures
  on its own.
- **Hyperbolic motion,** chosen here: `g = 0.1` along `y` from
  `c + o + (0.5, 0, 0)`, until the speed reaches 0.05, `t = 0.50`. `g` is
  the size of the first run's accelerations (`vΩ = 0.08`), and `1/g = 10`
  keeps the retarded field defined over the whole ball `W_x` reaches,
  `W_x` one to 3 and zero from 3.75 about `c`. The error against the
  reference is taken in the ball `R₁` without the ball of radius 1/4
  about the particle, every `1/32`, and the force every step.

**From the retarded field, not only from instantaneous data.** The plan
sets the circle up from instantaneous data, `(1 - W)Ψ̂`, and measures
the force once the junk has left the near zone, at `t ≈ 5`. At `v = 0.05`
and `r_c = 0.5` the near zone reaches `1/Ω = 10`, and the junk is not
gone by `t ≈ 5` ([Prescribed paths](#prescribed-paths-1) in "Measured
results"). So the convergence series starts from the circle's retarded
field (`--data=retarded`, `W_x` one to 24 and zero from 30), whose
remainder is the steady one from the start; the instantaneous data run
beside it at two resolutions, to record the junk.

**What the remainder held that the particle field lacked.** The
particle field is the rest-frame field at the boosted position ([The
near field](#the-near-field)), and the remainder carries the lab-slice
terms it lacks. In M4's first series the largest was the `t′²` term of
`Φ_C`, `-½γ²(v·r)²(α′·∇Φ₀)`, which outside the particle behaves like
`r⁰`: `-(Q/8π) v²(v̂·n̂)²(α′·n̂)`. Its force on the particle is along
`α′`, so radial on the circle and along the acceleration in hyperbolic
motion, and the grid read it the more strongly the finer the elements,
as it read the self-inertia before the near field was carried
(INSPIRAL.md, "Why the near field must be analytic"). That kept M4's
gate shut, and the owner chose on 2026-11-02 to carry the term ahead of
M5; since then the particle field carries it from order 1 on ([The near
field](#the-near-field)), which settles the extension proposed after
M1. `bin/prescribed.jl --reading` reads the exact remainder's force
without evolving it, as it is and with the term put back into it (in
its point form), as it was before; see the measurements.

**Face crossings.** The script records the time of every crossing of a
plane of the finest level's element faces by the particle's centre,
interpolated between steps, beside the force every step. While the
support straddles a face, the pairing `∫(∇ρ)Φ_r,h` holds the jump of the
discontinuous `Φ_r,h` across the face weighted by `ρ` on it, a term of
size `Q[Φ_r,h]/ℓ` that the smooth field does not have
([Prescribed paths](#prescribed-paths-1) in "Measured results"). The
averaging over shared nodes takes that term away, and the inspiral now
writes its source and reads its forces through it ([The
averaging](#the-averaging)): a straddling support then reads the jump of
the gradient, of size `Q[∇Φ_r,h]`, in place of `Q[Φ_r,h]/ℓ`. `--plain`
runs without it, as M4's first two series did, and `--both` records the
force read the other way as well; M4's read-only runs, the source
written without the averaging and the force read both ways, are
`--plain --both` (then `--continuous`).

## File layout

- `src/TreeWaveOLDG.jl`: the module and its exports.
- `src/element.jl`: Gauss–Lobatto nodes and weights, `Element`.
- `src/layout.jl`: `check_layout`, node positions, `fill_at_nodes!`,
  `quadrature_norm`.
- `src/evolution.jl`: the kernels, `WaveProblem` (from a forest, or
  from a state set after `regrid!`), `wave_rhs!`,
  `truncate_state!`, `truncate_stage!`, `initial_state`, `wave_energy`,
  `oldg_timestep`, `convergence_rate`.
- `src/stepping.jl`: `state_partition` and `wave_integrator`.
- `src/sinewave.jl`: `wave_forest`, uniform, two-level or refined level
  by level, `nested_cubes`, the standing sine mode and its driver
  `wave_errors`.
- `src/refinement.jl`: `modal_indicator`, `mode_amplitude`,
  `refine_mark`, `refine_flags` and `refinement_buffer`.
- `src/blast.jl`: the radial blast wave, its exact solution, and its
  drivers `uniform_blast` and `adaptive_blast`.
- `src/precision.jl`, `src/device.jl`: `ceilint`, `floorint`, `wrap`,
  `tofloat64`, `chunk_count`, `chunk_bounds` and `to_backend`, taken from
  TreeWave.
- `src/particle_forms.jl`: generated by `derivations/generate.wl`, not
  edited by hand; the profile's radial functions, the near-field terms
  and their Laplacians, the lab slice of orders 1–3 and its Laplacians
  (coefficients per group, then the groups' sum), and the self-force
  coefficients.
- `src/particle.jl`: `ParticleModel`, `Trajectory`, the boost,
  `rest_frame_derivatives`, the lab slice (`lab_slice(m, Val(N), x, traj)`),
  `Ψ`, `𝒟Ψ`, `ParticleJets`
  and the source bracket, `Window`, and `ParticleSource`.
- `src/transfer.jl`: `gauss_nodes_weights`, `GaussRule`,
  `TransferTables`, `sinh_rule`, `ParticleSupport`, the far, graded and
  nested rules, the
  projection's kernels and `project_source!`, `support_elements`,
  `pairing`, `reconstruct`, `reconstruct_gradient`, `ball_rule`, and
  `check_finest`, which refuses a ball or a support off the finest level.
- `src/inspiral.jl`: `Prescribed`, `uniform_motion`, `InspiralProblem`,
  the state's parts (`field_state`, `particle_state`),
  `inspiral_initial_state` (with the retarded field inside a window, for
  M4), `inspiral_rhs!`, `inspiral_stage!`, `inspiral_partition`,
  `inspiral_integrator`, `particle_forces` and `ball_norm` (with a
  hole).
- `src/prescribed.jl`: `circular_motion` and `hyperbolic_motion`, the
  retarded field of a point charge on any path by root-finding,
  `retarded_field`, and `point_model`.
- `derivations/`: the Wolfram Language derivations (`profile.wl`,
  `nearfield.wl`, `selfforce.wl`, and `labslice.wl`, the `t′²` term of
  `Φ_C` and M5's lab slice), each runnable alone with
  `wolframscript -file` and asserting its identities, the generator
  `generate.wl`, and `common.wl`, which they load.
- `bin/visualize2d.jl`, `bin/Project.toml`: the viewer and its
  environment; `bin/output/` is gitignored.
- `bin/benchmark.jl`: the cost of a step and its parts, one
  tab-separated row per phase, run in the package environment;
  `bin/symmetry/benchmark.sh` runs it at several thread counts on a
  Symmetry AMD node, with `bin/symmetry/simwatch.sh` reporting its
  stages to SimWatch.
- `bin/projection.jl`: the cost of the transfer on the first run's
  source, one tab-separated row per quantity, run in the package
  environment.
- `bin/uniform_motion.jl`: one particle of the first run in uniform
  motion, in M3's uniform box or, with `--levels`, on nested boxes, its
  error against the exact remainder, its force and, with `--cost`, the
  cost of a right-hand side and its parts and the bytes per node, one
  tab-separated row per quantity, run in the package environment;
  `bin/symmetry/uniform_motion.sh` runs M3's convergence series,
  `bin/projection.jl` and its clean-time runs on a Symmetry AMD node, and
  `bin/symmetry/nested_motion.sh` M3b's on nested boxes.
- `bin/prescribed.jl`: one particle of the first run on the slow circle
  or in hyperbolic motion on nested boxes, its force every step in
  components, its face crossings, and its error against the retarded
  field; `--reading` reads the exact remainder's force without evolving
  it; `bin/symmetry/prescribed.sh` runs it on a Symmetry AMD node, one
  run per `path:hdiv:order[:tend]` of `TREEWAVEOLDG_RUNS`.
- `bin/nested_waves.jl`: the exact outgoing pulse through nested boxes,
  the field alone, its error and the exact solution in each level's
  ball, run in the package environment; `bin/symmetry/nested_waves.sh`
  runs it on a Symmetry AMD node.
- `test/reference.jl`: dense global operators built from the element's
  tables, and the state rearranged into global arrays, which the kernels
  are checked against.

`Project.toml` requires TreeAMR 0.2.7 (for the element centering) and
Julia 1.13, as TreeWave does, and IMEXRungeKutta 1.4 (for `reuse`), which
is not registered and comes from its repository through `[sources]`, as
in TreeHydro; `test/Project.toml` names it the same way. TreeAMR 0.2.7
is not registered yet either: until it is, `Project.toml`,
`test/Project.toml` and `bin/Project.toml` take it from the branch of
its release pull request (eschnett/TreeAMR.jl#32) through `[sources]`,
and those entries go once it is. The particle needs
ForwardDiff 1 (time and space derivatives by dual numbers), StaticArrays
1.9 (`isbits` 3-vectors) and LinearAlgebra. `.github/workflows/CI.yml` is TreeWave's test job:
Julia 1.13 on Linux and macOS with `--check-bounds=yes`, plus a
four-thread cell on Linux, with coverage to Codecov, and a viewer job
that renders both blast-wave figures and uploads them, as TreeWave's
does.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'`, about six minutes and
ten seconds on the laptop at one thread and four and a quarter minutes
at four, of which the particle, the transfer, the inspiral, the nested
boxes and the prescribed paths take about four fifths; the inspiral's
coarse 3D runs about 50 s of it on four threads, the nested boxes' about
40 s more and the prescribed paths' about 50 s, and the refined forests'
tests about 40 s on one.

- `element_tests.jl`: the nodes integrate to degree `2P-1`, and no
  further; the wide derivative is the paper's matrix for `P = 2, 3, 4`
  and has its structure for every `P`; across a continuous face it is
  exact to degree `P`; it is summation by parts on a periodic line; the
  projector keeps degree `< P`, removes `L_P`, and is idempotent and
  self-adjoint; each type's table is the `BigFloat` one rounded.
- `layout_tests.jl`: the nodes sit where the element says, face nodes
  coincide bit for bit, the quadrature norm is exact on what the element
  represents, and a layout OLDG cannot run on is refused.
- `evolution_tests.jl`: the right-hand side equals the dense global
  operator in 1D, 2D and 3D, and `initial_state` truncates the initial
  `v` exactly when asked; the in-place truncation is `Π ⊗ … ⊗ Π` in 1D,
  2D and 3D, leaves `u` alone and is idempotent; every stage and every
  step of a truncating run has `v` truncated, RK4 without the limiters
  does not, and `wave_integrator` is that truncating run; the stage
  arithmetic by owner is bitwise the broadcast; the right-hand side
  neither writes `u` nor remembers a call; the energy never grows and its loss
  falls at RK4's rate as the step shrinks; RK4's limit is `cfl ≈ 5/√D`
  for `P = 1, 2, 4`, found by power iteration through `wave_rhs!`.
- `sinewave_tests.jl`: order `P` with truncation for `P = 2, 3, 4` in 1D
  and 2D; without it, `u` is more accurate; `P = 1` is first order;
  `Float32` agrees with `Float64`.
- `refined_tests.jl`: on two-level forests in 1D to 3D, periodic with
  the middle refined and reflecting with a corner at the walls refined,
  `⟨g, D_d u⟩_w + ⟨D_d g, u⟩_w = 0` to roundoff; `dE/dt` from
  `wave_rhs!` vanishes, with and without truncation; the right-hand side
  reads no edge or corner ghost; `D_d` is exact to the operators'
  degrees across a coarse-fine face, and not beyond them at coarse
  nodes; positions are TreeAMR's bit for bit and `quadrature_norm` its
  `volume_weighted_norm`; what a refined forest cannot run is refused;
  the standing mode converges at order `P` for `P = 2, 3, 4` in 1D and
  2D, its error in the refined middle between the uniform runs'; the
  energy is lost to RK4 only, at its rate; RK4's limit is the uniform
  forest's at the finest spacing; a reflecting box refined at its walls
  keeps order `P` and its energy; and, as a measured shortfall, the 2D
  L∞ rate at `P = 3` is below `P - ½` (**(open)**).
- `adaptive_tests.jl`: one mode's amplitude vanishes on data below its
  degree, with ghosts and on a refined forest, and the indicator on
  constants only; both fall as `hₑ^mode` on smooth data, where they are
  equal; the indicator scores every mesh coarser than the blast wave's
  `σ` above every finer one, where mode 2 alone does not, and the
  initial adaptation reaches `hₑ = 0.39σ` at the centre, where mode 2
  alone stops at `1.56σ`; the block verdict and its box; a regrid keeps
  `Σ w u` to `1e-13` in 1D to 3D, coarsening brings `v`'s top mode back
  and the truncation removes it; a problem rebuilt after a regrid is a
  fresh one bit for bit, periodic and reflecting; the adaptive blast
  refines the ring to the cap, its energy changes only at regrids that
  change the mesh, and at the recommended thresholds it matches the
  uniform run at the cap to `1.01` with at most 82% (cap 2) and 65% (cap
  3) of its node-steps. About 36 s at one thread.
- `reflect_tests.jl`: the reflecting box — the Neumann `cos` mode converges
  at order `P` in 1D and 2D, the wall is an exact discrete symmetry
  (energy conserved to the integrator's loss), the parities the wall imposes
  are the ones the field sets carry, and `Float32` agrees at coarse
  resolution.
- `blast_tests.jl`: the exact blast solution at `t = 0` is the initial
  data to `1e-5`; from 8 to 16 blocks per dimension at `P = 3` the L2
  error falls by more than `2³` and the energy lost stays below `1e-6`.
- `particle_tests.jl`: the generated `Φ₀`, `ρ₀` and `B_k`, inside and
  outside, equal `BigFloat` quadrature of their definitions to `1e-14`
  in `Float64`, and so do `δm` and `∬ρρ′|x - x′|`; a particle below
  its classical radius is refused; `□_vΦ_C = ρ` at `v = 0.2` in several
  directions; each order's generated Laplacian is its nested-dual
  Laplacian; `Λ₁` minus the `t′²` coefficient of the displaced `Φ₀` at
  the boosted position has the Laplacian `Q(v·y)²(α′·y)ρ₀′` inside and
  none outside, every `Λ_N` vanishes at `v = 0`, and for a point `Λ₁` is
  `-(Q/8π)v²(v̂·n̂)²(α′·n̂)`; along a motion through rest the bracket is
  `∇²Φ⁽ᴺ⁺¹⁾` at the first order not carried, for orders 0 and 1, order 3
  enters through `-∇²Φ⁽³⁾`, and a point particle's bracket has no
  order-3 part, where without `Λ₁` it would keep `(a·r)²(a·∇Φ₀)`; along
  a motion at `v = 0.19`, inside and outside, the bracket with orders up
  to `N` carried falls as `ε^{N+1}`; the point limits are the
  hyperbolic-motion terms and leave a residual falling as `r³`;
  `rest_frame_derivatives` gives back the `α′`, `B`, `C` of a rest-frame
  worldline seen from a moving lab; each lab-slice order's `Float64`
  Hessian, and its `Float64` closed forms, equal its `BigFloat` closed
  forms to `1e-13`, inside, outside and at the edge; and for uniform
  acceleration seen from a lab where the particle moves, the rest-frame
  derivatives are `gn`, 0 and `-3g³n`, and orders 0–3 with the lab slice
  leave a residual against the Born field on the
  lab's slice falling as `r³`, and without it as `r⁰`; `𝒟Ψ` and the
  bracket equal `BigFloat` finite differences in `t` and `x`; the `Float64` bracket equals the naive
  difference in `BigFloat` to `1e-15 Q|a|/ℓ²` at `v = 0.2` and `0.05`,
  and to `1e-12` relative without the near field, and vanishes to
  `1e-15 Q/ℓ³` without acceleration; the window's step is exactly
  `C^m`, and its closed-form derivatives are its dual-number ones; the
  source equals `∂ₜ²(WΨ̂) - ∇²(WΨ̂) + Σρ` evaluated naively in
  `BigFloat`, and `ParticleSource` is `isbits` and does not allocate;
  and, where `wolframscript` is available, regenerating
  `particle_forms.jl` reproduces it byte for byte.
- `transfer_tests.jl`: the transfer tables are the `BigFloat` ones
  rounded, and the Gauss rules exact to degree `2n - 1`; `(P S)ₙ - S(xₙ)`
  falls at order `P` for `P = 2, 3, 4` in 1D and 2D, and in a coarse 3D
  check; `⟨P S, g⟩_w = ∫ S g_h` to `10⁻¹²` in 1D, 2D and 3D, for a source
  with a compact bump across a face, a Lorentzian peak and a smooth part,
  against a graded composite Gauss rule; the particle's `ρ` against a
  polynomial is exact to `10⁻¹²` from `ℓ/hₑ = 10⁻¹` to `10⁻⁴`, at rest
  and moving, across a corner, and `Σ wₙ (P ρ)ₙ = Q/γ`; the sinh rule
  reaches `10⁻¹²` with `16 + 8 log₁₀(hₑ/ℓ)` points per line; the graded
  rule's contraction is the tensor Gauss rule's, in 1D, 2D and 3D, and on
  the first run's source each band, and the far rule at `r_far`, reach
  `10⁻¹²` at their inner edge in directions spread over the sphere; the
  pairing's second differences in `z` across a face, an edge and a corner
  are the exact ones to `10⁻¹²`, and a source that does not move projects
  the same with a particle on either side of `r_near` and of a face; the
  pairing over the support equals the projection folded over the mesh,
  for `ρ` and `∇ρ`, at rest and moving; `reconstruct` and its gradient
  are exact on polynomials of degree `P` per dimension and give the
  stored value at a node; the ball rule's moments are exact.
- `inspiral_tests.jl`, a coarse 3D box (`P = 3`, `hₑ = 1/2` and `1/4`,
  one particle of `ℓ = 0.01`): the inspiral's right-hand side is
  `wave_rhs!` minus the projected source on the field part, moves the
  particles as their path says, neither writes `u` nor remembers a call;
  every stage and step has `v` truncated, RK4 without `inspiral_stage!`
  does not, and the stage arithmetic by owner is bitwise the broadcast;
  a particle at rest and one moving at `v = 0.2` keep their remainder
  `(1 - W)Φ_C(z + vt)` to an error that falls by more than 3 from
  `hₑ = 1/2` to `1/4` at `t = 1/4`, their force falls by more than 100,
  and their state follows the path to `10⁻¹³`; `ball_norm` is
  `quadrature_norm` over a ball holding the box and a node's value over
  a ball holding only it.
- `nested_tests.jl`, nested cubes about the centre of a coarse 3D box
  (`P = 3`, the finest of half-width 2 at `hₑ = 1/2` and `1/4`, one
  particle of `ℓ = 0.01`): `wave_forest` refines level by level, each
  level inside its cube and outside the next finer one with `roots·E`
  elements across, balanced, and `refined = true` and a single region
  keep their two-level forest; on a three-level forest the projection,
  the pairing, the force and the initial remainder on the finest level
  equal a uniform box's over the finest cube element for element to an
  ulp, with all three of the transfer's rules in the ball, and the
  initial remainder on the coarser levels is `(1 - W)Ψ̂` at their own
  nodes; a ball or a support that reaches a coarser block is refused by
  `InspiralProblem`, `project_source!` and the pairing, naming the
  level, and one that stays on the finest level is not; `ball_norm`
  inside the finest level is the uniform box's; and a particle moving
  at `v = 0.2` keeps its remainder to an error that falls by more than
  3 from `hₑ = 1/2` to `1/4` at `t = 1/4`, its force falls by more than
  100, and its state follows the path to `10⁻¹³`.
- `prescribed_tests.jl`: the circle's and hyperbolic motion's `v`, `a`,
  `ȧ`, `ä` and `a⃛` are the dual-number derivatives of their positions,
  the circle has its radius, speed and `ȧ = -Ω²v`, and hyperbolic motion
  `a = g`, `ä = -3g³` at `t = 0` and proper acceleration `g` later; the
  retarded field is the boosted Coulomb field in uniform motion and
  `INSPIRAL.md`'s closed form at `t = 0` of hyperbolic motion, with
  `∂ₜΦ = 0` there, solves the wave equation away from the particle on
  both paths, its second value is its time derivative, and a point
  behind the particle by more than `1/g` is refused; at `t = 0` of
  hyperbolic motion the initial remainder is the rest of the Legendre
  series beyond orders 2 and 3; on M3b's coarse three-level forest
  (`hₑ = 1/2` and `1/4`, `ℓ = 0.01`, `g = 0.2` to `v = 0.05`), started from
  the retarded field, the error against it away from the particle falls
  by more than 5 and the force by more than 10, and the force at `t = 0`
  is below `10⁻⁸`; on a circle at `v = 0.3` started from the retarded
  field the tangential force is `(Q²/12π)ȧ` to a factor 2, at `t = 0` and
  at `t = 1/8`; and, read without evolving it, the exact remainder of
  hyperbolic motion at `v = 0.05` exerts less than 1% of the force along
  the acceleration it exerts with the `t′²` term put back, and that of
  the circle less than `0.1 F_RR` radially, falling from `hₑ = 1/2` to
  `1/4`, where with the term put back it more than doubles.

## Measured results

### Convergence

The standing sine mode (`m = 1`, unit box) to a quarter period, `E = 2`
elements per block, `roots = 2, 4, 8` blocks per dimension, RK4 at
`cfl = 1/2`. Quadrature-weighted L2 errors of `u` and of `v` at the three
resolutions, and the least-squares rate.

| D | P | truncate | `u` errors | rate | `v` errors | rate |
|---|---|---|---|---|---|---|
| 1 | 1 | yes | 5.06e-1 2.72e-1 1.38e-1 | 0.94 | 3.16 1.70 8.67e-1 | 0.93 |
| 1 | 1 | no | 1.10e-1 2.83e-2 7.12e-3 | 1.98 | 4.92e-1 1.17e-1 2.87e-2 | 2.05 |
| 1 | 2 | yes | 1.00e-1 2.69e-2 6.65e-3 | 1.96 | 7.76e-1 1.88e-1 5.45e-2 | 1.92 |
| 1 | 2 | no | 8.08e-2 3.46e-3 4.38e-4 | 3.76 | 6.93e-1 1.13e-1 2.48e-2 | 2.40 |
| 1 | 3 | yes | 1.48e-2 1.70e-3 2.08e-4 | 3.08 | 9.37e-2 1.15e-2 1.29e-3 | 3.09 |
| 1 | 3 | no | 2.77e-3 8.41e-5 2.59e-6 | 5.03 | 4.38e-2 1.01e-2 1.29e-3 | 2.54 |
| 1 | 4 | yes | 1.42e-3 1.02e-4 6.63e-6 | 3.87 | 1.12e-2 7.99e-4 4.90e-5 | 3.92 |
| 1 | 4 | no | 2.32e-4 6.90e-6 1.44e-7 | 5.33 | 1.00e-2 1.08e-4 2.49e-5 | 4.33 |
| 1 | 5 | yes | 1.12e-4 3.23e-6 9.85e-8 | 5.08 | 7.21e-4 1.98e-5 6.26e-7 | 5.08 |
| 1 | 5 | no | 1.36e-5 1.44e-7 1.10e-9 | 6.79 | 2.56e-4 1.20e-5 4.08e-7 | 4.65 |
| 2 | 1 | yes | 4.35e-1 2.61e-1 1.37e-1 | 0.83 | 3.86 2.32 1.21 | 0.83 |
| 2 | 1 | no | 7.80e-2 2.00e-2 5.04e-3 | 1.98 | 4.92e-1 1.17e-1 2.87e-2 | 2.05 |
| 2 | 2 | yes | 1.04e-1 2.56e-2 6.54e-3 | 2.00 | 8.95e-1 2.58e-1 7.34e-2 | 1.80 |
| 2 | 2 | no | 6.73e-2 4.68e-3 3.67e-4 | 3.76 | 2.59e-1 8.05e-2 2.66e-2 | 1.64 |
| 2 | 3 | yes | 1.47e-2 1.71e-3 2.08e-4 | 3.07 | 1.43e-1 1.55e-2 1.87e-3 | 3.13 |
| 2 | 3 | no | 2.89e-3 3.99e-4 4.85e-5 | 2.95 | 5.23e-2 5.70e-3 8.05e-4 | 3.01 |
| 2 | 4 | yes | 1.46e-3 1.02e-4 6.61e-6 | 3.89 | 1.20e-2 9.31e-4 6.38e-5 | 3.78 |
| 2 | 4 | no | 2.87e-4 5.15e-6 1.05e-7 | 5.71 | 1.01e-2 4.11e-4 2.69e-5 | 4.27 |
| 2 | 5 | yes | 1.12e-4 3.23e-6 9.86e-8 | 5.08 | 1.03e-3 2.90e-5 8.78e-7 | 5.10 |
| 2 | 5 | no | 5.42e-6 3.55e-7 1.28e-8 | 4.36 | 5.99e-4 1.66e-5 4.65e-7 | 5.17 |

With truncation the order is `P` for both variables, in 1D and 2D, as
the paper's pointwise bound (eq. 66) says. Without it, `u`'s error is
smaller at every resolution but its rate is irregular, between `P` and
`P + 2`, and `v`'s rate is often below `P`. The paper's choice buys clean
convergence and costs accuracy in `u`. The combined `(u, v)` rates that
`wave_errors` reports and `sinewave_tests.jl` asserts follow `v`, whose
amplitude is `ω = 2π√D` times `u`'s.

The same 1D runs at `dt = h/10` agree with a separate implementation
using dense global matrices to the three digits printed, for `P = 1 … 6`,
with and without truncation.

### Reflecting walls

The same convergence study on a reflecting box (the Neumann `cos` mode of
[`reflect_exact`](@ref), `m = 1`, unit box, to a quarter period), with
the wall's parities on the field sets and the same quadrature:

| D | P | `l2` errors | rate | `E₁/E₀` |
|---|---|---|---|---|
| 1 | 2 | 6.9e-2  1.98e-2  5.72e-3 | 1.8 | 1.0 |
| 1 | 3 | 4.23e-3  4.81e-4  6.0e-5 | 3.07 | 1.0 |
| 2 | 2 | 9.31e-2  2.64e-2  6.05e-3 | 1.97 | 1.0 |

The order is `P` again, in 1D and 2D, and the wall is an exact
discrete symmetry: the energy is conserved to the integrator's own loss
(`E₁/E₀ = 1.0`), as on the periodic box.

### Refined forests

The standing sine mode (`m = 1`, unit box) to a quarter period, as in
[Convergence](#convergence), on two-level forests with the middle half
refined, `E = 2`, `roots = 4, 8, 16, 32` root blocks per dimension (so
the fine level has the spacing of the uniform forest at twice that), RK4
at `cfl = 1/2` at the finest spacing, with truncation. L2 errors of `u`
and `v` together over the box, with the rate between neighbouring
resolutions, beside the uniform forests at the same `roots`:

| D | P | two-level L2 | rates | uniform L2 | rates |
|---|---|---|---|---|---|
| 1 | 2 | 1.12e-01 3.26e-02 9.12e-03 1.95e-03 | 1.78 1.84 2.23 | 1.34e-01 3.88e-02 1.13e-02 2.22e-03 | 1.79 1.79 2.34 |
| 1 | 3 | 5.66e-03 6.75e-04 8.32e-05 1.04e-05 | 3.07 3.02 3.01 | 8.20e-03 9.27e-04 1.16e-04 1.45e-05 | 3.15 3.00 3.00 |
| 1 | 4 | 3.70e-04 2.37e-05 1.64e-06 8.38e-08 | 3.96 3.85 4.29 | 5.70e-04 3.50e-05 2.34e-06 1.19e-07 | 4.03 3.90 4.30 |
| 2 | 2 | 1.67e-01 4.70e-02 1.09e-02 2.97e-03 | 1.83 2.10 1.88 | 1.83e-01 5.21e-02 1.19e-02 3.26e-03 | 1.82 2.13 1.87 |
| 2 | 3 | 9.63e-03 1.17e-03 1.45e-04 1.86e-05 | 3.04 3.01 2.97 | 1.11e-02 1.33e-03 1.63e-04 2.04e-05 | 3.05 3.03 3.00 |
| 2 | 4 | 5.68e-04 3.97e-05 2.58e-06 1.60e-07 | 3.84 3.94 4.01 | 6.62e-04 4.54e-05 2.91e-06 1.77e-07 | 3.87 3.96 4.04 |

The L2 order is `P`, as on a uniform forest. The same runs' L∞ errors:

| D | P | two-level L∞ | rates | uniform L∞ | rates |
|---|---|---|---|---|---|
| 1 | 2 | 3.10e-01 8.52e-02 2.64e-02 6.07e-03 | 1.86 1.69 2.12 | 2.75e-01 8.21e-02 2.75e-02 5.01e-03 | 1.75 1.58 2.46 |
| 1 | 3 | 2.83e-02 3.06e-03 3.93e-04 4.98e-05 | 3.21 2.96 2.98 | 2.94e-02 3.04e-03 3.93e-04 4.96e-05 | 3.27 2.95 2.99 |
| 1 | 4 | 2.31e-03 7.52e-05 8.22e-06 2.90e-07 | 4.94 3.19 4.83 | 2.23e-03 8.43e-05 8.15e-06 2.93e-07 | 4.72 3.37 4.80 |
| 2 | 2 | 6.81e-01 1.97e-01 4.73e-02 1.32e-02 | 1.79 2.06 1.84 | 6.59e-01 2.05e-01 4.62e-02 1.31e-02 | 1.69 2.15 1.82 |
| 2 | 3 | 4.83e-02 4.94e-03 9.65e-04 1.90e-04 | 3.29 2.36 2.34 | 4.28e-02 5.02e-03 5.84e-04 7.18e-05 | 3.09 3.10 3.03 |
| 2 | 4 | 3.58e-03 4.40e-04 3.15e-05 2.77e-06 | 3.02 3.80 3.51 | 2.31e-03 2.14e-04 1.30e-05 8.65e-07 | 3.43 4.04 3.91 |

In 1D the L∞ error is the uniform coarse run's, within `0.89–1.21`; in
2D it falls short of order `P`, which is the coarse ghost's degree loss
([Accuracy](#accuracy), **(open)**). At `roots = 64` the 2D rates are
`1.28` at `P = 2` and `2.73` at `P = 3`, while the L2 rates are `2.02`
and `2.94`.

Inside the refined region `(1/4, 3/4)^D`, the uniform coarse run, the
two-level forest and the uniform fine run (`2 roots`):

| D | P | roots | L2 coarse, two-level, fine | L∞ coarse, two-level, fine |
|---|---|---|---|---|
| 1 | 2 | 8 | 3.93e-02 1.25e-02 1.13e-02 | 8.21e-02 5.03e-02 2.75e-02 |
| 1 | 2 | 16 | 1.13e-02 2.51e-03 2.23e-03 | 2.75e-02 8.64e-03 5.01e-03 |
| 1 | 3 | 8 | 9.36e-04 2.10e-04 1.16e-04 | 3.04e-03 1.14e-03 3.93e-04 |
| 1 | 3 | 16 | 1.16e-04 1.96e-05 1.45e-05 | 3.93e-04 1.28e-04 4.96e-05 |
| 1 | 4 | 8 | 3.52e-05 4.68e-06 2.35e-06 | 8.43e-05 1.53e-05 8.15e-06 |
| 1 | 4 | 16 | 2.35e-06 2.12e-07 1.19e-07 | 8.15e-06 8.94e-07 2.93e-07 |
| 2 | 2 | 8 | 5.21e-02 1.51e-02 1.19e-02 | 2.05e-01 1.38e-01 4.62e-02 |
| 2 | 2 | 16 | 1.19e-02 4.24e-03 3.25e-03 | 4.62e-02 4.00e-02 1.31e-02 |
| 2 | 3 | 8 | 1.33e-03 3.69e-04 1.63e-04 | 5.02e-03 3.42e-03 5.84e-04 |
| 2 | 3 | 16 | 1.63e-04 6.67e-05 2.04e-05 | 5.84e-04 9.65e-04 7.18e-05 |
| 2 | 4 | 8 | 4.53e-05 1.39e-05 2.91e-06 | 2.14e-04 1.98e-04 1.30e-05 |
| 2 | 4 | 16 | 2.91e-06 1.15e-06 1.77e-07 | 1.30e-05 1.42e-05 8.65e-07 |

In L2 the two-level error lies between the uniform runs', close to the
fine one in 1D; in 2D it is 1.3 to 6.5 times the fine one, since the
coarse part's error reaches the middle within a quarter period. In L∞
the 2D error leaves the bracket at `P = 3` and `4` at `roots = 16`.
`refined_tests.jl` asserts the L2 rates from `roots = 4` to 8, the
brackets at `roots = 4` in the ball of radius `1/4` about the centre
(L2, and L∞ in 1D), and, as a measured shortfall, the 2D L∞ rate at
`P = 3` from `roots = 8` to 16 in that ball (`1.77`, below `P - ½`).

The degree loss, isolated. A refined stripe `(1/4, 3/4) × (0, 1)` has
coarse-fine faces but no corners of the refined region. Its fine part's
L∞ error, with the energy-compatible `R` and with the exact `L²`
projection in its place (an experimental TreeAMR, not the package):

| P | roots | `R`: fine L∞ | rates | exact projection: fine L∞ | rates |
|---|---|---|---|---|---|
| 2 | 4–64 | 4.34e-01 9.15e-02 3.24e-02 8.83e-03 3.86e-03 | 2.25 1.50 1.88 1.19 | 3.83e-01 8.24e-02 2.33e-02 5.29e-03 1.29e-03 | 2.22 1.82 2.14 2.03 |
| 3 | 4–64 | 1.89e-02 2.61e-03 5.58e-04 1.17e-04 2.01e-05 | 2.86 2.22 2.26 2.54 | 1.46e-02 1.62e-03 1.84e-04 2.41e-05 2.89e-06 | 3.17 3.14 2.93 3.06 |
| 4 | 4–32 | 1.13e-03 2.05e-04 1.86e-05 1.54e-06 | 2.47 3.47 3.59 | 1.25e-03 9.82e-05 5.76e-06 3.86e-07 | 3.67 4.09 3.90 |

At `P = 2` and `roots = 128` the `R` run's rate is `1.64`. The L2 error
over the box differs by at most 13% between the two at a quarter period.
At a full period (2D, middle half refined, `roots = 4, 8, 16, 32`):

| P | uniform L2 | two-level, `R` | two-level, exact projection |
|---|---|---|---|
| 3 | 6.85e-04 7.22e-05 4.22e-06 1.23e-07 | 1.54e-03 2.05e-04 3.25e-05 5.55e-06 | 1.32e-03 1.05e-04 7.67e-06 6.08e-07 |
| rates | 3.25 4.10 5.10 | 2.92 2.65 2.55 | 3.66 3.77 3.66 |

At `P = 2` and `4` the full period shows nothing of it: the two-level
L2 error is `0.81–0.82` and `0.83–1.03` of the uniform coarse run's,
with either coarse ghost.

Energy. From a non-separable state over 200 steps at `cfl = 1` and
`1/2` and `1/4` (finest spacing), the relative energy lost falls by
`61.6–63.9` per halving of the step, RK4's `2^6`, on two-level forests
in 1D to 3D, periodic and reflecting, with and without truncation: for
example `1.64e-5`, `2.65e-7`, `4.15e-9` in 2D at `P = 2`. The standing
mode's runs above lose between `2e-8` and `2e-16` of their energy.

Walls. The Neumann mode on a reflecting box with the corner `(0, 1/2)^D`
refined, `roots = 4, 8, 16`:

| D | P | two-level L2 | rates | L∞ rates | `E₁/E₀ - 1` at `roots = 4` |
|---|---|---|---|---|---|
| 1 | 2 | 1.43e-02 3.93e-03 8.77e-04 | 1.86 2.17 | 1.27 2.73 | -3.1e-09 |
| 1 | 3 | 3.57e-04 4.36e-05 5.40e-06 | 3.03 3.01 | 2.85 2.97 | -2.1e-13 |
| 1 | 4 | 1.22e-05 8.16e-07 4.44e-08 | 3.90 4.20 | 3.02 5.06 | -2.2e-14 |
| 2 | 2 | 2.27e-02 5.33e-03 1.44e-03 | 2.09 1.89 | 2.19 1.83 | -1.4e-09 |
| 2 | 3 | 5.95e-04 7.25e-05 9.03e-06 | 3.04 3.00 | 3.11 2.99 | -1.1e-12 |
| 2 | 4 | 1.98e-05 1.28e-06 7.78e-08 | 3.95 4.04 | 4.04 3.91 | -1.2e-13 |

The L2 error in the refined corner lies between the uniform runs' in
every row. With `m = 2`, whose `cos(2πx_d)` does not vanish on the coarse-fine
faces, the 2D L∞ rates at `P = 3` are `3.27, 2.38, 2.34` (to
`roots = 32`), against `3.19, 3.10, 3.01` with the exact projection.

### The RK4 limit

The largest `cfl` (in `dt = cfl · hₑ/(P+1)²`) at which RK4 is stable on
the 1D periodic line, from the eigenvalues of the semi-discrete
Laplacian, `|λ dt| ≤ 2√2`:

| P | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| truncated | 5.66 | 5.20 | 5.24 | 5.17 | 5.10 | 5.03 |
| plain | 5.66 | 5.03 | 4.99 | 4.97 | 4.94 | 4.90 |

The step therefore scales as `hₑ/(P+1)²` with a constant near 5, and in
`D` dimensions as `5/√D`, which `evolution_tests.jl` checks.

On a two-level forest, the middle half refined (`roots = 4`, 3 in 3D,
`E = 2`), the same power iteration through `wave_rhs!` with truncation
gives, as `cfl` at the finest spacing times `√D`, against the uniform
forest at the same `roots`:

| D | P | uniform | two-level | reflecting, corner refined |
|---|---|---|---|---|
| 1 | 1 | 5.66 | 5.75 | 5.76 |
| 1 | 2 | 5.20 | 5.21 | 5.21 |
| 1 | 3 | 5.24 | 5.24 | 5.24 |
| 1 | 4 | 5.17 | 5.18 | 5.18 |
| 2 | 1 | 5.66 | 5.76 | 5.76 |
| 2 | 2 | 5.20 | 5.22 | 5.21 |
| 2 | 3 | 5.24 | 5.27 | 5.24 |
| 2 | 4 | 5.17 | 5.18 | 5.18 |
| 3 | 1 | 5.66 | 6.04 | 6.08 |
| 3 | 2 | 5.20 | 5.29 | 5.28 |

The coarse-fine faces add no mode stiffer than the uniform fine
forest's: the limit is within 2% above the uniform one in 1D and 2D, and
up to 7% above it in 3D, where the fine region is a single refined root
block. `refined_tests.jl` checks the ratio for `D = 1, 2` at
`P = 1, 2, 4` and for `D = 3` at `P = 1`.

### The blast wave's error

`uniform_blast` with `E = 2` and `roots = 4, 8, 16` blocks per
dimension, RK4 at `cfl = 1/2`, to `t = 0.4`; L∞ and L2 errors of `u`
and `∂ₜu` together, whose amplitude starts at about `1/σ`. Four threads,
about a second per run at the finest.

| P | truncate | L∞ errors | L2 errors | L2 rate |
|---|---|---|---|---|
| 2 | yes | 3.61 2.44 1.26 | 6.25e-1 5.65e-1 2.62e-1 | 0.63 |
| 2 | no | 4.67 2.60 6.29e-1 | 8.24e-1 5.75e-1 6.12e-2 | 1.88 |
| 3 | yes | 3.22 6.92e-1 2.06e-2 | 5.29e-1 9.02e-2 3.16e-3 | 3.69 |
| 3 | no | 1.75 1.83e-1 2.24e-2 | 3.45e-1 2.12e-2 2.02e-3 | 3.71 |
| 4 | yes | 5.12e-1 2.94e-2 8.83e-3 | 8.40e-2 2.79e-3 2.98e-4 | 4.07 |
| 4 | no | 4.31e-1 7.81e-2 3.60e-3 | 7.32e-2 8.53e-3 2.17e-4 | 4.20 |

The coarse meshes are not yet in the asymptotic range, with 2 to 6
nodes across `σ`, so the rates are rough. In the viewer's default run
(`P = 4`, 16 × 16 blocks) the error of `u` alone at `t = 0.4` is
`1.8e-4` in L∞ and `1.5e-5` in L2, and the energy lost is `4.7e-10`.

### Adaptive refinement

The blast wave (`σ = 0.08`, `E = 2`, truncated, `cfl = 1/2`) from
`roots = 4`, on eight threads of the laptop, each run a second or two.

**The indicator before the mesh resolves the data.** The largest score
over the mesh of the initial Gaussian, centred on an element corner as
in `adaptive_blast`, on uniform meshes; mode `P - 1` alone against the
indicator, which reads modes 1 to `P - 1`:

| P | | roots 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 3 | mode 2 alone | 2.17e-1 | **2.82e-2** | 5.74e-2 | 2.21e-2 | 6.14e-3 |
| 3 | indicator | 2.17e-1 | 9.58e-2 | 5.74e-2 | 2.21e-2 | 6.14e-3 |
| 4 | mode 3 alone | **4.89e-2** | 7.70e-2 | 1.30e-2 | 1.87e-3 | 2.40e-4 |
| 4 | indicator | 6.22e-2 | 7.70e-2 | 1.30e-2 | 1.87e-3 | 2.40e-4 |

Scanning `hₑ/σ` from 1 to 6 at `roots = 8`, the lowest score at or above
`hₑ = σ` against the score at `hₑ = σ`, centred: mode 2 alone `1.9e-2`
against `6.3e-2`, the indicator `5.7e-2` against `6.3e-2` (`P = 3`);
mode 3 alone `1.5e-3` against `2.8e-2`, the indicator `2.8e-2` (`P =
4`); mode 3 of the plain `P = 3` alone `2.8e-3` against `2.9e-2`, the
indicator `2.9e-2`; mode 4 at `P = 5` alone `3.7e-4` against `1.1e-3`,
the indicator `3.1e-3`. Off-centre (the centre moved by `(0.013,
0.029)`) the two agree.

The initial adaptation (cap 5, `buffer = 2`, `coarsen_tol =
refine_tol/10`) over `σ = 0.04, 0.06, 0.08, 0.10, 0.12` and `roots = 2,
3, 4, 5, 6, 8`, the element width at the centre in `σ`: at `refine_tol =
1e-2` the two give the same width in all 90 cases (`P = 3` mode 2, `P =
4` mode 3, plain `P = 3` mode 3). At `3e-2`, mode 2 alone stops at
`1.56σ` or `1.67σ` in 7 of 30 cases and the plain scheme's mode 3 at
`3.12σ` in 3, where the indicator gives `0.39σ`–`0.42σ` and `0.78σ`;
everywhere else the two agree.

**The thresholds.** Error at the end against the uniform run at the
cap's spacing (L∞ and L2 of `u` and `∂ₜu` together), node-steps as a
fraction of the uniform run's, blocks after the initial adaptation and
at the end, regrids that changed the mesh, and the largest energy change
at a regrid relative to `E₀`. Chunk `0.05` at cap 2 and `0.025` at cap
3, the derived buffer:

| P | cap | refine | coarsen | t | L∞ ratio | L2 ratio | node-steps | blocks | regrids | ΔE at a regrid |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 1e-2 | 1e-5 | 0.2 | 1.000 | 1.000 | 0.79 | 160 → 244 | 3/3 | 2.7e-11 |
| 3 | 2 | 1e-2 | 1e-5 | 0.4 | 1.001 | 1.000 | 0.90 | 160 → 256 | 4/7 | 2.7e-11 |
| 3 | 2 | 1e-2 | 1e-4 | 0.4 | 1.000 | 1.000 | 0.89 | 160 → 256 | 3/7 | 3.8e-11 |
| 3 | 2 | 1e-2 | 1e-3 | 0.4 | 0.967 | 1.003 | 0.82 | 112 → 256 | 5/7 | 1.7e-7 |
| 3 | 2 | 1e-2 | 3e-3 | 0.4 | 1.467 | 1.110 | 0.77 | 112 → 256 | 6/7 | 1.8e-6 |
| 3 | 3 | 1e-2 | 1e-5 | 0.2 | 1.003 | 1.000 | 0.62 | 412 → 880 | 7/7 | 1.2e-11 |
| 3 | 3 | 1e-2 | 1e-5 | 0.4 | 1.003 | 1.000 | 0.80 | 412 → 1024 | 13/15 | 1.2e-11 |
| 3 | 3 | 1e-2 | 1e-4 | 0.4 | 1.535 | 1.145 | 0.73 | 340 → 1024 | 13/15 | 1.9e-8 |
| 3 | 3 | 1e-2 | 1e-3 | 0.4 | 8.087 | 3.899 | 0.63 | 208 → 916 | 14/15 | 7.3e-7 |
| 4 | 2 | 1e-2 | 1e-5 | 0.2 | 0.999 | 1.001 | 0.77 | 160 → 244 | 2/3 | 1.5e-12 |
| 4 | 2 | 1e-2 | 1e-5 | 0.4 | 0.995 | 0.999 | 0.88 | 160 → 256 | 3/7 | 1.5e-12 |
| 4 | 2 | 1e-2 | 1e-4 | 0.4 | 1.041 | 1.120 | 0.81 | 112 → 256 | 5/7 | 7.0e-9 |
| 4 | 2 | 1e-2 | 1e-3 | 0.4 | 4.209 | 5.377 | 0.64 | 88 → 232 | 5/7 | 1.9e-6 |
| 4 | 3 | 1e-3 | 1e-6 | 0.2 | 1.002 | 1.001 | 0.61 | 388 → 880 | 7/7 | 1.8e-13 |
| 4 | 3 | 1e-3 | 1e-6 | 0.4 | 0.997 | 1.002 | 0.79 | 388 → 1024 | 12/15 | 1.8e-13 |
| 4 | 3 | 1e-3 | 1e-5 | 0.4 | 1.243 | 1.571 | 0.71 | 340 → 1012 | 12/15 | 1.3e-10 |
| 4 | 3 | 3e-3 | 1e-4 | 0.2 | 292.6 | 177.6 | 0.25 | 112 → 484 | 7/7 | 7.6e-6 |

The uniform runs at the cap: `P = 3` L∞ `5.59e-2` and `2.06e-2` at cap 2
(`t = 0.2`, `0.4`), `3.99e-3` and `2.28e-3` at cap 3; `P = 4` `5.43e-3`
and `8.83e-3` at cap 2, `6.68e-4` and `6.06e-4` at cap 3. Over the wider
scan (`refine_tol` `3e-3` to `1e-1`, `coarsen_tol` `1e-4` to `1e-2`):

- `refine_tol = 1e-1` leaves the root mesh as it is (`hₑ = 1.56σ`), and
  no later regrid recovers: L∞ 9.0 at `t = 0.2` (`P = 3`, cap 2), energy
  changes of `3.5e-2` at a regrid.
- At `P = 4`, `refine_tol = 3e-2` stops at level 1 and so does the run,
  with the uniform level-1 run's accuracy (4 times the cap-2 run's).
- At `P = 3` any `refine_tol` from `3e-3` to `3e-2` builds the same mesh
  at cap 2: the cap binds.
- With the peak frozen at its initial value instead of refreshed, the
  ring's score falls as it spreads and blocks around it coarsen: `P =
  3`, cap 3, `1e-2`/`1e-4`, L∞ ratio 1.90 instead of 1.04 at `t = 0.2`
  and 7.7 instead of 1.5 at `0.4`; `P = 3`, cap 2, `1e-2`/`1e-3`, 1.42
  instead of 0.97 at `0.4`. It avoids the last row of the table only by
  never leaving level 2 (7.9 times the cap-3 run).

**The chunk and the buffer**, at the recommended thresholds, L∞ ratio
and node-steps to `t = 0.4` (regrids that changed the mesh, of those
done):

| P | cap | chunk short | middle | long (default) | buffer 0 | buffer 2–4 |
|---|---|---|---|---|---|---|
| 3 | 2 | 1.001, 0.90 (4/31) | 1.000, 0.92 (4/15) | 1.001, 0.90 (4/7) | 4.35, 0.66 | 1.001, 0.90 |
| 3 | 3 | 1.003, 0.79 (23/63) | 1.009, 0.81 (18/31) | 1.003, 0.80 (13/15) | 67.0, 0.46 | 1.102, 0.77 |
| 4 | 2 | 0.995, 0.87 (5/31) | 0.993, 0.86 (5/15) | 0.995, 0.88 (3/7) | 15.6, 0.45 | 1.010, 0.83 |
| 4 | 3 | 0.995, 0.78 (20/63) | 1.012, 0.77 (16/31) | 0.997, 0.79 (12/15) | 77.8, 0.42 | 0.962, 0.76 |

The chunks are `0.0125`, `0.025`, `0.05` at cap 2 and `0.00625`,
`0.0125`, `0.025` at cap 3; the default buffer is 8 at `P = 3` and 9 at
`P = 4`. Buffers of 2 and 4 points build the same meshes as each other,
and 6 the default's. The largest energy change at a regrid stays below
`3e-11` at every chunk.

**The wake.** The exact solution's indicator at `t = 0.4` over the
blocks behind the ring (centres at `r < t - 3σ`), on uniform meshes, the
peak being `0.137`:

| P | level 1 | level 2 | level 3 |
|---|---|---|---|
| 3 | 3.3e-3 | 2.1e-3 | 3.3e-4 |
| 4 | 3.0e-4 | 1.2e-4 | 9.9e-6 |

At the recommended thresholds the whole disc behind the ring is at the
cap at `t = 0.4`, at cap 2 and 3. It leaves the cap in the runs with
`coarsen_tol = 1e-3` at `P = 3` cap 3 (all of it) and `P = 4` cap 2 (61%
of it), whose errors are 8 and 4 times the uniform run's.

**Refining evolved data.** The last row of the threshold table, traced:
the start stopped at level 2, and at `t = 0.075` the centre went to
level 3. Error of `v` at the nodes before the regrid `8.30e-3`, after
the transfer and the truncation `8.30e-3`, at `t = 0.1` `0.124`, with
the largest error at block corners near the centre, behind the ring.
From exact data at `t = 0.075` on the same mesh, `t = 0.1` gives
`1.9e-3`, and on the uniform level-3 and level-2 meshes `1.0e-3` and
`4.5e-3`.

### Threads on Symmetry

`bin/symmetry/benchmark.sh`, job 586013 on `cn085`, 2026-10-30: a
dual-socket AMD EPYC 7532 (Zen 2, 64 cores, 8 NUMA domains), Julia
1.13.3, threads pinned (`JULIA_EXCLUSIVE=1`, `-t N,0`), pages placed by
first touch. 3D, `P = 3`, the standing sine mode on 16.8M nodes, in two
block sizes: 4096 blocks of `16³` (`E = 4`) and 512 of `32³` (`E = 8`).
Each time is the least of five. The "before" columns are the package
before the changes of [The right-hand side](#the-right-hand-side),
[Truncation](#truncation) and [The integrator](#the-integrator)
(`main` at 939be39, OrdinaryDiffEq's RK4 with both limiters), run in the
same job on the same node.

At 64 threads, ms:

| phase | `16³`, before | `16³`, after | `32³`, before | `32³`, after |
|---|---|---|---|---|
| RK4 step | 714 | **72.1** | 688 | **63.5** |
| right-hand side | 24.1 | 12.3 | 17.1 | 10.6 |
|   copy of `u`, and its exchange | — | 0.35 + 1.44 | — | 0.34 + 0.74 |
|   gradient pass, and its exchanges | — | 2.73 + 1.42 | — | 2.45 + 0.96 |
|   Laplacian pass | — | 3.84 | — | 3.63 |
| truncation | 80.2 | 0.73 | 78.2 | 0.70 |
| stage arithmetic (empty RHS) | 234 | 17.1 | 234 | 16.9 |

The step over one thread at the same size:

| threads | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `16³`, before, s | 10.04 | 1.98 | 1.21 | 0.869 | 0.714 (14.1×) |
| `16³`, after, s | 2.35 | 0.595 | 0.294 | 0.142 | 0.072 (32.6×) |
| `32³`, before, s | 8.21 | 1.57 | 1.06 | 0.809 | 0.688 (11.9×) |
| `32³`, after, s | 1.93 | 0.478 | 0.240 | 0.122 | 0.064 (30.4×) |

What it says:

- **A step is 10–11 times faster at 64 threads**, and 4.3 times faster
  at one: 3.8 ns per node and step at `32³`. Before, the serial stage
  broadcasts (234 ms, flat from 1 to 64 threads), the truncation with
  its serial copy (80 ms, 9× over one thread) and the fifth right-hand
  side made up most of a 64-thread step; the five right-hand sides were
  12–17% of it.
- **Now the right-hand sides are two thirds of a step** (4 × 10.6 of
  63.5 ms), the stage arithmetic a quarter (16.9 ms, 9.9× over one
  thread: RK4's 15 passes over a 268 MB state stream about 240 GB/s),
  and the truncations 4%. Every phase scales; the stage arithmetic least,
  since it is bandwidth-bound and pinned runs below 64 threads fill one
  NUMA domain at a time.
- **The right-hand side is 1.6–2.0 times faster** than before at 64
  threads, and 2.1–2.4 times at one: the exchanges are about a third of
  what they were in points, and the kernels faster with `d` constant.
  It moves about 100 bytes per node, so at `32³` it streams roughly
  160 GB/s, below the node's 410 GB/s nominal bandwidth.
- **`32³` blocks beat `16³`** by 13% on the step at 64 threads, most of
  it in the exchanges, whose points per node halve. At 8 blocks
  per thread there are still enough to balance.
- Pinned runs below 64 threads fill the NUMA domains one after another,
  so only the 64-thread column compares like with like across sizes;
  before and after compare at every count.

### The source bracket

Measured before M5, with `T₂` alone; the suite's check, `10⁻¹⁵ Q|a|/ℓ²`
against the naive difference, holds with the lab slice carried.
`source_bracket` with orders 1–3 carried, against the naive difference
`∂ₜ²Ψ - ∇²Ψ + ρ` evaluated in `BigFloat`: the largest error over twelve
points at `σ = 0.3, 0.9, 1.5, 4` in three directions, for `Q = 0.07`,
`ℓ = 10⁻³` and a particle moving perpendicular to its acceleration, with
`|ä| = 4va`; relative, and in units of `Qa/ℓ²`, the size of the terms the
cancellation-free bracket subtracts.

| `v` | `a` | cancellation-free, relative | in `Qa/ℓ²` | naive, relative |
|---|---|---|---|---|
| 0.2 | 0.08 | `3.1e-9` | `3.3e-17` | `6.1e-4` |
| 0.05 | 0.0025 | `2.9e-6` | `8.1e-17` | `2.1` |

The naive difference loses all its digits at `v = 0.05`, the speed of
`INSPIRAL.md`'s first tests. `INSPIRAL-PLAN.md`'s estimate of three
digits, `log₁₀(v²/aℓ)`, compared the large terms with `A`, but the
bracket is smaller still: `A`'s order-1 part cancels against `∇²Φ⁽¹⁾`,
and, since M4, the `v²a` part of what is left against `∇²T₂`. The
cancellation-free bracket is good to `10⁻¹⁶` of `Qa/ℓ²`, which is
roundoff in the terms it is formed from. Relative to the bracket that is
up to `3·10⁻⁹` and `3·10⁻⁶` on the line along the acceleration, where
the bracket is now `10⁻⁸` and `10⁻¹¹` of `Qa/ℓ²`; before the `t′²` term
was carried the bracket was larger there and the same error was
`1.7·10⁻¹⁴` and `5.0·10⁻¹³` of it (`HISTORY.md`, "Particles").

### The transfer's cost

`bin/projection.jl`, 2026-10-30, on the laptop (Apple M3 Pro, Julia
1.13.3, four threads) and on Symmetry (job 586064, `cn101`, dual AMD EPYC
7543, 64 threads pinned): the first run of `INSPIRAL.md` (`q² = 0.005`,
`ℓ = 10⁻³`, `v = 0.2`, orders 1–3, the window `R₁ = 1.25`, `R₂ = 2.5`),
`P = 3`, `hₑ = d/8`, in a periodic box of side 5 that just holds the
window: 4.1M nodes, 2.4M of them in the 37 224 elements that meet the
window's ball. Seconds, the least of three runs after a warm-up, with the
source of [The split](#the-split) and the rules of
[Quadrature](#quadrature). The particles sit on element corners, or off
the grid by `(0.31, 0.17, 0.43)hₑ` (`--offgrid`), where a moving particle
generically is.

| | laptop, on corners | laptop, off the grid | Symmetry, on corners | Symmetry, off the grid |
|---|---|---|---|---|
| far elements, points | 31 272, 6.8M | 31 102, 6.7M | | |
| graded elements, points | 5 936, 2.5M | 6 078, 2.7M | | |
| nested elements, points | 16, 2.5M | 44, 1.6M | | |
| one projection | **0.77** | **0.77** | **0.153** | **0.190** |
|   far rule | 0.47 | 0.50 | 0.095 | 0.094 |
|   graded rule | 0.12 | 0.13 | 0.021 | 0.023 |
|   nested rule | 0.18 | 0.14 | 0.036 | 0.068 |
| `wave_rhs!`, whole box | 0.014 | 0.014 | 0.0039 | 0.0039 |
|   its share on the window's nodes | 0.0083 | 0.0081 | 0.0023 | 0.0023 |
| pairing of `ρ`, of `∇ρ` | 0.011, 0.012 | 0.0014, 0.0015 | 0.013, 0.015 | 0.0016, 0.0020 |
| projection and fold of `ρ` | 0.043 | 0.021 | 0.045 | 0.042 |

One evaluation of the source, both particles, is 0.16 µs on the laptop
and 0.40 µs on Symmetry, one thread, against `INSPIRAL.md`'s estimate of
0.3 µs for "roughly ten times the wave equation's work". What it says:

- **The projection is about 70 to 95 times the wave equation's work** on
  the same elements, where the plan's gate is ten. That is the number of
  points: the far rule evaluates the source at 3.4 points per node of its
  elements, the graded rule at 7, and the nested rule at 560 off the
  grid and 2 500 on corners. The gate is the owner's ([Open
  questions](#open-questions)); M2 only measured it.
- **The far rule is more than half,** almost all of it the source (0.03 s
  on the laptop with a source that costs nothing). Its `q = P + 3` is
  what the window's transition and the elements far from the particles
  get, where the source is smooth on the scale of the transition. On
  Symmetry it takes 0.094 s where its 6.7M source evaluations, at
  0.40 µs on one thread, would take 0.042 s spread evenly over 64: its
  kernels run each block on the block's owner, and the blocks that meet
  the window's ball are half the box, owned by some of the threads only
  **(open)**.
- **The graded rule costs what its source evaluations do,** 0.16 µs a
  point on one thread of the laptop, and scales by 3.9 on four.
- **The nested rule costs about two and a half times its source
  evaluations** on the laptop (0.14 s off the grid, where its 1.6M
  evaluations take about 0.06 s), for its breakpoints and the product
  with the element's basis at every point, on the few dozen elements
  within `hₑ` of a particle. With the particles on corners, the 16
  elements that touch them take 160 000 points each. It is threaded over
  the points of each element's outermost line, since the element that
  holds a particle can be a third of its work ([Quadrature](#quadrature);
  `HISTORY.md` has the measurements).
- **The pairing over the support** takes 1.4 to 15 ms for `ρ` and `∇ρ`,
  at 1 to 8 elements whatever the mesh, the most when the support meets
  eight. The `mesh_fold` it replaces, a projection of `ρ` and a fold over
  the 4.1M nodes, takes 21 to 45 ms and grows with the mesh. So the
  pairing stays host-side, over the support's elements, as the plan
  proposed.

### Uniform motion

`bin/symmetry/uniform_motion.sh` on Symmetry, 2026-10-30, jobs 586061,
586064 and 586068 on `cn102`, `cn101` and `cn103`, each a dual AMD EPYC
7543 (Zen 3, 64 cores), Julia 1.13.3, 64 threads pinned. The series' errors
are 586061's and the costs 586064's; 586068 repeated both with the
threaded near pass, the errors to every digit printed and the costs
within 3%. One particle of the first run (`Q² = 0.005`,
`ℓ = 10⁻³`, orders 1–3, the window `R₁ = 1.25`, `R₂ = 2.5`), `P = 3`,
in a periodic box of side `L = 6`, starting `(0.5, 0, 0)` from the
centre, off the element corners, and moving along `y` at `v = 0.2`, or
at rest; RK4 at `cfl = 1/2`. The error is that of `(Φ_r, Π_r)` against
`(1 - W)Φ_C(z + vt)`, the exact remainder, in quadrature-weighted L2 over
a ball, and the force is the remainder's force on the particle.

**Convergence.** At `t = 1/2`, `v = 0.2`, with the least-squares rate:

| `hₑ` | `d/4` | `d/6` | `d/8` | `d/10` | `d/12` | rate |
|---|---|---|---|---|---|---|
| error, ball `R₁` about the centre | 3.25e-6 | 9.18e-7 | 2.60e-7 | 1.40e-7 | 8.16e-8 | 3.43 |
| error, ball `1/2` about the particle | 1.61e-6 | 4.99e-7 | 1.94e-7 | 9.90e-8 | 5.26e-8 | 3.11 |
| error, ball `R₂`, at `t = 1/8` | 7.81e-6 | 8.95e-7 | 4.74e-7 | 2.14e-7 | 5.21e-8 | 4.21 |
| largest force over `t ≤ 1/2` | 2.7e-7 | 2.1e-8 | 3.0e-8 | 9.8e-9 | 2.5e-9 | 3.74 |

against an exact remainder of `1.26·10⁻³` in the `R₂` ball. At rest the
errors agree to two digits and the rates are 3.51, 3.12 and 3.84 over
`d/4 … d/10`, and the force's 3.24. So:

- **The remainder converges at order `P`,** with or without motion, and
  the particle's crossing of element faces, one at `d/12` over the run,
  shows in nothing: the uniformly moving field is exact, the window's
  terms are resolved, and the particle's own field exerts no force.
- **The force is zero to the discretization error,** which falls at
  about order `P` too, irregularly, and is `10⁻⁷` and less: four orders
  of magnitude below the first run's radiation-reaction force, about
  `10⁻⁵`.
- **The particle's state follows its path** to `10⁻¹³`.
- **The `R₂` ball is clean for less than `L/2 - R₂ = 1/2`.** At `t = 1/4`
  its error has stopped falling (`2.2·10⁻⁵` to `7.0·10⁻⁶`, rate 0.97), and
  by `t = 1/2` it grows with resolution (`1.7·10⁻⁴` to `2.8·10⁻⁴`), with
  the L∞ error at the sphere `3·10⁻³`. The faces' error outruns light;
  see the clean time below.

**The clean time.** The same particle at `v = 0.2` to `t = 5`, at `d/4`,
`d/6`, `d/8` and `d/12` (`d/12` to `t = 4.2`, where the job's hour ended),
sampled every `1/16`. When the error in each ball first exceeds three
times its clean level, and the force ten times, against the ball's
distance from the nearest face:

| where | distance | arrives at `d/4`, `d/6`, `d/8`, `d/12` | speed |
|---|---|---|---|
| the `R₂` sphere | 0.5 | 0.25, 0.19, 0.19, 0.19 | 2.0–2.7 |
| the `R₁` sphere | 1.75 | 0.88, 0.81, 0.81, 0.88 | 2.0–2.2 |
| the ball of radius `1/2` about the particle | 1.99 | 0.88, 0.94, 0.94, 0.94 | 2.1–2.3 |
| the particle, its force | 2.49 | 1.19, 1.13, 1.25, 1.19 | 2.0–2.2 |

- **The faces' error travels at about twice the speed of light,** at
  every resolution, so it is a property of the scheme's dispersion and
  not of `hₑ`: the kinks the non-periodic initial data have at the faces
  excite high modes, whose group velocity OLDG does not bound by one. A
  point at distance `r` from the centre of a box of side `L` is clean
  until about `(L/2 - r)/2`, half the time the speed of light allows.
- **Once there it is the whole error,** `10⁴` times the discretization
  error at `d/12`, and the same at every resolution. From about `t = 2`
  on, it is the net charge's drift: the source carries `∫S = Q/γ`, so in
  the periodic box the mean of `Π_r` falls at `Q/(γV) = 3.2·10⁻⁴` and that
  of `Φ_r` at `1.6·10⁻⁴t²`, which predict `1.25·10⁻³` at `t = 3` and
  `3.0·10⁻³` at `t = 4.9` for the error measured at `1.2·10⁻³` and
  `2.9·10⁻³`, more than the remainder itself.
- **The force after the arrival** is `10⁻⁵` to `3·10⁻⁴`, up to thirty
  times the first run's radiation-reaction force, so no force can be
  measured past the clean time. A uniform drift of `Φ_r` exerts none;
  this is its uneven part, and `vΠ̄_r`.
- **What it sizes.** M4's circle measures the force at `t ≈ 5` at
  `r ≈ 0.5`, which needs `L ≥ 21` in a uniform box. That fired a gate of
  `INSPIRAL-PLAN.md`, which the owner answered with nested boxes, its
  M3b.

**The cost** at `v = 0.2`, seconds per call, the least of three after a
warm-up, at 64 threads:

| `hₑ` | `d/4` | `d/6` | `d/8` | `d/10` | `d/12` |
|---|---|---|---|---|---|
| nodes | 0.88M | 3.0M | 7.1M | 13.8M | 23.9M |
| `inspiral_rhs!` | 0.042 | 0.069 | 0.117 | 0.200 | 0.311 |
|   `wave_rhs!` | 0.0023 | 0.0034 | 0.0051 | 0.0081 | 0.0148 |
|   `project_source!` | 0.041 | 0.066 | 0.112 | 0.190 | 0.292 |
| `particle_forces` | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 |
| RK4 step | 0.17 | 0.29 | 0.48 | 0.82 | 1.28 |

and in bytes per node: the state 16, the wave problem's working arrays
54, the projection's scratch 27, RK4's scratch 80, about 180 in all,
which `INSPIRAL-PLAN.md`'s memory tables use.

- **The source is 20 times the wave equation's work over the whole
  box** of side 6, at every resolution, and over 90% of a step. Over the
  window's elements alone it is about 65 times, and 84 for the binary's
  source ([The transfer's cost](#the-transfers-cost)). A larger box
  dilutes it: in a uniform box of side 24, which M4's circle would need,
  it would be about a third of the wave equation's work.
- **Before the near pass was threaded over its points** (job 586061, on
  the same CPU model), the source took 0.118 s at `d/4` and 0.368 s at
  `d/12`: at `d/4` the element that holds the particle was a third of
  the near work and ran on one thread.

### Nested boxes

The suite's coarse forest on the laptop (Apple M3 Pro, four threads),
not Symmetry, so these are no timings: one particle of `ℓ = 0.01` moving
at `v = 0.2`, the window `R₁ = 1`, `R₂ = 1.75`, `P = 3`, on three nested
cubes with the finest of half-width `w₀ = 2`, which is M3's suite box of
side 4 (`inspiral_tests.jl`), against that uniform periodic box at the
same spacing. L2 errors against the exact remainder in the balls about
the centre, and the force:

| `hₑ` | `t` | box | ball `R₂` | ball `R₁` | force |
|---|---|---|---|---|---|
| 1/2 | 1/8 | uniform, side 4 | 5.23e-4 | 2.57e-4 | 4.29e-6 |
| 1/2 | 1/8 | nested, 3 levels | 5.13e-4 | 2.57e-4 | 4.28e-6 |
| 1/2 | 1/4 | uniform, side 4 | 6.65e-4 | 3.29e-4 | 3.81e-5 |
| 1/2 | 1/4 | nested, 3 levels | 5.89e-4 | 3.29e-4 | 3.81e-5 |
| 1/4 | 1/8 | uniform, side 4 | 1.43e-4 | 5.02e-5 | 1.43e-9 |
| 1/4 | 1/8 | nested, 3 levels | 1.40e-4 | 5.02e-5 | 1.43e-9 |
| 1/4 | 1/4 | uniform, side 4 | 2.02e-4 | 3.13e-5 | 1.45e-7 |
| 1/4 | 1/4 | nested, 3 levels | 9.47e-5 | 3.14e-5 | 1.45e-7 |

- **Inside M3's clean ball the two agree,** in `R₁` and in the force to
  the digits shown, as they must before anything from the faces arrives.
- **In the `R₂` ball the nested boxes are cleaner** once the uniform
  box's faces' error has arrived there, from `t ≈ (2 - 1.75)/2 = 1/8`
  on: at `hₑ = 1/4` and `t = 1/4` the uniform box's error has stopped
  falling with `hₑ` and is twice the nested one's, which falls by 6.2
  from `hₑ = 1/2`. The coarse-fine face at `w₀ = 2` sends back less than
  the periodic face there.
- **Two levels give the same** to four digits or better as three at
  these times, so nothing from the second coarse-fine face, at 4, has
  arrived.

**On Symmetry.** `bin/symmetry/nested_motion.sh` and
`nested_waves.sh`, 2026-11-01, jobs 587559 and 587560 on `cn096` and
`cn098`, each a dual AMD EPYC 7543 (Zen 3, 64 cores), Julia 1.13.3, 64
threads pinned.

**Uniform motion on four levels** about M3's box: the finest cube of
half-width `w₀ = 3` is M3's periodic box of side 6, inside cubes of
half-width 6 and 12 in a periodic box of side 48; M3's particle, window
and settings, at `v = 0.2`. At `t = 1/2`, against M3's uniform box
([Uniform motion](#uniform-motion)):

| `hₑ` | `d/4` | `d/8` | `d/12` |
|---|---|---|---|
| error, ball `R₁` about the centre | 3.25e-6 | 2.60e-7 | 8.16e-8 |
| M3 | 3.25e-6 | 2.60e-7 | 8.16e-8 |
| error, ball `1/2` about the particle | 1.60e-6 | 1.94e-7 | 5.26e-8 |
| M3 | 1.61e-6 | 1.94e-7 | 5.26e-8 |
| error, ball `R₂`, at `t = 1/8` | 7.29e-6 | 4.26e-7 | 4.87e-8 |
| M3 | 7.81e-6 | 4.74e-7 | 5.21e-8 |
| largest force over `t ≤ 1/2` | 2.7e-7 | 3.0e-8 | 2.5e-9 |
| M3 | 2.7e-7 | 3.0e-8 | 2.5e-9 |

- **The same numbers,** except in the `R₂` ball at `t = 1/8`, where M3's
  are 7–10% higher: M3's faces' error, which reaches the `R₂` sphere at
  `t ≈ 0.19`, has begun to arrive.
- **The clean time,** at `d/8` to `t ≈ 15`, sampled every `1/16`. The
  error in the `R₂` ball stays at `2.8–5.0·10⁻⁷`, and in the `R₁` ball at
  `3.1–7.8·10⁻⁷`, until `t = 6.1`. The force stays at or below
  `1.3·10⁻⁷` to `t = 4.9`, then rises, `1.6·10⁻⁷` at 5.0, `9·10⁻⁷` at 5.56,
  `4·10⁻⁶` at 5.63, `10⁻⁴` at 6.1. The error grows outward from the
  particle: first the force, then the ball about the particle from 5.9,
  then the `R₁` and `R₂` balls from 6.1. That is the particle reaching
  the fixed window's `R₁` sphere, which it does at `t = 5.73`, not
  anything from the faces. After it the error is the window's: the force
  `10⁻²` to `3·10⁻²` over `t = 7` to 10, the L∞ error at the particle up
  to 1.5, the error in the `R₁` ball `1.4·10⁻³`, more than the remainder.
  The particle reaches the finest cube's face at `t = 15`, where the
  transfer refuses the forest and the run stopped (by design, but the
  runs on 5 to 7 levels did not run after it). A particle at rest is the
  way to see the faces past `t = 5.7`, below.

**At rest,** the clean time: jobs 587798 and 587799, 2026-11-02, on
`cn103` and `cn106`, dual AMD EPYC 7543 like the first two. The same
particle at `v = 0` at `d/8` to `t = 20`, sampled every `1/16`, on four
levels (outer box of side 48, `2.57·10⁷` nodes) and five (side 96,
`3.19·10⁷` nodes):

| | clean level, both, `t ≤ 14` | five levels, `t ≤ 20` | four levels departs from five by 5% at | four levels at `t = 20` |
|---|---|---|---|---|
| `Φ_r` alone, ball `R₂` | `0.8–2.2·10⁻⁸` | as before | `t = 14.5` | `8.2·10⁻⁷` |
| error, ball `R₂` | `2.8–5.0·10⁻⁷` | as before | `t = 15.1` | `3.5·10⁻⁶` |
| L∞ error, ball `R₂` | `0.4–2.0·10⁻⁵` | as before | `t = 15.9` | `2.2·10⁻⁵` |
| error, ball `R₁` | `2.0–7.8·10⁻⁷` | as before | `t = 16.3` | `4.2·10⁻⁶` |
| error, ball `1/2` about the particle | `1.1·10⁻⁷–1.4·10⁻⁶` | as before | `t = 16.8` | `5.3·10⁻⁶` |
| force | `≤ 2.7·10⁻⁷` | as before | `t = 16.1` | up to `6.4·10⁻⁷` |

- **Nothing comes back from the coarse-fine faces,** at 3, 6 and 12
  (and 24 on five levels): the two runs share them and agree to every
  digit printed until `t ≈ 14`, and the five-level run stays at its
  clean level, without a trend, to the end.
- **The four-level run's outer faces,** at 24, arrive first in `Φ_r`
  alone, at the `R₂` sphere at `t ≈ 14`, then in the `R₁` ball and at
  the particle, about 2 later. Against `(L/2 - r)/2 = 10.75` at the `R₂`
  sphere in M3's uniform box, the error crosses the coarse levels at
  about `21.5/14 ≈ 1.5` times the speed of light. By `t = 20` the error
  of `Φ_r` alone has grown fifty-fold, the force not even threefold: a
  near-uniform drift, which exerts none.
- **The net charge's drift** is not measured apart from this: in M3's
  box it shows from `t ≈ 0.19`, here only with the outer faces' error.
- **The cost** at rest equals that in motion: on four levels a step
  takes 1.12 s, `inspiral_rhs!` 0.272 s of which the source 0.255 s; on
  five 1.21 s, 0.291 s and 0.268 s, the wave equation 0.023 s against
  0.018 s.

**The cost** of the same runs, seconds per call, the least of three
after a warm-up, at 64 threads, against M3's uniform box of the finest
cube:

| `hₑ` | `d/4` | `d/8` | `d/12` |
|---|---|---|---|
| nodes, four levels | 3.21M | 25.7M | 86.6M |
| nodes, M3 | 0.88M | 7.1M | 23.9M |
| `inspiral_rhs!` | 0.050 | 0.270 | 0.890 |
|   `wave_rhs!` | 0.0040 | 0.0174 | 0.0954 |
|   `project_source!` | 0.046 | 0.252 | 0.793 |
| `particle_forces` | 0.0037 | 0.0037 | 0.0052 |
| RK4 step | 0.21 | 1.11 | 3.66 |
| RK4 step, M3 | 0.17 | 0.48 | 1.28 |

- **Each coarser level is 7/8 of the finest one's nodes,** the shell
  between two cubes, so four levels are 3.6 times M3's box.
- **The source costs 2.2 to 2.7 times M3's,** though it integrates the
  same elements: about 8 ns per node of the coarse levels, from its far
  kernels' range over every block and the plan's pass over every block
  on every call. It is still most of the step.
- **The wave equation costs 3.4 times M3's at `d/8`** for 3.6 times the
  nodes, **and 6.4 times at `d/12`.**
- **Bytes per node** are M3's: the state 16, the wave problem 54–62, the
  projection's scratch 27, RK4's 80.

**Outgoing waves through the levels,** the field alone:
`[g(r - t) - g(r + t)]/r`, `g(s) = exp(-(s/σ)²)cos(2πs/λ)`,
`σ = λ/2`, about the centre of nested cubes with the finest of
half-width `w₀ = 4` at `hₑ = 1/8`, 64 elements across every level (the
layout chosen for M4 to M7), on four levels (outer box of
side 64, `6.1·10⁷` nodes) and five (side 128, `7.5·10⁷` nodes), each to
`t = 2^levels·w₀`, sampled every `1/2`. The element width is `1/8` in the
cube of half-width 4, `1/4` to 8, `1/2` to 16, 1 to 32 and 2 to 64. The
L∞ error in the cube's inscribed ball of radius 4, against the exact
solution's largest value there while the pulse passes:

| `λ` | levels | while the pulse passes, `t ≤ 10` | after, `t = 12–24` | from the face at 16, `t ≈ 28–40` | from the face at 32, `t ≈ 60–68` |
|---|---|---|---|---|---|
| 4 | 4, 5 | `3–11·10⁻⁵` of 5.9 | `≤ 1.5·10⁻⁴` | `10⁻³` to `9.5·10⁻³` | (images) |
| 16 | 5 | `1–4·10⁻⁶` of 0.61 | `1.1–2.6·10⁻⁶` | `1.4–2.0·10⁻⁶` | `1.0–1.5·10⁻⁵` |
| 16 | 4 | as five | as five | from `t ≈ 38`, the outer faces | (images) |

- **Four and five levels agree to every digit printed until the outer
  faces at 32 come in,** at `λ = 4` until `t ≈ 40`, so what comes back
  before is the inner faces'.
- **`λ = 4`.** The face at 8, whose coarse side has 8 elements per
  wavelength, sends back about `1.5·10⁻⁴` from `t ≈ 14` against the
  pulse's `0.19` there, `8·10⁻⁴`. The face at 16, with 4 elements per
  wavelength on its coarse side, sends back up to `9.5·10⁻³` at the
  centre at `t ≈ 34` against the pulse's `0.09` there: a tenth, where
  the six faces' reflections meet.
- **`λ = 16`.** The faces at 8 and 16 send back nothing above the
  discretization error while the pulse passed, about `2·10⁻⁶`. The face
  at 32, with 8 elements per wavelength on its coarse side, sends back
  `1.5·10⁻⁵` at `t ≈ 68` against the pulse's about `3·10⁻²` there,
  `5·10⁻⁴`.
- **The periodic images** come in when the outer faces are reached: the
  exact solution is not periodic, so from then on the error is theirs.
- **The energy** is constant to `2·10⁻¹⁰` relative in all four runs.
- **The cost:** about 0.28 s a step on four levels and 0.34 s on five,
  the field alone, measurements included.


### Prescribed paths

`bin/symmetry/prescribed.sh` on Symmetry, 2026-11-02, Julia 1.13.3, 64
threads pinned, one dual-socket AMD node per job, in two series. **The
first**, without the `t′²` term in the particle field: jobs 587903
(`cn093`), 587923 (`cn096`), 587926 (`cn093`), 587927 (`cn098`), 587928
(`cn101`) and 587929 (`cn103`) on EPYC 7543 (Zen 3), and 587930, 587933
and 587934 (`cn086`, `cn087`, `cn086`) on EPYC 7532 (Zen 2). **The
second**, with it, after the owner's decision: jobs 587990, 587991,
587992 and 587993 (`cn096`, `cn101`, `cn103`, `cn109`) on EPYC 7543, and
587989 and 587994 (`cn083`, `cn084`) on EPYC 7532. The runs of
[Prescribed paths](#prescribed-paths): the layout for M4 to M7, five
levels, `P = 3`, the first run's particle and window, `v = 0.05` on the
circle of radius 0.5 (`Ω = 0.1`, radiation reaction
`(Q²/12π)|ȧ| = 6.63·10⁻⁸`, written `F_RR` below), `g = 0.1` in hyperbolic
motion to `t = 0.50`. The two `d/16` circles (587992 with the term,
587933 without) ran through a closing of the SSH tunnel to Symmetry and
were collected whole once it was reopened.

**The circle's force, from the retarded field.** The tangential (`F_t`,
along `v`) and radial (`F_r`) components, each divided by the
radiation-reaction force along `v`, `(Q²/12π)ȧ·v̂ = -F_RR` (the script's
`rr`), so that `F_t/F_RR = 1` is radiation reaction opposing the motion
and a positive `F_r/F_RR` points inward, here and in the readings below;
averaged over every step of `t = 2` to 8, leaving out `0.25`
about each face crossing; `sd` is the standard deviation of the single
steps about that mean. Orders 2 and 3 give the same numbers to the
digits shown, in both series, so one row stands for both:

| `hₑ` | first series, `F_t/F_RR` | `F_r/F_RR` | with the `t′²` term, `F_t/F_RR` | `F_r/F_RR` | sd of `F_t` | sd of `F_r` | `F_n/F_RR` |
|---|---|---|---|---|---|---|---|
| `d/4` | 0.97 | 0.30 | 0.949 | 0.14 | 2.5 | 4.8 | -0.06 |
| `d/6` | 0.98 | 0.22 | 0.975 | 0.001 | 1.25 | 2.4 | -0.01 |
| `d/8` | 1.03 | 0.28 | 0.993 | -0.004 | 0.61 | 0.94 | -0.004 |
| `d/12` | 1.06 | 0.33 | 1.003 | 0.000 | 0.105 | 0.14 | -0.007 |
| `d/16` | 1.06 | 0.42 | 1.002 | 0.000 | 0.022 | 0.045 | -0.006 |

- **With the `t′²` term carried, the gate's two conditions hold at
  order 3:** the tangential force converges to radiation reaction, 0.95
  to 1.003, against `γ⁴ = 1.005` for the relativistic correction, and
  the radial force to zero, below 0.005 `F_RR` from `d/6` on. The single
  steps' scatter about the mean, the window's discretization error,
  falls at about second order to `d/8`, fourth to `d/12` and fifth to
  `d/16`.
- **Without it the radial force stayed at 0.2–0.4 `F_RR`,** 0.42 at
  `d/16`, and the tangential one drifted above 1 as the elements shrank
  (1.06 at `d/12` and `d/16`), as the reading below predicts.
- **Orders 2 and 3 agree** to 1% of `F_RR` and better at every
  resolution, in both series, in the force and in the error: at this
  speed and radius the order-3 terms, `ä = Ω²|a|` and `a³`, are too small
  for the grid to see, so the rates 1 and 2 cannot be told apart on the
  circle. Hyperbolic motion shows them (below).

**The exact remainder's force, read without evolving it**
(`--reading`, jobs 587930, 587934 and 587994): the circle's at `t = 0`
and hyperbolic motion's at `t = 0.50`, with the `t′²` term carried and
with it put back into the remainder, as in the first series, in units
of `F_RR` for the circle and absolute for hyperbolic motion. The first
series' diagnostic, which took the term out of the remainder in the
script, and the second series' readings with the term carried agree to
every digit printed:

| `hₑ` | `d/4` | `d/6` | `d/8` | `d/12` | `d/16` |
|---|---|---|---|---|---|
| circle `F_t/F_RR`, with the term | 1.051 | 1.041 | 1.033 | 1.021 | 1.012 |
| without it | 1.091 | 1.100 | 1.125 | 1.157 | 1.079 |
| circle `F_r/F_RR`, with the term | -0.005 | -0.004 | -0.003 | 0.000 | 0.002 |
| without it | 0.144 | 0.254 | 0.374 | 0.550 | 0.545 |
| hyperbolic `F_a`, with the term | -5.6e-10 | -5.6e-11 | -2.4e-10 | -7.9e-10 | -9.4e-10 |
| without it | 7.9e-7 | 7.6e-7 | 7.8e-7 | 1.38e-6 | 2.13e-6 |

- **The first series' radial force was the `t′²` term of `Φ_C`'s
  lab-slice re-expansion.** Without it, the circle's radial force read
  from the exact remainder grows with resolution, 0.14 to 0.55 `F_RR`,
  and hyperbolic motion's is `10⁻⁶`; with it they are below 0.005 `F_RR`
  and `10⁻⁹`. Its `r⁰` structure is not resolved by any `hₑ ≫ ℓ`, and the
  grid reads it the more strongly the finer it is, as `INSPIRAL.md`'s
  table says of any `r⁰` structure. Order 2 reads the same, to `10⁻³`.
- **With it the tangential reading converges to radiation reaction,**
  1.051 to 1.012 at `d/16`, against `γ⁴ = 1.005`; without it it is
  8–16% high and does not converge.

**Face crossings.** The largest `|F_t|` and `|F_r|` within 0.25 of a
crossing of the face plane `x = c + 0.5` (slow, at 0.011 across the
face) and of the `y` planes (at 0.04–0.05, the largest over them), in
units of `F_RR`, order 3 (order 2 gives the same), in the first series
and with the `t′²` term:

| `hₑ` | `x`, `|F_t|` | `x`, `|F_r|` | `y`, `|F_t|` | `y`, `|F_r|` | with the term: `x`, `|F_t|` | `x`, `|F_r|` | `y`, `|F_t|` | `y`, `|F_r|` |
|---|---|---|---|---|---|---|---|---|
| `d/4` | 136 | 612 | 810 | 445 | 129 | 581 | 839 | 461 |
| `d/6` | 40 | 184 | 192 | 163 | 32 | 147 | 187 | 145 |
| `d/8` | 22 | 64 | 48 | 27 | 34 | 75 | 81 | 45 |
| `d/12` | 11 | 56 | 25 | 20 | 4.1 | 19 | 14 | 4.4 |
| `d/16` | 8.9 | 31 | 27 | 18 | 5.1 | 4.4 | 1.6 | 0.9 |

- **Every crossing still gives an impulse above radiation reaction**
  with the term carried, lasting while the support straddles the face,
  `2ℓ` over the speed across it, 0.18 for the slow `x` crossing; at
  `d/4` the run steps through it as a smooth bump of 0.15 time units, not
  a single step. With the term the impulses fall with resolution, at
  about third to fourth order from `d/8` to `d/12` (the slow crossing's
  radial one 75 to 19, the `y` crossings' tangential one 81 to 14); in
  the first series the slow crossing's radial impulse stopped falling
  there, at 64 and 56. At `d/16` the slow crossing's radial impulse is
  4.4 with the term, a rate of about 5 from `d/12`, and 31 without; its
  0.25 shares a `y` crossing 0.16 later, so its row holds both. The
  four `y` crossings apart from it, and the two faster `x` crossings,
  give at most 1.6 `F_t` and 0.9 `F_r` with the term, 27 and 18
  without.
- **The likely cause** is the pairing over a support that straddles a
  face: `∫(∇ρ)Φ_r,h` holds the jump of `Φ_r,h` across the face times `ρ`
  on it, of size `Q[Φ_r,h]/ℓ`. Without the `t′²` term the jump of its
  unresolved `r⁰` structure did not shrink with `hₑ`; with it, the jump
  of the discretization error is what is left. The averaged reading
  below takes that term away, and with it the impulses.

**The averaged reading** (`--continuous`, now `--plain --both`; jobs
588208, `cn104`, `d/4` to `d/8`, and 588209, `cn083`, `d/12`, to
`t = 8`, and 588210, `cn108`, `d/16` to `t = 4`; 2026-11-02, with the
`t′²` term, order 3; `cn083` is an EPYC 7532, the others 7543). The
force as the pairing reads it is the same to the bit as in the runs
above; beside it, the force read from the field made continuous by
averaging shared nodes. The largest `|F_t|` and `|F_r|` within 0.25 of
the slow `x` crossing and of all the other crossings, in units of
`F_RR`, plain and averaged; then the averaged force's mean over the
window and the scatter of both:

| `hₑ` | slow `x`, plain | averaged | others, plain | averaged | averaged `F_t/F_RR` | `F_r/F_RR` | sd of `F_t`, plain → averaged | sd of `F_r` |
|---|---|---|---|---|---|---|---|---|
| `d/4` | 129 / 581 | 5.4 / 8.0 | 839 / 461 | 7.3 / 9.2 | 0.912 | 0.027 | 2.5 → 1.96 | 4.8 → 3.4 |
| `d/6` | 32 / 147 | 1.9 / 2.7 | 187 / 145 | 5.8 / 4.4 | 0.966 | 0.001 | 1.25 → 1.07 | 2.4 → 1.8 |
| `d/8` | 34 / 75 | 1.6 / 1.9 | 81 / 56 | 2.6 / 1.5 | 0.997 | 0.009 | 0.61 → 0.40 | 0.94 → 0.65 |
| `d/12` | 4.1 / 19 | 1.20 / 0.62 | 14 / 4.4 | 1.31 / 0.31 | 1.002 | 0.001 | 0.105 → 0.066 | 0.143 → 0.111 |
| `d/16` (to `t = 4`) | 5.1 / 4.4 | 1.18 / 0.20 | 1.5 / 0.34 | 1.08 / 0.10 | 1.003 | 0.000 | 0.030 → 0.015 | 0.067 → 0.027 |

The `d/16` row's mean and scatter are over `t = 2` to 4, where the plain
reading gives 1.003 and 0.003; its slow crossing's window also holds a
`y` crossing.

- **Averaging takes the impulses away.** They are 20 to 100 times
  smaller at every resolution, and from `d/12` on the largest `|F_t|`
  near a crossing, 1.1–1.3, is radiation reaction itself and its
  scatter, and `|F_r|` stays below 0.7. The slow crossing's radial
  impulse falls 8.0, 2.7, 1.9, 0.62 and 0.20 from `d/4` to `d/16`, at
  about third to fourth order from `d/8`. What is left is the jump of
  the gradient, which the support reads over the crossing as a step
  from one element's gradient to the other's, as the design's "Face
  crossings" estimates ([Prescribed paths](#prescribed-paths)).
- **The means do not change** beyond the scatter: from `d/8` on the
  averaged and the plain reading agree to 0.01 `F_RR`, and at `d/4`, where
  the scatter is twice the force, to 0.04. The scatter away from
  crossings falls by a third to a half, so most of it is not the faces.
- **These runs read through the averaging and wrote without it,** so
  their averaged force was not the gradient of the discrete interaction
  energy. The owner then had the source written through it too, which
  the runs below measure ([The averaging](#the-averaging)).

**The averaged source and force** (the default since 2026-11-03,
[The averaging](#the-averaging); `--both`, jobs 588573, `cn092`, EPYC
7532, `d/4` to `d/8`, 588574, `cn099`, 7543, `d/12`, and 588575, `cn092`,
`d/16`, all to `t = 8`, with the `t′²` term, order 3). The force, and
the force read from the same field without the averaging (`plain`),
as in the table above:

| `hₑ` | slow `x`, force | plain | others, force | plain | `F_t/F_RR` | `F_r/F_RR` | sd of `F_t` | sd of `F_r` |
|---|---|---|---|---|---|---|---|---|
| `d/4` | 5.1 / 7.4 | 141 / 634 | 6.1 / 7.5 | 672 / 368 | 0.927 | 0.022 | 1.69 | 2.99 |
| `d/6` | 1.8 / 2.5 | 30 / 138 | 5.2 / 3.9 | 156 / 125 | 0.970 | 0.001 | 0.93 | 1.59 |
| `d/8` | 1.5 / 1.7 | 28 / 67 | 2.4 / 1.3 | 71 / 50 | 0.998 | 0.008 | 0.35 | 0.57 |
| `d/12` | 1.17 / 0.56 | 3.7 / 18 | 1.28 / 0.28 | 12 / 3.8 | 1.002 | 0.001 | 0.057 | 0.098 |
| `d/16` | 1.17 / 0.17 | 4.9 / 3.0 | 1.13 / 0.11 | 1.6 / 1.0 | 1.002 | -0.001 | 0.014 | 0.024 |

- **The gate's impulse condition holds from `d/12` on:** near every
  crossing the largest `|F_t|` is radiation reaction and its scatter,
  and `|F_r|` stays below 0.6 `F_RR`. The slow crossing's radial
  impulse falls 7.4, 2.5, 1.7, 0.56 and 0.17 from `d/4` to `d/16`.
- **Writing the source through the averaging helps a little more than
  reading through it alone:** the impulses and the scatter are 5–15%
  below the read-only runs', and the field itself jumps less, as the
  plain reading of it shows (28 and 67 at `d/8`'s slow crossing, against
  34 and 75 with the source written plain).
- **The means are the same,** 0.998 and 1.002 `F_RR` at `d/8` and `d/12`
  against `γ⁴ = 1.005`, and the radial force below 0.01.
- **It costs nothing measurable:** a step takes 0.896 s at `d/8` on an
  EPYC 7532 against 0.880 s without (job 587989), and 1.96 s at `d/12`
  on a 7543 against 2.00 s (587990). At `d/16` it took 5.06 s on a 7532,
  where the run without it had a 7543 (4.36 s, 587992), so the two are
  not compared.

**The circle from instantaneous data,** `(1 - W)Ψ̂`, in the first series
(job 587929, order 3, averages over 2 time units): at `d/4` `F_t/F_RR`
is -0.1 to 0.2 to `t = 8`, where the run from the retarded field has 0.9
to 1.2, then falls, -0.1 to -0.7 at `t = 10–18`, -1.6 at 20, -2.9 at 30,
and back to -1.1 at 38, and `F_r/F_RR` falls from about 0.3 to -7.2 at
38; at `d/8`, to `t = 16`, `F_t/F_RR` is 0.0 to 0.1 to `t = 8` and -0.1
to -0.6 from there, and `F_r/F_RR` 0.2 to 0.5 throughout (the first
series' radial offset). The junk does not
leave: the instantaneous field lacks the radiation-reaction field near
the particle for at least `t ≈ 8`, and what arrives from `t ≈ 10` pushes
the force the other way, by more as time goes on. Where it comes from
is not measured apart: the near zone's edge at `1/Ω = 10`, beyond which
`Ψ̂`'s near-field terms, which the instantaneous data carry everywhere,
no longer resemble the retarded field, and the coarse-fine face at 4,
whose reflection of the start-up junk returns from about `t ≈ 8`, both
fit. Neither is radiation reaction within the clean time of the
layout, so the convergence series starts from the retarded field.

**Hyperbolic motion.** The remainder's force along the acceleration,
`F_a`, at times `t` (where `v = gt`), in the first series and with the
`t′²` term:

| `hₑ` | order | first series, `t = 1/4` | `t = 1/2` | with the term, `t = 0` | `t = 1/8` | `t = 1/4` | `t = 1/2` |
|---|---|---|---|---|---|---|---|
| `d/4` | 3 | 1.2e-7 | 7.0e-7 | -7.8e-12 | -3.5e-10 | -5.2e-11 | -2.2e-8 |
| `d/6` | 3 | 2.1e-7 | 1.18e-6 | -3.8e-12 | -3.2e-10 | -1.24e-9 | -8.3e-9 |
| `d/8` | 3 | 2.8e-7 | 1.50e-6 | -2.1e-12 | -3.4e-10 | -1.34e-9 | -2.3e-9 |
| `d/12` | 3 | 3.8e-7 | 1.68e-6 | -7.0e-13 | -3.5e-10 | -1.34e-9 | -3.6e-9 |
| `d/16` | 3 | 4.2e-7 | 1.64e-6 | -7.3e-13 | -3.7e-10 | -1.33e-9 | -2.8e-9 |
| `d/4` | 2 | 1.2e-7 | 7.0e-7 | 1.73e-9 | 2.2e-9 | 2.1e-9 | -2.0e-8 |
| `d/8` | 2 | 2.9e-7 | 1.51e-6 | 1.12e-9 | 1.4e-9 | 9.4e-11 | -7.1e-11 |
| `d/12` | 2 | 3.8e-7 | 1.68e-6 | 5.6e-10 | -6.1e-11 | -8.3e-10 | -2.6e-9 |

- **With the `t′²` term the force along the acceleration falls by two
  to three orders,** from `1.6·10⁻⁶` to a few `10⁻⁹` at `t = 1/2`. What
  is left with order 3 grows as `t²`, that is as `v²`, and is the same at
  every resolution from `d/6` on, `-1.34·10⁻⁹` at `t = 1/4`: a
  continuum force of the remainder, not a grid error, from the parts of
  the lab slice not carried yet (M5's) or the finite-size terms beyond
  them. M5 settled it: it was the lab slice (below). Without the term
  the force grew as `v²` and with resolution until `d/12`, the term's
  reading.
- **At `t = 0` the rates are those of "Why the near field must be
  analytic".** There the remainder is the time-symmetric field beyond
  the orders carried, and nothing has evolved: with order 2 the force
  falls at first order, `1.7·10⁻⁹` to `5.6·10⁻¹⁰` from `d/4` to `d/12`
  (rate 1.0), and with order 3 at second order, `7.8·10⁻¹²` to
  `7.0·10⁻¹³` (rate 2.2), where it stops at `d/16`. After the start the
  `v²` force above covers them.
- **`Φ_r + Φ_p` is the retarded field away from the particle.** The
  error of `(Φ_r, Π_r)` at `t = 1/2` in the ball `R₁` without the ball of
  radius 1/4 about the particle is the same in both series: `3.3·10⁻⁶`,
  `9.4·10⁻⁷`, `2.8·10⁻⁷`, `1.2·10⁻⁷` and `8.7·10⁻⁸` from `d/4` to `d/16`
  with order 3, falling by 12 from `d/4` to `d/8`, against an exact
  remainder of `7.7·10⁻⁷` there in the first series and `9.7·10⁻⁸` with
  the term (whose part of the remainder is now in the particle field).
  With order 3 the error of `Φ_r` alone stops at `2.7·10⁻⁸` at every
  resolution, where order 2's falls to `1.4·10⁻⁹`: the bracket omits
  `d⁴a·∂_äΦ⁽³⁾` ([The particle](#the-particle)), and in hyperbolic motion
  `d⁴a = 45g⁵` at `t = 0`, a source error of about `3·10⁻⁷r²` over the
  window, which over `t = 1/2` leaves about this much in `Φ_r`. On the
  circle `d⁴a` is `Ω⁴|a| = 5·10⁻⁷`, a thousand times smaller.
- **The first series' `d/8` order-3 row is job 587903's,** run before
  the far kernels moved to the ball's blocks and before the retarded
  field's root-finding was made type-stable, which changed neither.

**The cost,** seconds per call at 64 threads, the least of three after
a warm-up, order 3, with the `t′²` term (EPYC 7543; `d/12` from the
circle, the others from hyperbolic motion, which costs the same), and the
first series' (7543, but `d/16` on a 7532):

| `hₑ` | `d/4` | `d/6` | `d/8` | `d/12` | `d/16` |
|---|---|---|---|---|---|
| nodes | 9.4M | 31.9M | 75.5M | 255M | 604M |
| `inspiral_rhs!` | 0.057 | 0.099 | 0.179 | 0.417 | 0.905 |
|   `wave_rhs!` | 0.0059 | 0.028 | 0.073 | 0.190 | 0.452 |
|   `project_source!` | 0.051 | 0.070 | 0.103 | 0.225 | 0.452 |
| RK4 step | 0.24 | 0.42 | 0.74 | 1.96 | 4.33 |
| first series, step | 0.20 | 0.38 | 0.68 | 1.83 | 4.75 |
| first series, `project_source!` | 0.042 | 0.062 | 0.087 | 0.189 | 0.475 |
| before the far kernels' change, step | | | 1.40 | 4.64 | |
| before it, `project_source!` | | | 0.269 | 0.892 | |

- **The `t′²` term costs the source 18–20%** (0.087 to 0.103 s at `d/8`
  and 0.189 to 0.225 s at `d/12`, both on 7543s), a fifth invariant on
  the nested dual numbers, and a step 7–9%. On a 7532 the source at
  `d/8` takes 0.134 s.
- **The far kernels over the ball's blocks** cut the source by 3.1 at
  `d/8` and 4.7 at `d/12` and a step by 2.1 and 2.5: the "before" rows
  are job 587903, on `cn093` like 587926's `d/8`. The source's cost is
  now of the ball's blocks, as on M3's uniform box (0.112 s at `d/8` on
  its 7.1M nodes), and no longer of the forest
  ([Nested boxes](#nested-boxes-1)).
- **A circle to `t = 8`** takes 2048 steps at `d/8`, about 30 minutes,
  and 4096 at `d/16`, about 5.5 hours.

**Hyperbolic motion with the lab slice** (M5; branch
`inspiral-m5-labslice` against `main` at `16217a9`, 2026-11-05,
Julia 1.13.3, 64 threads, EPYC 7543 throughout: readings jobs 590189–91
and 590195–97, evolutions 590192–94 and 590198–200, on `cn097`, `cn106`
and `cn110`). The layout and `g = 0.1` as above, at the times where
`v = 0.05`, 0.1 and 0.2 (`t = 0.501`, 1.005 and 2.041). The exact
remainder's force along the acceleration, read without evolving it
(`--reading`), with the lab slice carried and on `main`:

| `hₑ` | order | `v = 0.05` | 0.1 | 0.2 | `main`, 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|
| `d/4` | 3 | -4.2e-12 | -7.0e-12 | 2.3e-12 | -5.7e-10 | -8.2e-10 | 3.7e-9 |
| `d/6` | 3 | -1.4e-12 | -5.9e-12 | -4.0e-12 | -6.0e-11 | -2.9e-9 | -9.9e-9 |
| `d/8` | 3 | -6.0e-13 | -1.3e-12 | 2.5e-12 | -2.4e-10 | -1.3e-9 | 1.3e-8 |
| `d/12` | 3 | -6.6e-13 | 1.5e-12 | 7.3e-13 | -7.9e-10 | 3.6e-9 | 1.0e-8 |
| `d/4` | 2 | 1.1e-9 | -2.1e-9 | -3.3e-10 | 5.0e-10 | -3.6e-9 | 1.7e-9 |
| `d/6` | 2 | 4.9e-10 | -1.2e-9 | -1.5e-9 | 3.7e-10 | -4.6e-9 | -1.3e-8 |
| `d/8` | 2 | -6.8e-11 | -1.6e-10 | -1.1e-9 | -3.7e-10 | -1.7e-9 | 1.1e-8 |
| `d/12` | 2 | -6.9e-10 | -7.7e-10 | -2.2e-10 | -1.5e-9 | 2.6e-9 | 9.5e-9 |

- **The `v²` force is gone at order 3.** With the lab slice the reading
  is a few `10⁻¹²` at every speed and resolution, the size of the
  `t = 0` reading above, where `main`'s grows with `v` to `10⁻⁸` at
  `v = 0.2` and does not fall with resolution. So `main`'s force was the
  continuum force of the lab slice it did not carry, as M4 guessed. The
  force across the acceleration reads the same: at most `5·10⁻¹²` with
  the lab slice, up to `4·10⁻⁸` on `main`.
- **Order 2 reads about `10⁻⁹`** at every speed, with or without, since
  it does not carry `Λ₃`; with the lab slice it no longer grows with `v`.

**The evolutions** to `t = 2.041` (`v = 0.2`) at `d/4` and `d/8`, orders
2 and 3, from the retarded field. The field's error is `main`'s to
three digits at every time (`err_R1` `2.4·10⁻⁷` at `t = 1/2` and
`6.1·10⁻⁷` at the end at `d/8`, order 3), so the lab slice's terms are
below the evolution's own error away from the particle. The force along
the acceleration, every step, order 3:

| `t` | 0.125 | 0.25 | 0.5 | 1.0 | 1.5 | 2.04 |
|---|---|---|---|---|---|---|
| `d/8`, lab slice | -1.9e-11 | -1.5e-10 | 8.5e-10 | -1.0e-8 | -6.9e-8 | -2.5e-8 |
| `d/8`, `main` | -2.1e-10 | -8.2e-10 | -7.4e-10 | -9.6e-9 | -5.1e-8 | -2.3e-8 |
| `d/4`, lab slice | 2.8e-11 | 1.3e-9 | -2.3e-8 | -6.6e-8 | -1.7e-7 | -4.9e-8 |
| `d/4`, `main` | -2.3e-10 | 4.1e-10 | -2.6e-8 | -6.8e-8 | -1.7e-7 | -1.9e-8 |

- **Up to `t ≈ 1/4` the evolved force is the reading's,** five to ten
  times below `main`'s at `d/8`. After that both grow alike to
  `10⁻⁸`–`10⁻⁷`, and the force across the acceleration with them,
  `3·10⁻⁷` at `d/4` and `3.5·10⁻⁸` at `d/8` averaged over `t = 1/2` to 2,
  falling by 8.5 from `d/4` to `d/8`: the evolved field's
  discretization error, which the exact remainder does not have. The
  evolutions therefore cannot show the lab slice beyond `t ≈ 1/4` at
  these resolutions; the readings do.
- **The cost**, seconds per call at 64 threads after a warm-up (each
  job's own node; the order-3 `d/8` pair on `cn097` and `cn110`):

| `hₑ`, order | `project_source!` | `main` | RK4 step | `main` |
|---|---|---|---|---|
| `d/4`, 3 | 0.394 | 0.053 | 1.64 | 0.25 |
| `d/4`, 2 | 0.133 | 0.038 | 0.59 | 0.21 |
| `d/8`, 3 | 0.845 | 0.103 | 3.71 | 0.75 |
| `d/8`, 2 | 0.273 | 0.074 | 1.44 | 0.63 |

  The source costs 8.2 times `main`'s at order 3 and a step 4.9 times,
  as the per-point measurement above predicts ([The near
  field](#the-near-field)). A circle to `t = 8` at `d/8` would take
  about 2.1 hours rather than 30 minutes, and at `d/16` about a day.
- **The full suite on Symmetry** (job 590202, `cn108`, EPYC 7543) took
  60 minutes on one thread, 61 with `--check-bounds=yes` and 28 on four
  threads, mostly compiling the `BigFloat` particle tests.

## What carried over from TreeWave, and what did not

Carried over: the equation and its variables, the periodic box, the sine
mode and the blast wave with their exact solutions, the 2D viewer of the
blast wave without its refinement panels, `WaveProblem` and the `scatter!` →
`fill_ghosts!` → `map_blocks!` pattern (with `u` copied by a kernel of
its own rather than `scatter!`), the pure right-hand side, the
generic element type, the backend keyword, `ceilint` and `to_backend`,
`convergence_rate`, the static two-level forest of `wave_forest`, and the
conventions in `CLAUDE.md`.

Changed: the ghost width is 1 whatever the order, and the centering is
TreeAMR's element centering, with the element-local operators; there
are `1 + D` exchanges per right-hand side rather than one, each of one
variable; positions, initial data and norms are at the Gauss–Lobatto
nodes under their quadrature; the time step scales as `hₑ/(P+1)²`
rather than `h`; the integrator is IMEXRungeKutta's rather than
OrdinaryDiffEq's.

Carried over, with changes: adaptive refinement. `refine_mark`,
`refinement_buffer`, the chunked cycle and the refreshed global scale
are TreeWave's; the indicator is modal, since Löhner's would fire at
every element face, where the solution jumps; and the dead band between
the thresholds is a factor 1000, not 4 ([The verdict](#the-verdict)).

Not carried over: the travelling pulse, the 1D viewer,
TreeWave's benchmark (this package has its own), the vertex centering,
and the device and threading test files.

## Possible extensions

- **(proposed)** A refinement that does not amplify the evolved data's
  jumps across element faces, so that the dead band between the
  thresholds could be narrower (see the open question below).
- The paper's modal filter (eq. 76), as a term in the right-hand side.
- A comparison with TreeWave's finite differences at equal cost: same
  number of points and same wall clock.
- **(proposed)** A low-storage (2N) explicit Runge–Kutta method in
  IMEXRungeKutta, for the inspiral. Its RK4 keeps the state and five
  scratch arrays, OrdinaryDiffEq's about nine; a 2N scheme keeps two in
  all, which matters where memory sets the box size, as in
  `INSPIRAL-PLAN.md`'s table. It would also cut the stage arithmetic, a
  quarter of a 64-thread step ([Threads on
  Symmetry](#threads-on-symmetry)), by about half.
- The paper's averaging post-processing (its eq. 98–100) for plotting;
  the transfer has it already ([The averaging](#the-averaging)).
- A 1D viewer, a device run, and a subprocess thread-count test, as in
  TreeWave.
- **(proposed)** Cheaper transfer, if the cost gate is to be met (see
  [The transfer's cost](#the-transfers-cost)), in the order the plan
  gives: `q = P + 1` far from the particles, then `INSPIRAL.md`'s
  exterior and core split. Far from the particles the window's spheres
  limit the far rule ([Quadrature](#quadrature)), so a smaller `q` there
  would come with a split at `R₁` and `R₂`. The nested rule's own cost,
  its breakpoints, allocations and per-point basis products, is most of
  its time.
- The graded and nested passes on a device, and the pairing there too;
  they run on the host now.
- A smooth reading kernel of width about `hₑ` for the forces
  (`INSPIRAL.md`, "The transfer operator", and the M4 fallback of
  `INSPIRAL-PLAN.md`), should the gradient's jump that the averaged
  reading leaves matter.

## Open questions

- Whether the coarse ghost's degree loss should be removed. The
  energy-compatible `R` keeps the energy an exact invariant on a refined
  forest, and in `D ≥ 2` costs the L∞ error part of an order at the
  coarse-fine faces, and the L2 error too where the regular error is
  small, as at a full period at `P = 3` (see [Accuracy](#accuracy)). The
  remedy is a degree-preserving compatible operator through a finer
  mortar, upstream in TreeAMR, and the decision is the owner's.
- Whether refining evolved data can be made harmless. Today it is
  avoided by thresholds three orders apart, which keep the ring's
  surroundings at the cap before it arrives and so cost work (61–90% of
  the uniform run's node-steps on the blast wave). The mechanism is
  inferred, not shown: the transfer is exact at the nodes and the same
  mesh from exact data is fine, which points at the face jumps a finer
  element weights more. A transfer that projects them out, or a modal
  filter, would be candidates, at a cost in conservation or energy.
- Whether truncation should stay the default for a linear problem,
  where it costs accuracy in `u` and buys no stability (both schemes are
  linearly stable).
- Whether the bracket should hold `d⁴a·∂_äΦ⁽³⁾`. It is of order 5 near
  the particle but polynomial in `r` across the window, and in
  hyperbolic motion at `g = 0.1` it leaves an error in `Φ_r` of 3.5% of
  the remainder that no resolution removes ([Prescribed
  paths](#prescribed-paths-1)); on the circle and for the binary it is a
  thousand times smaller.
- Whether a circle at `v = 0.05` and `r_c = 0.5` can tell orders 2 and 3
  apart at all: its order-3 terms are too small for the grid to see, and
  the two agree to 1% in both series. A smaller radius, or M5's circle at
  `v = 0.2`, would.
- What to do about the cost of the projection, about 90 times the wave
  equation's work where `INSPIRAL-PLAN.md`'s "Gates and fallbacks" asks
  for ten (see [The transfer's cost](#the-transfers-cost)). It is
  accepted for now and to be revisited, probably with threading or a
  device.
- Whether the far rule's kernels reach the speed their source
  evaluations allow, now that they are balanced over all threads (M4):
  on M3's uniform box they reached about half of it (see [The
  transfer's cost](#the-transfers-cost)); not measured since.
- The quadrature error on the elements the window's spheres cut, which
  `INSPIRAL.md`'s `10⁻¹²` does not cover (see [Quadrature](#quadrature)):
  whether it matters, and if so a split at `R₁` and `R₂` like the one at
  the support, or a smoother window.

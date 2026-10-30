# TreeWaveOLDG design

The scalar wave equation in 2nd-order form, discretized with the
operator-based local discontinuous Galerkin method of Miller and
Schnetter (OLDG, [arXiv:1604.00075](https://arxiv.org/abs/1604.00075);
"the paper" below), on a [TreeAMR](https://github.com/eschnett/TreeAMR.jl)
mesh. It is [TreeWave](https://github.com/eschnett/TreeWave.jl) with the
finite-difference stencil replaced by the OLDG one, and it is the
*unrefined prototype* that TreeAMR's `CODE.md` proposes under
"Element-local DG stencils": it runs OLDG on a uniform forest with the
mesh as it is, to judge the method on its equations before TreeAMR grows
what a refined forest needs.

**Status.** Uniform periodic forests in any number of dimensions, any
order `P ≥ 1`, two initial conditions (a standing sine mode and a 2D
radial blast wave) and a viewer for the second, on the CPU, stepped by
IMEXRungeKutta's RK4 with its stage arithmetic run by block owner. The
first parts of `INSPIRAL.md`'s inspiral exist: the analytic particle
(`INSPIRAL-PLAN.md`'s M1), its field, the field's time derivative, the
effective source and the window, the transfer between particles and
grid (M2), the projection of a source onto the nodes and its adjoint, and
the coupled right-hand side for particles on a prescribed path (M3), in
which a particle at rest or in uniform motion keeps its remainder; see
[Particles](#particles). The suite passes, also with
`--check-bounds=yes` and with four threads.

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
  - [What a refined forest needs](#what-a-refined-forest-needs)
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
  - [Quadrature](#quadrature)
  - [The right-hand side with particles](#the-right-hand-side-with-particles)
- [File layout](#file-layout)
- [Tests](#tests)
- [Measured results](#measured-results)
  - [Convergence](#convergence)
  - [Reflecting walls](#reflecting-walls-1)
  - [The RK4 limit](#the-rk4-limit)
  - [The blast wave's error](#the-blast-waves-error)
  - [Threads on Symmetry](#threads-on-symmetry)
  - [The source bracket](#the-source-bracket)
  - [The transfer's cost](#the-transfers-cost)
  - [Uniform motion](#uniform-motion)
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

- **Uniform forests only.** `check_layout` refuses a refined forest.
  TreeAMR fills a coarse-fine ghost by interpolating equally spaced cell
  centres, and Gauss–Lobatto nodes are not that; see [What a refined
  forest needs](#what-a-refined-forest-needs).
- **Periodic or reflecting boxes.** Every driver builds a fully periodic
  forest by default, and passes no boundary hook to `fill_ghosts!`. A
  box with `reflecting=true` puts a solid (Neumann) wall on every face
  as a TreeAMR reflection symmetry, which the exchange implements by parity-
  signed mirrored transfers rather than a boundary hook; see [Reflecting
  walls](#reflecting-walls). An *outflow* or partly reflecting box would
  need the paper's boundary treatment (its `Ω₀` and `Ω_{K+1}`
  singleton elements), which is not implemented.
- **Two initial conditions**, the standing sine mode and the 2D radial
  blast wave, both on a uniform forest. TreeWave's travelling pulse
  existed to exercise a moving refined region, which is out of scope, and
  the blast wave here keeps only what does not need refinement.
- **One `P` per run**, the same in every element. `P` adaptivity is out
  of scope in TreeAMR too (`N` is the forest's).
- **No filter.** The paper's tunable modal filter (its eq. 76) is not
  implemented; its tests ran without it, with truncation only.

### What a refined forest needs

All of it is TreeAMR's, and is listed in TreeAMR's `CODE.md` under
"Element-local DG stencils (proposed)":

- a per-dimension Gauss–Lobatto centering beside `:cell` and `:vertex`,
  with `P` a field-set property, so that `coordinates`,
  `fill_by_coordinates!`, `boundary_by_coordinates` and
  `volume_weighted_norm` work at the nodes (this package's
  [Positions and quadrature](#positions-and-quadrature) are the stand-ins);
- an element-local operator family: prolongation that evaluates the
  coarse element's polynomial at the fine nodes, and restriction by `L²`
  projection, since a coarse node is not a fine one;
- a change in the ghost schedule builder, because an OLDG ghost sits
  *at* the face, on the other side of the jump, so which side's value it
  takes has to become part of how a ghost names its source;
- the invariant that a fine block covers whole parent elements, `N`
  divisible by `2(P+1)`.

Whether interpolated ghosts at a coarse-fine face keep the energy
estimate is **(open)**; mortar projections that satisfy the
summation-by-parts compatibility condition are the known remedy.

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
computes it.

## The layout

The node-duplicated layout fits TreeAMR's **cell-centered** field set
with `G = 1` and no change to the mesh. A block of `N = E(P+1)` points
per dimension holds `E` elements and owns all `N`, both face nodes of
every element, so the state layout is the usual `N^D` per variable. At
`G = 1` the same-level exchange puts the neighbouring block's first
owned node, its low face node, into the high ghost plane, and its last
owned node into the low one: exactly the outside value the wide
derivative reads. Faces between elements inside a block need no
exchange. The operators a `GhostSchedule` requires are a formality: a
uniform forest has no coarse-fine ghost to fill.

`N` must be a whole number of elements and, for TreeAMR, even; for even
`P` that makes `E` even. `check_layout` and `wave_forest` refuse
anything else with the reason.

### Positions and quadrature

TreeAMR's `coordinates`, `fill_by_coordinates!` and
`volume_weighted_norm` assume equally spaced cell centres. The
replacements, in [`src/layout.jl`](src/layout.jl), take the `Element`:

- `node_coordinates(fs, el, b, I)`: the node of owned point `I` (over
  `1:N`, not a stored index). Element `q = (I-1) ÷ (P+1)` and node
  `i = (I-1) % (P+1)` sit at `origin + h·(q(P+1) + offsets[i])`, and the
  offsets of the two face nodes are the integers `0` and `P+1`, so the
  two face nodes at one position get bit-identical coordinates.
- `fill_at_nodes!(f, fs, el)`: `fill_by_coordinates!` at the nodes, a
  kernel through `map_blocks!`.
- `quadrature_norm(fs, el, u; p)`: the `p`-norm under the tensor-product
  quadrature, normalized by the volume, through TreeAMR's `mesh_fold`.
  Equal weights would over-count the nodes clustered near faces.

## The right-hand side

`wave_rhs!` in [`src/evolution.jl`](src/evolution.jl) does

    u → ufs; fill_ghosts!(ufs)                  # u's face neighbours
    gradient pass; fill_ghosts!(grads[d]) ∀d    # ∂_d u's neighbours along d
    Laplacian pass                              # du, with v from u

The wide derivative along `d` reads one point past its element along
`d` and nowhere else, so the exchange carries exactly that, in three
kinds of field set, all cell-centered:

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
schedules' operators are conservative of order 1, a formality on a
uniform forest, and the one family TreeAMR accepts for a ghost-free
dimension. So a right-hand side makes `1 + D` exchanges. In 3D at
`N = 16` they fill `1736 + 3 · 512` ghost points per block, where one
exchange of `u, v` and one of a `D`-variable gradient set with `G = 1`
filled `5 · 1736`, and `v` is no longer copied. Measured on the
laptop (3D, `P = 3`, 512 blocks of `16³`, one thread) the two
exchanges took 33 ms of a 65 ms right-hand side, and the `1 + D` take
7–12 ms of 26 (the laptop was not idle, and runs differ that much).

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
`cfl = 1/2`.

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
the state and the exact state at the same nodes. What TreeWave's version
has and this one does not: the refined annulus, the refinement criterion
and its scales, the coverage measure, and the backend, centering and
type options of its viewer.

## The viewer

`bin/visualize2d.jl` draws the blast wave, in an environment of its own
(`bin/Project.toml`, with CairoMakie and SixelTerm) so that CairoMakie is
not a dependency of the package, as in TreeWave:

    julia --project=bin bin/visualize2d.jl [--P=4] [--E=2] [--roots=16] [--plain]

It writes `bin/output/blast_2d.png` (`blast_2d_plain.png` with
`--plain`) and shows it inline on a terminal. One figure, three parts:

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
  lost, over time.

The run comes from `uniform_blast`'s observer, so the script has no
time-stepping loop of its own. CI renders both schemes and keeps the
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
that needs no mesh; M2, the transfer between particles and grid; and
M3, the coupled right-hand side for particles on a prescribed path, which
subtracts the projected source after `wave_rhs!` and leaves `wave_rhs!`
unchanged. The particles do not yet move under their forces (M6a).
Sections of `INSPIRAL.md` are named below without links.

The closed forms are derived in the Wolfram Language and generated into
Julia:

- `derivations/profile.wl`, `nearfield.wl` and `selfforce.wl` derive the
  profile's potentials, the near field and its self-force, and assert
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

**Moving particles.** Until the lab-slice re-expansion (M5),
`near_field(m, Val(N), x, traj)` is the rest-frame term at the
rest-frame position `y = Λ(v)(x - z)`, with the proper acceleration
`α′ = γ²Λ(v)a` for `a` and with `ä` as given. At rest it is the
time-symmetric field order by order. Moving, it lacks the re-expansion,
whose part `≈ δm v² a` does not converge, as `INSPIRAL.md` says.

It also lacks something at rest, which `INSPIRAL.md` does not say. The
bracket of [The split](#the-split) takes `∂ₜ²`, which sees the field at
neighbouring times, where `v ≠ 0`. The re-expansion's `t′²` term,
`½(v·r)²(-a·∇Φ₀)`, vanishes at rest, but its second time derivative does
not. So along a trajectory through rest the bracket with orders 1–3
carried keeps `(a·r)²(a·∇Φ₀)` at order 3, which outside the particle is
`(Q/4π)(a·n̂)³` plus terms in `(ℓ/r)²` (`nearfield.wl` asserts the
point form, `particle_tests.jl` the bracket's). It behaves like `r⁰`,
as the whole order-3 residual does with only orders 1–2 carried. Until
M5, carrying order 3 therefore removes only the `ä` part of that
residual. On a circle `a³` is `v²` times `ä`, so at small `v` that part
is suppressed, as `INSPIRAL.md` assumes; in hyperbolic motion at
`t = 0` it is not (see [Open questions](#open-questions)).

### The split

The particle field and its time derivative, `INSPIRAL.md`'s "The
particle field":

- `particle_field(m, x, traj; order)` is `Ψ = Φ_C + Σ_{N ≤ order} Φ⁽ᴺ⁾`,
  `order` from 0 (the particle paper's `Φ_C` alone) to 3, so that M4
  can measure what each order buys.
- `particle_rate(m, x, traj; order)` is `𝒟Ψ`, a dual-number derivative
  in `t` of `Ψ` along the trajectory's Taylor polynomial.

`source_bracket(m, x, traj; order)` is one particle's `𝒟²Ψ - ∇²Ψ + ρ`.
Near the particle `∇²Ψ`, `(v·∇)²Ψ` and `ρ` are each of size `Q/ℓ³`, and
the bracket is assembled without subtracting them:

- `ρ` cancels against `□_vΦ_C = ∇²Φ_C - (v·∇)²Φ_C` analytically, and
  each near-field term has `∇²_x = ∇²_y + γ²(v·∇_y)²`, where
  `γ²(v·∇_y)² = (v·∇_x)² = (v·∂_z)²`. So the bracket is
  `𝒟²Ψ - (v·∂_z)²Ψ`, the part of `Ψ`'s second time derivative the
  acceleration causes, minus `∇²_y` of the near-field terms, which is
  their generated closed form.
- `Ψ` depends on `x` and `t` only through four invariants: `q = |y|²`,
  `α′·y`, `|α′|²` and `ä·y`. With `r = x - z` and `M = Λ(v)² = I + γ²vvᵀ`
  they are `rᵀMr`, `k·r` with `k = γ²Ma`, `γ⁴aᵀMa`, and `e·r` with
  `e = Λ(v)ä`, so their dependence on time is through `r`, with `ṙ = -v`
  and `r̈ = -a`, and through `M`, `k`, `γ⁴aᵀMa` and `e`, which do not
  depend on `x`. Of each invariant `I` the bracket needs the rate `İᵤ`
  of uniform motion (`r` moving with `-v`, everything else held) and the
  acceleration's parts `δİ = İ - İᵤ` and `δÏ = Ï - Ïᵤ`, written out
  term by term: for `q`, `δq̇ = rᵀṀr` and
  `δq̈ = -2aᵀMr - 4vᵀṀr + rᵀM̈r`. Then

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
`x`-independent `M`, `k`, `|α′|²` and `e` with their first two time
derivatives, by dual numbers along the trajectory, once per source and
not per point, and is `isbits`, so that it can be a kernel argument. In
the window's transition the same evaluation of `Ψ` carries a second
outer partial seeded with `∇W·∇I`, which gives `∇W·∇Ψ`, and its value
gives `Ψ`. On the laptop a point inside the window costs 0.14 µs for
both particles and one in the transition 0.22 µs, of which 11 ns is
`window_terms`; building the jets costs 0.8 µs per particle
([The transfer's cost](#the-transfers-cost)).

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
  skipped, blocks by a per-block mask. Elements at least `rfar·hₑ`
  from every particle take the far rule in two kernels launched through
  `TreeAMR.launch_by_owner!`: one over `E·q` points per dimension and
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
refuses a support that reaches a face. The graded and nested rules and
the pairing run on the CPU.

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
  carries nothing between them. The motion is `Prescribed(path)`, with
  `path(t)` the particles' `Trajectory`s at `t`, `isbits`;
  `uniform_motion(z, v)` is one. `Dynamic` is M6a's.
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
  with each node's position from its block's origin, which reaches the
  fold as its tag.

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
arrives and how large it is.

## File layout

- `src/TreeWaveOLDG.jl`: the module and its exports.
- `src/element.jl`: Gauss–Lobatto nodes and weights, `Element`.
- `src/layout.jl`: `check_layout`, node positions, `fill_at_nodes!`,
  `quadrature_norm`.
- `src/evolution.jl`: the kernels, `WaveProblem`, `wave_rhs!`,
  `truncate_state!`, `truncate_stage!`, `initial_state`, `wave_energy`,
  `oldg_timestep`, `convergence_rate`.
- `src/stepping.jl`: `state_partition` and `wave_integrator`.
- `src/sinewave.jl`: the standing sine mode and its driver `wave_errors`.
- `src/blast.jl`: the radial blast wave, its exact solution, and its
  driver `uniform_blast`.
- `src/precision.jl`, `src/device.jl`: `ceilint`, `floorint`, `wrap`,
  `tofloat64` and `to_backend`, taken from TreeWave.
- `src/particle_forms.jl`: generated by `derivations/generate.wl`, not
  edited by hand; the profile's radial functions, the near-field terms
  and their Laplacians, and the self-force coefficients.
- `src/particle.jl`: `ParticleModel`, `Trajectory`, the boost, `Ψ`,
  `𝒟Ψ`, `ParticleJets` and the source bracket, `Window`, and
  `ParticleSource`.
- `src/transfer.jl`: `gauss_nodes_weights`, `GaussRule`,
  `TransferTables`, `sinh_rule`, `ParticleSupport`, the far, graded and
  nested rules, the
  projection's kernels and `project_source!`, `support_elements`,
  `pairing`, `reconstruct`, `reconstruct_gradient` and `ball_rule`.
- `src/inspiral.jl`: `Prescribed`, `uniform_motion`, `InspiralProblem`,
  the state's parts (`field_state`, `particle_state`),
  `inspiral_initial_state`, `inspiral_rhs!`, `inspiral_stage!`,
  `inspiral_partition`, `inspiral_integrator`, `particle_forces` and
  `ball_norm`.
- `derivations/`: the Wolfram Language derivations (`profile.wl`,
  `nearfield.wl`, `selfforce.wl`), each runnable alone with
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
  motion, its error against the exact remainder, its force and, with
  `--cost`, the cost of a right-hand side and its parts and the bytes per
  node, one tab-separated row per quantity, run in the package
  environment; `bin/symmetry/uniform_motion.sh` runs its convergence
  series, `bin/projection.jl` and its clean-time runs on a Symmetry AMD
  node.
- `test/reference.jl`: dense global operators built from the element's
  tables, and the state rearranged into global arrays, which the kernels
  are checked against.

`Project.toml` requires TreeAMR 0.2.6 (for `mesh_fold`) and Julia 1.13,
as TreeWave does, and IMEXRungeKutta 1.4 (for `reuse`), which is not
registered and comes from its repository through `[sources]`, as in
TreeHydro; `test/Project.toml` names it the same way. The particle needs
ForwardDiff 1 (time and space derivatives by dual numbers), StaticArrays
1.9 (`isbits` 3-vectors) and LinearAlgebra. `.github/workflows/CI.yml` is TreeWave's test job:
Julia 1.13 on Linux and macOS with `--check-bounds=yes`, plus a
four-thread cell on Linux, with coverage to Codecov, and a viewer job
that renders both blast-wave figures and uploads them, as TreeWave's
does.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'`, about three and a
half minutes on the laptop, of which the particle, the transfer and the
inspiral take about four fifths; the inspiral's coarse 3D runs about
50 s of it on four threads.

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
  Laplacian; along a motion through rest the bracket is `∇²Φ⁽ᴺ⁺¹⁾` at
  the first order not carried, for orders 0 and 1, order 3 enters
  through `-∇²Φ⁽³⁾`, and a point particle keeps `(a·r)²(a·∇Φ₀)`; the
  point limits are the hyperbolic-motion terms and leave a residual
  falling as `r³`; `𝒟Ψ` and the bracket equal `BigFloat` finite
  differences in `t` and `x`; the `Float64` bracket equals the naive
  difference in `BigFloat` to `1e-12` at `v = 0.2`, and vanishes to
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

`source_bracket` with orders 1–3 carried, against the naive difference
`∂ₜ²Ψ - ∇²Ψ + ρ` evaluated in `BigFloat`: the largest relative error
over twelve points at `σ = 0.3, 0.9, 1.5, 4` in three directions, for
`Q = 0.07`, `ℓ = 10⁻³` and a particle moving perpendicular to its
acceleration, with `|ä| = 4va`.

| `v` | `a` | cancellation-free, `Float64` | naive, `Float64` |
|---|---|---|---|
| 0.2 | 0.08 | `1.7e-14` | `4.0e-5` |
| 0.05 | 0.0025 | `5.0e-13` | `1.0` |

The naive difference loses eleven digits at the first run's speed and
all of them at `v = 0.05`, the speed of `INSPIRAL.md`'s first tests.
`INSPIRAL-PLAN.md`'s estimate of three digits, `log₁₀(v²/aℓ)`, compared
the large terms with `A`, but the bracket is smaller still: `A`'s
order-1 part cancels against `∇²Φ⁽¹⁾`. The cancellation-free bracket
loses about three digits, in that remaining cancellation between terms
of size `Qa/ℓ²`.

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
  `r ≈ 0.5`, which needs `L ≥ 21` and is a gate of `INSPIRAL-PLAN.md`
  ([Open questions](#open-questions)).

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
54, the projection's scratch 27, RK4's scratch 80, about 180 in all, a
little above the 160 that `INSPIRAL-PLAN.md`'s memory table assumes.

- **The source is 20 times the wave equation's work over the whole
  box** of side 6, at every resolution, and over 90% of a step. Over the
  window's elements alone it is about 65 times, and 84 for the binary's
  source ([The transfer's cost](#the-transfers-cost)). A larger box
  dilutes it: in a box of side 24, which the deferred binary run needs,
  it would be about a third of the wave equation's work.
- **Before the near pass was threaded over its points** (job 586061, on
  the same CPU model), the source took 0.118 s at `d/4` and 0.368 s at
  `d/12`: at `d/4` the element that holds the particle was a third of
  the near work and ran on one thread.

## What carried over from TreeWave, and what did not

Carried over: the equation and its variables, the periodic box, the sine
mode and the blast wave with their exact solutions, the 2D viewer of the
blast wave without its refinement panels, `WaveProblem` and the `scatter!` →
`fill_ghosts!` → `map_blocks!` pattern (with `u` copied by a kernel of
its own rather than `scatter!`), the pure right-hand side, the
generic element type, the backend keyword, `ceilint` and `to_backend`,
`convergence_rate`, and the conventions in `CLAUDE.md`.

Changed: the ghost width is 1 whatever the order, and the centering is
always cell; there are `1 + D` exchanges per right-hand side rather than
one, each of one variable; positions, initial data and norms are this
package's rather than TreeAMR's; the time step scales as `hₑ/(P+1)²`
rather than `h`; the integrator is IMEXRungeKutta's rather than
OrdinaryDiffEq's.

Not carried over: everything about refinement (the Löhner indicator,
`regrid!`, the tracking drivers), the travelling pulse, the 1D viewer,
TreeWave's benchmark (this package has its own), the vertex centering,
and the device and threading test files. Löhner's indicator would also fire at every
element face, where the solution jumps; a refinement criterion for this
layout would be modal (the decay of the top Legendre coefficients)
**(proposed)**.

## Possible extensions

- A refined forest, once TreeAMR has [what a refined forest
  needs](#what-a-refined-forest-needs).
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
- The paper's averaging post-processing (its eq. 98–100), which makes
  the interpolated solution continuous for plotting.
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
- **(proposed)** The `t′²` term of the lab-slice re-expansion of `Φ_C`,
  `½γ²(v·r)²(-α′·∇Φ₀)`, in the particle field ahead of M5, which would
  make the bracket at rest the first order not carried at order 3 too
  (see [The near field](#the-near-field)). Its finite-size part, the
  `(ℓ/r)²` terms, would need the full re-expansion.

## Open questions

- Whether interpolated coarse-fine ghosts keep the energy estimate; see
  [What a refined forest needs](#what-a-refined-forest-needs).
- Whether truncation should stay the default for a linear problem,
  where it costs accuracy in `u` and buys no stability (both schemes are
  linearly stable).
- Whether M4's hyperbolic-motion test should expect second order with
  orders 1–3 carried. Until the lab-slice re-expansion, the bracket at
  rest keeps `(a·r)²(a·∇Φ₀)`, which behaves like `r⁰` and so would limit
  the force to first order where `a³` is not small against `ä`, as in
  hyperbolic motion (see [The near field](#the-near-field)).
- Which `ä` the order-3 term should take for a moving particle. M1 takes
  the lab one; the rest-frame proper-time derivative is M5's.
- What to do about the cost of the projection, about 90 times the wave
  equation's work where `INSPIRAL-PLAN.md`'s "Gates and fallbacks" asks
  for ten (see [The transfer's cost](#the-transfers-cost)). It is
  accepted for now and to be revisited, probably with threading or a
  device.
- What to do about the clean time, about `(L/2 - r)/2` where `L/2 - r`
  was expected, since the error of the box's faces outruns light (see
  [Uniform motion](#uniform-motion)). It fires the gate of
  `INSPIRAL-PLAN.md`'s "Gates and fallbacks" for M4's circle, measured at
  `t ≈ 5`, which would need a box of side about 24: 4.5·10⁸ nodes and
  80 GB at `d/8`, and more than a node's memory at `d/12`. The plan's
  fallbacks are a smaller circle radius or `P = 4` at a coarser `hₑ`;
  others are initial data that are periodic, or carry no net charge, so
  that the faces start no error, and an outflow boundary. The owner
  decides before M4.
- Whether the far rule's kernels should balance the window's blocks over
  all threads instead of running each block on its owner: on Symmetry
  they reach about half the speed their source evaluations allow (see
  [The transfer's cost](#the-transfers-cost)).
- The quadrature error on the elements the window's spheres cut, which
  `INSPIRAL.md`'s `10⁻¹²` does not cover (see [Quadrature](#quadrature)):
  whether it matters, and if so a split at `R₁` and `R₂` like the one at
  the support, or a smoother window.

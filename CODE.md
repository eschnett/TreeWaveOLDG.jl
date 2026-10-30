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
suite passes, also with `--check-bounds=yes` and with four threads.

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
- [Truncation](#truncation)
- [Time step](#time-step)
- [The blast wave](#the-blast-wave)
- [The viewer](#the-viewer)
- [Precision](#precision)
- [Threads and devices](#threads-and-devices)
  - [The integrator](#the-integrator)
- [File layout](#file-layout)
- [Tests](#tests)
- [Measured results](#measured-results)
  - [Convergence](#convergence)
  - [The RK4 limit](#the-rk4-limit)
  - [The blast wave's error](#the-blast-waves-error)
  - [Threads on Symmetry](#threads-on-symmetry)
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
- `bin/visualize2d.jl`, `bin/Project.toml`: the viewer and its
  environment; `bin/output/` is gitignored.
- `bin/benchmark.jl`: the cost of a step and its parts, one
  tab-separated row per phase, run in the package environment;
  `bin/symmetry/benchmark.sh` runs it at several thread counts on a
  Symmetry AMD node, with `bin/symmetry/simwatch.sh` reporting its
  stages to SimWatch.
- `test/reference.jl`: dense global operators built from the element's
  tables, and the state rearranged into global arrays, which the kernels
  are checked against.

`Project.toml` requires TreeAMR 0.2.6 (for `mesh_fold`) and Julia 1.13,
as TreeWave does, and IMEXRungeKutta 1.4 (for `reuse`), which is not
registered and comes from its repository through `[sources]`, as in
TreeHydro; `test/Project.toml` names it the same way. `.github/workflows/CI.yml` is TreeWave's test job:
Julia 1.13 on Linux and macOS with `--check-bounds=yes`, plus a
four-thread cell on Linux, with coverage to Codecov, and a viewer job
that renders both blast-wave figures and uploads them, as TreeWave's
does.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'`, about a minute.

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
- **(proposed)** Two changes to IMEXRungeKutta, for the inspiral:
  - Its by-owner path takes a plain `Array` only, and `INSPIRAL-PLAN.md`
    plans the inspiral's state as `ArrayPartition(field, particles)` for
    OrdinaryDiffEq. Either the 12 particle values go at the end of the
    field's state vector, owned by thread 1 in the partition, which
    needs no change there, or IMEXRungeKutta learns to partition an
    `ArrayPartition`.
  - A low-storage (2N) explicit Runge–Kutta method. Its RK4 keeps the
    state and five scratch arrays, OrdinaryDiffEq's about nine; a 2N
    scheme keeps two in all, which matters where memory sets the box
    size, as in `INSPIRAL-PLAN.md`'s table. It would also cut the stage
    arithmetic, a quarter of a 64-thread step ([Threads on
    Symmetry](#threads-on-symmetry)), by about half.
- The paper's averaging post-processing (its eq. 98–100), which makes
  the interpolated solution continuous for plotting.
- A 1D viewer, a device run, and a subprocess thread-count test, as in
  TreeWave.

## Open questions

- Whether interpolated coarse-fine ghosts keep the energy estimate; see
  [What a refined forest needs](#what-a-refined-forest-needs).
- Whether truncation should stay the default for a linear problem,
  where it costs accuracy in `u` and buys no stability (both schemes are
  linearly stable).

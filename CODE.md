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
order `P ≥ 1`, one initial condition, on the CPU. The suite passes, also
with `--check-bounds=yes` and with four threads.

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
- [Precision](#precision)
- [Threads and devices](#threads-and-devices)
- [File layout](#file-layout)
- [Tests](#tests)
- [Measured results](#measured-results)
  - [Convergence](#convergence)
  - [The RK4 limit](#the-rk4-limit)
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
- **Periodic boxes only.** Every driver builds a fully periodic forest
  and passes no boundary hook to `fill_ghosts!`. An outer face would need
  the paper's boundary treatment (its `Ω₀` and `Ω_{K+1}` singleton
  elements), which is not implemented.
- **One initial condition**, the standing sine mode. TreeWave's pulse
  and blast wave existed to exercise refinement, which is out of scope.
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

    scatter!(fs, u); fill_ghosts!(fs)      # u's face neighbours
    gradient pass; fill_ghosts!(grad)      # ∂_d u's face neighbours
    Laplacian pass                         # du, or a scratch array
    truncation pass                        # only with truncation

`WaveProblem` holds the state field set (`u`, `v`), the gradient field
set (`D` variables), the one schedule both share (a schedule belongs to
a layout, and the two have one), the per-block spacings on the kernels'
backend, the scratch array, and the element. As in TreeWave, `u` is
never written and nothing is carried between calls; the tests check
both.

The gradient pass writes into the gradient field set's working array,
from which the exchange fills its ghosts. The window of a node in the
first or last element of a block reaches exactly one ghost plane, which
is what the kernels' `@inbounds` asserts; a run with
`--check-bounds=yes` re-checks it (TreeWave's `CODE.md`, "What bounds
checking costs", has why the annotation is worth having).

## Truncation

The paper holds each conjugate variable one degree below the primary
ones (§II I): at every integrator substep the coefficient of the `P`-th
Legendre mode of `v` is set to zero. Done literally, that writes into
the state between stages, which an off-the-shelf integrator does not
allow and which would break the right-hand side contract.

Here truncation projects `∂ₜv` instead, and `initial_state` projects
the initial `v`. The two are the same scheme: the projector is linear,
every Runge–Kutta stage is the step's starting state plus a combination
of right-hand sides, so if `v` starts truncated and every `∂ₜv` is
truncated, every stage's `v` is truncated too, to roundoff.

The one-dimensional projector is `Π = I - |L_P⟩⟨L_P|/γ` in the
quadrature inner product, where Legendre modes of different degree are
discretely orthogonal. In several dimensions the projector is the tensor
product `Π ⊗ … ⊗ Π`, which removes every mode whose degree is `P` in
*any* dimension, so `v` lives in `Q_{P-1}`. Whether the paper's 3D
implementation removed the same set of modes is **(open)**; its text
describes one dimension.

`truncate=false` gives the plain scheme with both variables at degree
`P`. What each costs is under [Convergence](#convergence). In short,
truncation gives a clean order `P` for `u` and `v` alike. Without it,
`u` is more accurate and often converges faster, but `v` converges
irregularly and more slowly. The default is the paper's scheme.

## Time step

`oldg_timestep(p; cfl)` is `cfl · hₑ/(P+1)²`, the paper's DG step
(eq. 64), which is `cfl · h/(P+1)` in the mesh's spacing. Measured, RK4
is stable up to `cfl ≈ 5` in one dimension for every `P` from 1 to 6,
and the limit falls as `1/√D` (the Laplacian's spectral radius adds over
dimensions); see [The RK4 limit](#the-rk4-limit). The drivers default to
`cfl = 1/2`.

RK4's error is fourth order in `dt ∝ h`, so beyond `P = 4` it eventually
dominates the spatial error. A convergence study at higher order needs a
smaller `cfl` or a higher-order `alg`. At `P = 5` and `cfl = 1/2` the
rates over the measured resolutions are still 5.

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

Every kernel writes one output point per work item and reads only the
state, so the answer does not depend on the thread count. The folds are
TreeAMR's, which are bit-identical across thread counts on the CPU. The
suite passes with `--threads=4`; there is no subprocess test comparing
thread counts bit for bit, as TreeWave has **(open)**.

The element is a tuple of tuples and so travels as a kernel argument.
The spacings and origins follow the field sets' backend (`to_backend`).
Nothing has been run on a device yet **(open)**. The truncation kernel
reads its element through runtime tuple indices, which a GPU may spill
to local memory.

## File layout

- `src/TreeWaveOLDG.jl`: the module and its exports.
- `src/element.jl`: Gauss–Lobatto nodes and weights, `Element`.
- `src/layout.jl`: `check_layout`, node positions, `fill_at_nodes!`,
  `quadrature_norm`.
- `src/evolution.jl`: the kernels, `WaveProblem`, `wave_rhs!`,
  `initial_state`, `wave_energy`, `oldg_timestep`, `convergence_rate`.
- `src/sinewave.jl`: the standing sine mode and its driver `wave_errors`.
- `src/precision.jl`, `src/device.jl`: `ceilint` and `to_backend`, taken
  from TreeWave.
- `test/reference.jl`: dense global operators built from the element's
  tables, and the state rearranged into global arrays, which the kernels
  are checked against.

`Project.toml` requires TreeAMR 0.2.6 (for `mesh_fold`) and Julia 1.13,
as TreeWave does. `.github/workflows/CI.yml` is TreeWave's test job:
Julia 1.13 on Linux and macOS with `--check-bounds=yes`, plus a
four-thread cell on Linux, with coverage to Codecov.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'`, about 50 seconds.

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
  operator in 1D, 2D and 3D, with and without truncation; it neither
  writes `u` nor remembers a call; the energy never grows and its loss
  falls at RK4's rate as the step shrinks; RK4's limit is `cfl ≈ 5/√D`
  for `P = 1, 2, 4`, found by power iteration through `wave_rhs!`.
- `sinewave_tests.jl`: order `P` with truncation for `P = 2, 3, 4` in 1D
  and 2D; without it, `u` is more accurate; `P = 1` is first order;
  `Float32` agrees with `Float64`.

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

## What carried over from TreeWave, and what did not

Carried over: the equation and its variables, the periodic box, the sine
mode and its exact solution, `WaveProblem` and the `scatter!` →
`fill_ghosts!` → `map_blocks!` pattern, the pure right-hand side, the
generic element type, the backend keyword, `ceilint` and `to_backend`,
`convergence_rate`, and the conventions in `CLAUDE.md`.

Changed: the ghost width is 1 whatever the order, and the centering is
always cell; there are two exchanges per right-hand side rather than
one; positions, initial data and norms are this package's rather than
TreeAMR's; the time step scales as `hₑ/(P+1)²` rather than `h`.

Not carried over: everything about refinement (the Löhner indicator,
`regrid!`, the tracking drivers), the pulse and blast-wave cases, the
viewers, the benchmark, the vertex centering, and the device and
threading test files. Löhner's indicator would also fire at every
element face, where the solution jumps; a refinement criterion for this
layout would be modal (the decay of the top Legendre coefficients)
**(proposed)**.

## Possible extensions

- A refined forest, once TreeAMR has [what a refined forest
  needs](#what-a-refined-forest-needs).
- The paper's modal filter (eq. 76), as a term in the right-hand side
  like truncation.
- A comparison with TreeWave's finite differences at equal cost: same
  number of points and same wall clock.
- A kernel per element rather than per node, which would apply the
  tensor-product projector dimension by dimension at `(P+1)` per node
  per dimension rather than `(P+1)^D`.
- The paper's averaging post-processing (its eq. 98–100), which makes
  the interpolated solution continuous for plotting.
- Viewers, a device run, and a subprocess thread-count test, as in
  TreeWave.

## Open questions

- Which modes the paper's multi-dimensional truncation removes; see
  [Truncation](#truncation).
- Whether interpolated coarse-fine ghosts keep the energy estimate; see
  [What a refined forest needs](#what-a-refined-forest-needs).
- Whether truncation should stay the default for a linear problem,
  where it costs accuracy in `u` and buys no stability (both schemes are
  linearly stable).

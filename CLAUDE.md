# Working notes for Claude in TreeWaveOLDG.jl

Read `CODE.md` first: it is the design document and says why things
are the way they are. This file is only about mechanics.

**Contents**

- [Where things go](#where-things-go)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Sharp edges](#sharp-edges)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## Where things go

| file | for | holds |
|---|---|---|
| `README.md` | users | what the package does, installation, an example, the status, a pointer to `CODE.md` |
| `CLAUDE.md` | agents; loaded into every session, so kept short | what to read, commands, sharp edges as short rules pointing at `CODE.md`, repository facts |
| `CODE.md` | anyone changing the code | the current design and its reasons, the tests, current measurements, limitations, extensions, in the present tense, with **(open)** or **(proposed)** on what is not settled |
| `HISTORY.md` | anyone asking how it got here | who decided what and when, alternatives dropped, superseded measurements, by topic in `CODE.md`'s order |
| `INSPIRAL.md` | anyone building the inspiral | the design proposal for two inspiralling scalar particles; frozen, not edited |
| `INSPIRAL-PROMPT.md` | anyone asking what `INSPIRAL.md` answers | the task it was written for; frozen, not edited |
| `INSPIRAL-PLAN.md` | anyone building the inspiral | concrete planned work, milestone by milestone, with how one knows each is done; a status line per milestone, and for a finished one only a line saying where its results went |
| `PLAN.md` | anyone improving the package outside the inspiral | proposed work with what it would save, measured where it says so, and how one knows each is done |

To amend the documents, fix `CODE.md` to describe the current state and
record in `HISTORY.md` what changed, when and why. When a milestone of
`INSPIRAL-PLAN.md` is finished, keep its number and replace its text by
one line saying where its results went, and move the text to
`HISTORY.md` ("What the plan said of its finished milestones"). Where
the inspiral code departs from `INSPIRAL.md` or settles one of its open
items, that goes into `CODE.md` and `HISTORY.md` too, never into
`INSPIRAL.md`. Keep
each file's table of contents current (`##` and `###` in `CODE.md`,
`INSPIRAL-PLAN.md` and `HISTORY.md`, `##` in `README.md`, `PLAN.md` and
here), and
update every reference to a heading you rename, in code comments and
docstrings too.

## What this package is

TreeWave with the finite-difference stencil replaced by the OLDG one
(arXiv:1604.00075), on uniform or static refined forests through
TreeAMR's element centering. TreeAMR is the mesh; anything about trees,
ghosts, coarse-fine operators or interpolation belongs upstream ("Scope
and non-goals").

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The first takes about 6m50 on the laptop, the third about 4m30. The
kernels are `@inbounds`, so run the second one too after touching a
kernel or the layout.

The viewer, in its own environment (CairoMakie is not a package
dependency), about 20 seconds; it writes `bin/output/blast_2d.png`, and
with `--adaptive` the adaptive run's `blast_2d_adaptive.png`, block
faces coloured by level:

```bash
julia --project=bin bin/visualize2d.jl
julia --project=bin bin/visualize2d.jl --adaptive
```

The annulus's viewer, in the same environment, about half a minute on
four threads; it writes `bin/output/annulus_2d.png`, a pulse crossing the
glued faces and the rotating mode's error ("The viewer"):

```bash
julia --project=bin -t 4,0 bin/visualize_annulus.jl
```

The benchmark, in the package environment; on Symmetry,
`sbatch bin/symmetry/benchmark.sh` from a rsynced checkout runs it at
1–64 threads ("Threads on Symmetry" in `CODE.md`):

```bash
julia --project=. -t 4,0 bin/benchmark.jl --scan=4:8
```

The same on the thirteen-patch ball, by default the benchmark ball of
`1.9·10⁷` nodes, about a minute on four threads (5 GB);
`sbatch bin/symmetry/ball_benchmark.sh` after its `setup` runs it at
1–64 threads beside the flat box:

```bash
julia --project=. -t 4,0 bin/ball_benchmark.jl
```

On a GPU ("On a device"): the device tests and both benchmarks take
`TREEWAVEOLDG_TEST_BACKEND=metal` or `--backend=metal --type=f32` (the
laptop's GPU has no `Float64`), in an environment of your own that
`Pkg.develop`s the checkout and adds Metal, TreeAMR, StaticArrays,
ForwardDiff and IMEXRungeKutta, since no device package is a dependency;
about a minute. On Symmetry `sbatch bin/symmetry/gpu.sh` runs them on
one H200 in `Float64` and `Float32`, and `bin/far_rule.jl`, the far rule
of the source projection on the device:

```bash
TREEWAVEOLDG_TEST_BACKEND=metal julia --project=<env> -t 4,0 test/device_tests.jl
julia --project=<env> -t 4,0 bin/ball_benchmark.jl --backend=metal --type=f32 --scan=2:1:4
```

The transfer's cost on the first run's source ("The transfer's cost"),
a few minutes on four threads:

```bash
julia --project=. -t 4,0 bin/projection.jl --hdiv=8
```

A particle in uniform motion against its exact remainder ("Uniform
motion"), in the package environment; its series, clean-time runs and
costs run on Symmetry, `sbatch bin/symmetry/uniform_motion.sh` from a
rsynced checkout after `bash bin/symmetry/uniform_motion.sh setup`:

```bash
julia --project=. -t 4,0 bin/uniform_motion.jl --hdiv=4 --tend=0.125 --cost
```

The same on nested boxes, and outgoing waves through them, the field
alone ("Nested boxes"), in the package environment; on Symmetry
`sbatch bin/symmetry/nested_motion.sh` and `nested_waves.sh`, each after
its `setup`. The smallest runs take under half a minute on four threads:

```bash
julia --project=. -t 4,0 bin/uniform_motion.jl --hdiv=4 --levels=2 --tend=0.0625
julia --project=. -t 4,0 bin/nested_waves.jl --levels=2 --w0=2 --hdiv=4 --lambda=4 --tend=1
```

The same pulse, and its dipole, through the uniform seven-patch ball's
radiative sphere ("The radiative sphere"), in the package environment;
on Symmetry `sbatch bin/symmetry/sphere_waves.sh` after its `setup`, with
the runs in `TREEWAVEOLDG_RUNS`. The run below takes about two and a
half minutes on four threads, half of it `WaveProblem`'s serial setup;
the plan's ball at `R₁ = 50` takes 12 hours on 64:

```bash
julia --project=. -t 4,0 bin/sphere_waves.jl --R1=12 --Lc=4 --n=4 --m=2 --E=4 --lambda=8 --tend=4
```

A particle on a prescribed path, the slow circle or hyperbolic motion
("Prescribed paths"), in the package environment; on Symmetry
`sbatch bin/symmetry/prescribed.sh` after its `setup`, with the runs in
`TREEWAVEOLDG_RUNS`. The smallest runs, on three levels at `hₑ = 1/2`,
take about half a minute on four threads; `--reading` reads the exact
remainder's force without evolving it:

```bash
julia --project=. -t 4,0 bin/prescribed.jl --path=hyperbolic --hdiv=2 --levels=3 --tend=0.25
julia --project=. -t 4,0 bin/prescribed.jl --path=circle --data=retarded --rx1=10 --rx2=14 --hdiv=2 --levels=3 --tend=0.25
julia --project=. -t 4,0 bin/prescribed.jl --path=circle --data=retarded --levels=1 --hdiv=4 --rx1=3 --rx2=3.75 --reading
```

The four-patch annulus ("The annulus"), its convergence series and
energy loss, in the package environment, a few seconds per run on four
threads; on Symmetry `sbatch bin/symmetry/annulus.sh` after its `setup`,
from a rsynced checkout:

```bash
julia --project=. -t 4,0 bin/annulus.jl --P=3 --scales=1,2 --truncate=yes
```

The seven-patch ball ("The seven-patch ball"), the plane wave's
convergence series, in the package environment; on four threads a few
seconds at `--scales=1,2` for `P = 2`, half a minute for `P = 3`, and
`s = 3` takes minutes:

```bash
julia --project=. -t 4,0 bin/sevenpatch.jl --P=2,3 --scales=1,2
```

With `--boundary=radiative` it evolves an outgoing pulse instead, through
a radiative outer sphere ("The outer boundary"); by `--tend=5` the pulse
has left the ball, and what remains is the reflection:

```bash
julia --project=. -t 4,0 bin/sevenpatch.jl --P=3 --scales=1,2 --boundary=radiative --tend=5
```

With `--Rm=2` both run on the thirteen-patch ball instead ("The
thirteen-patch ball"), the seven-patch ball out to `|x| = 2` and six
exactly radial shells from there to `R1`, of `(2s, s·m1, s·m2)` roots;
about as long as the seven-patch runs:

```bash
julia --project=. -t 4,0 bin/sevenpatch.jl --P=2,3 --scales=1,2 --Rm=2
```

`nested_waves.jl`'s pulse through nested cubes inside the ball's cube,
with shells out to a radiative sphere ("A refined cube"), in the package
environment, a few minutes at these options on four threads, most of it
the shells' truncation tables; `--cost=yes` times it beside the nested
boxes. On Symmetry `sbatch bin/symmetry/nested_ball.sh` after its
`setup`:

```bash
julia --project=. -t 4,0 bin/nested_ball.jl --levels=2 --w0=2 --hdiv=4 --lambda=4 --tend=1
```

From M1 of `INSPIRAL-PLAN.md` on, `src/particle_forms.jl` is generated
from the Wolfram Language derivations in `derivations/`; do not edit it
by hand. Regenerate it, and let the suite check that the checked-in
file matches a fresh one (skipped when the variable is unset, as on
CI):

```bash
export TREEWAVEOLDG_WOLFRAMSCRIPT=/Applications/Wolfram.app/Contents/MacOS/wolframscript
$TREEWAVEOLDG_WOLFRAMSCRIPT -file derivations/generate.wl
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Sharp edges

Each is the rule; the `CODE.md` section named after it has the reason.

- **The particles need the finest level under their source's ball.**
  `WaveProblem` accepts any forest TreeAMR's element centering does, and
  `InspiralProblem`, `project_source!` and the pairing refuse a source's
  ball or a particle's support that reaches a block coarser than the
  finest, because the transfer measures its distances in the finest
  element width. Do not lift that refusal without making the transfer
  level-aware ("Nested boxes").
- **The refinement indicator reads `u`'s mode `P - 1` under truncation,
  `P` without it,** since truncation freezes `u`'s top mode in time; and
  it reads every mode below that too, weighted, because one mode alone
  can pass an under-resolved feature as resolved. Do not drop the lower
  modes or the weights ("Adaptive refinement").
- **After `regrid!`, rebuild the problem with `WaveProblem(fs, el)`,
  `gather!`, then `truncate_state!`.** The other sets and schedules
  belong to the old layout, and a coarsening brings `v`'s top mode back
  ("The driver").
- **Keep `coarsen_tol` about `refine_tol/1000`, and `refine_tol` low
  enough that the cap binds on the initial data.** Refining data already
  evolved on a coarser level costs accuracy that nothing recovers
  ("The verdict").
- **Every field set is element-centered at the element's order, with
  `Operators(family = ElementLocal)`.** The coarse ghost is the energy-
  compatible restriction, exact to degree `P - 1` tangentially; do not
  swap in another operator here: the coupling is TreeAMR's, and any
  other one breaks the energy invariant ("The coupling", "The energy").
  An element set's edge and corner ghosts are never written.
- **Positions, initial data and norms are TreeAMR's, at the nodes.**
  Use `node_coordinates`, `fill_at_nodes!` and `quadrature_norm`, which
  are TreeAMR's `coordinates`, `fill_by_coordinates!` and
  `volume_weighted_norm` for element sets and refuse any other set
  ("Positions and quadrature").
- **`N = E(P+1)`, and `E` even**, so that a fine block covers whole
  elements of its parent ("The layout").
- **Truncation sets `v`'s top mode to zero in the state, through the
  integrator's limiter hooks, never in the right-hand side.** An
  integrator must be given `truncate_stage!` as its stage and step
  limiter, which `wave_integrator` does; `initial_state` truncates the
  initial `v` ("Truncation").
- **Step with `wave_integrator`, IMEXRungeKutta by block owner, not
  OrdinaryDiffEq**, whose serial stage arithmetic and fifth right-hand
  side per step cost more than the right-hand side on many threads
  ("The integrator").
- **Each `∂_d u` set has ghosts along `d` only, and `v` none.** A kernel
  that reads `∂_d u` across another dimension's face, or `v` across any,
  reads past its array; `--check-bounds=yes` catches it ("The
  right-hand side"). On a full geometry the gradient is one set of `D`
  Cartesian components with ghosts on every face ("The seven-patch
  ball").
- **Pass `d` to the gradient and Laplacian kernels as a `Val`, through
  an `@inline` closure**; a runtime `d` costs 2.5× ("The right-hand
  side").
- **The tables are built in `BigFloat` and rounded once,** except a full
  geometry's truncation tables, per-element and factored, built in
  `Float64x2` when `T`
  is at most `Float64` (`table_type`), which a test holds to `BigFloat`'s
  bit for bit. Do not build them in `T` or convert from the `Float64`
  ones ("Precision").
- **The time step is `cfl · hₑ/(P+1)²`**, not TreeWave's `cfl · h`
  ("Time step").
- **A diagonal geometry glues by the identity only.** The `d`-th
  gradient set takes its ghosts across a glue from the same set of the
  patch across. A glue that swaps or reverses axes, or an off-diagonal
  metric, needs a `FullGeometry` ("Multi-patch forests").
- **On a diagonal geometry the gradient sets hold the flux `A_d D_d u`,
  not `∂_d u`, and the truncation is `√g`-orthogonal,** one rank-one pair
  per line from the factors of `√g`, which must be separable. The
  element's projector there would break the energy invariant ("Truncation
  with a metric").
- **On a full geometry the gradient is Cartesian and the flux is formed
  where it is read.** `cartgrad` holds `∇u` in Cartesian components,
  which a glue carries unchanged; the Laplacian forms `F^d = S[d] ⋅ ∇u`
  with the frame of the owned face node beside a ghost, never a metric
  at the ghost. The truncation is a `√g`-orthogonal projection per
  element, and the outer faces take Dirichlet data through `boundary`,
  for which `wave_rhs!` passes `t`, or a `RadiativeBoundary` ("The
  seven-patch ball"). A block whose frame is one constant diagonal (the
  cube's) takes the kernels' flat path, which must stay bit for bit the
  general path's result; a test sends the blocks through both.
- **A factored truncation needs a volume element that factorizes.** A
  full geometry that declares `factored_volume(geo, p) = k` gets the
  radial pair times the angular projection on patch `p`, which is
  `√g`-orthogonal only if `radial_volume · angular_volume` is
  `geometry_frame`'s `√g`; patches of one `angular_class` share angular
  tables, so their angular factors must be one function. Declaring
  either where it does not hold breaks the energy invariant ("The
  thirteen-patch ball").
- **A radiative outer face is written after the exchange, not by a
  TreeAMR hook.** Its ghosts read the interior: `u`'s is the face node's
  own, and the gradient's makes the face's central flux the target,
  `S[k] ⋅ V_g = 2F* − F`. Both the energy's boundary term and its sign
  rest on the two together; do not set `u`'s ghost from data ("The
  outer boundary").
- **On a geometry the time step is the smallest over the blocks of the
  block's spacing over its largest logical wave speed**, `√g^{dd}`, or
  `√λ_max(g^{de})` on a full geometry, which `oldg_timestep` does ("The
  metric", "The seven-patch ball").
- **A geometry takes refinement only inside a patch whose map is
  affine** (`affine_patch`), the seven-patch ball's cube: elsewhere the
  coarse-fine coupling's flat weights are not the `√g`-weighted ones and
  the energy invariant breaks. Keep the refined regions a root inside the
  cube, since the glue keeps the shells at the level across it; there
  the truncation is the element's own ("The seven-patch ball").
- **Everything a kernel reads is on the problem's backend, and isbits
  there.** Build tables and coefficients on the host and copy them
  (`to_backend`, `coefficient_set`); a struct of arrays a kernel takes
  whole needs an `Adapt` rule; a callback captures no `Type` and no
  global table, and no kernel writes a `Float64` literal, or Metal's
  `Float32` breaks ("On a device"). Run `test/device_tests.jl` on Metal
  after touching a kernel.
- **The `u` and `v` of a state vector interleave block by block**; use
  `statearray` and `selectdim`, not halves of the vector.
- **A reflecting box is a TreeAMR reflection symmetry, not a boundary
  hook.** Ask for it with `wave_forest(…; reflecting=true)`; the forest
  then has reflecting faces, and `WaveProblem` derives the parities from
  them into the field sets' ghosts (`u`, `v` even, `∂_d u` odd in
  `d`). Do not build the faces by hand and forget the parity, and do not
  try to add a boundary hook (see "Reflecting walls").
- **The source bracket is never formed by subtracting the large terms.**
  `ρ` cancels against `□_vΦ_C` analytically, the acceleration's parts
  of the invariants' rates are written out term by term, never as `İ`
  minus its uniform part, and near-field Laplacians are generated closed
  forms ("The split").
- **The particle field carries the lab slice of each order it carries**
  (`lab_slice(m, Val(N), …)`, generated from `derivations/labslice.wl`),
  and the near field takes the rest frame's `C` for `ä`
  (`rest_frame_derivatives`). It is the field of the lab-sliced
  density, not `INSPIRAL.md`'s re-expansion of the rest-frame near
  field. Do not drop it from `Ψ`, `𝒟Ψ` or the bracket: without it the
  bracket keeps `O(a)` inside the particle, and the remainder an `r⁰`
  structure whose force the grid reads the more strongly the finer it
  is ("The near field").
- **The generated lab slice sums with `+=`, is not inlined, writes
  Horner's rule out and nests its products, and the bracket reads its
  coefficients in place** (`_coefficient`). One long `a + b + …`
  allocates, inlining makes a `BigFloat` Hessian take minutes to compile,
  `evalpoly` and `a * b * c * d` on the nested duals are calls that take
  their arguments through memory, and wrapping the tables per point
  copies kilobytes ("The near field").
- **The source is written and the forces read through the averaging
  over shared nodes** (`InspiralProblem`'s `continuous`, the default):
  the transfer alone reads the jump of the field across a face the
  support straddles, times `1/ℓ`, as impulses far above radiation
  reaction. Keep the write and the read together, and the support's
  elements' neighbours on the finest level ("The averaging").
- **A particle's support is a breakpoint.** Integrate `ρ`, `∇ρ` or the
  source near a particle with the nested rule, never a Gauss or sinh rule
  across the support's edge, where the profile is only `C³`
  ("Quadrature").
- **The transfer looks for no periodic images.** The source's ball and
  every particle's support lie inside the box ("The transfer").
- **The inspiral's state is the field's state vector followed by six
  numbers per particle.** Use `field_state` and `particle_state`, and
  `inspiral_integrator`, whose partition gives the particles to thread 1;
  not an `ArrayPartition` ("The right-hand side with particles").
- **Error norms of an inspiral run are taken in the clean ball,** with
  `ball_norm`: the field of a charge is not periodic, and the box's
  faces' error, which travels at about twice the speed of light, spoils
  a point at `r` from the centre from `t ≈ (L/2 - r)/2` on ("Uniform
  motion"); on nested boxes it crosses the coarse levels at about 1.5
  times, from `t ≈ (L/2 - r)/1.5` on ("Nested boxes").
- **Sample a particle's force every step when looking for anything
  smaller than an impulse.** A face crossing lasts `2ℓ` over the speed
  across the face, a few steps, and its impulse is tens to hundreds of
  times radiation reaction; a sample every `1/8` misses it, as M3's did
  ("Prescribed paths").
- **The asserted rates are measurements.** If one moves, re-measure and
  update "Measured results" rather than widen a tolerance.

## Conventions

As in TreeWave and TreeAMR: 4-space indent, wrap at about 80 columns,
`return` on the last line of a non-trivial function, Unicode in
mathematical contexts, prose-first docstrings, testset names that are
claims, each testset opening with a comment naming the failure it
guards.

## Repository facts

- `origin` is `https://github.com/eschnett/TreeWaveOLDG.jl.git`, and
  `main` tracks it. Work on a branch; do not push, open a pull request,
  or merge to `main` without being asked.
- Not registered. TreeAMR 0.2.8 (with M17's forests of patches) comes
  from the General registry, not from a local checkout; IMEXRungeKutta,
  which is not registered, from its GitHub repository through
  `[sources]` in `Project.toml` and `test/Project.toml`.
- CI (`.github/workflows/CI.yml`) runs the suite with
  `--check-bounds=yes` on Linux at one and four threads, and on `main`,
  weekly and on dispatch also on macOS, with the coverage; Markdown-only
  changes and tags run nothing. So run the full suite locally before
  opening a pull request: that is the macOS run the pull request does not
  get. Do not drop its `check_bounds: 'yes'` line ("The right-hand side").
- `Manifest.toml` (root and `bin/`), `bin/output/` and `TODO.md` are
  gitignored. `bin/` has its own Manifest, so after a dependency change
  update both, or the viewer fails on an API the tests already use. `TODO.md` is the owner's
  own list: do not edit it, and move nothing into or out of it.

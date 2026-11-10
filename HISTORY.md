# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

M0 to M8, with M3b, M6a, M6b, M8a, M8b and M8c, are the milestones of
`INSPIRAL-PLAN.md`: M0 the documents, M1 the analytic particle, M2 the
transfer operator, M3 a particle at rest and in uniform motion, M3b
nested boxes, M4 prescribed paths, M5 the lab slice, M6a one particle
solved, M6b the binary's parts, M7 the binary's warm-up and orbits, M8
the particles in the seven-patch ball (proposed). M16
and M17 are TreeAMR's: its element centering and its forests of
patches.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [The right-hand side](#the-right-hand-side)
- [Reflecting walls](#reflecting-walls)
- [Multi-patch forests](#multi-patch-forests)
- [Refined forests](#refined-forests)
- [Adaptive refinement](#adaptive-refinement)
- [Truncation](#truncation)
- [The blast wave and the viewer](#the-blast-wave-and-the-viewer)
- [Threads and the integrator](#threads-and-the-integrator)
- [Particles](#particles)
  - [What the plan said of its finished milestones](#what-the-plan-said-of-its-finished-milestones)
- [Dependencies](#dependencies)

## Scope and non-goals

- 2026-10-26: package created by Erik Schnetter, to implement the scalar
  wave equation like TreeWave but with the OLDG stencils of
  arXiv:1604.00075. The first version was decided to be the unrefined
  prototype that TreeAMR's `CODE.md` proposes ("Element-local DG
  stencils"), to judge the method on a uniform forest, since a refined forest needs a Gauss–Lobatto centering,
  element-local transfer operators and a schedule-builder change in
  TreeAMR first.
- 2026-10-26: TreeWave's refinement criterion, regridding, pulse and
  blast-wave cases, viewers, benchmark and device tests were not carried
  over; all but the viewers and device tests exist to exercise
  refinement.
- 2026-10-30: Erik Schnetter adopted `INSPIRAL.md`, a design for two
  inspiralling scalar particles of fixed size `ℓ ≪ h` on this scheme
  (after arXiv:2606.06487, answering `INSPIRAL-PROMPT.md`), and
  `INSPIRAL-PLAN.md`, which implements it in milestones with one pull
  request each. `INSPIRAL.md` stays frozen; what the code settles goes
  into `CODE.md` and here. Decided with it: Wolfram Language for the
  symbolic work, with its closed forms generated into a checked-in Julia
  file; the particle state joined to the field by `ArrayPartition`; long
  runs on Symmetry's AMD nodes. No sponge or outflow boundary is added:
  runs stay in a periodic box that is causally clean for their duration.
  Since a uniform box that stays clean for even one orbit does not fit
  one node, the binary's warm-up and orbits are deferred until mesh
  refinement or a better outer boundary (multi-patch, or an outflow
  boundary) is available.

## The method

- 2026-10-26: the element tables were checked against the paper's Figs.
  5–7 (`P = 2, 3, 4`), and a 1D implementation using dense global
  matrices was written first as a reference. The package reproduces it
  to the printed digits for `P = 1 … 6`, and the dense operators became
  `test/reference.jl`.

## The right-hand side

- 2026-10-30: the exchange was cut to what the wide derivative reads,
  asked for by Erik Schnetter after an investigation of the cost of a
  threaded step. Before, `u` and `v` were scattered into one field set
  with `G = 1` and the gradient was one `D`-variable set with `G = 1`,
  sharing one schedule, two exchanges that filled every face, edge and
  corner of all five variables in 3D, `5 · 1736` points per block at
  `N = 16`. They took half a right-hand side on the laptop (33 of 65 ms;
  3D, `P = 3`, 512 blocks of `16³`, one thread). Now
  `v` is read from the state and never exchanged, `u` has a set of its
  own, and each `∂_d u` a set with ghosts along `d` only, `1 + D`
  exchanges of about 2.7 times fewer points, which took 7–12 ms of a
  26 ms right-hand side (the laptop was not idle, and runs differ that
  much), measured with cell-centered sets, when `u`'s exchange also
  filled its edges and corners, `1736` points. The state set keeps
  `G = 0`. TreeAMR refuses a ghost-free dimension under point-value
  operators, so the schedules use conservative order 1, a formality on
  a uniform forest.
- 2026-10-30: written with a runtime loop over `d`, the new kernels,
  which index a tuple of `D` arrays, were 3.5 times slower than the
  one-set kernels; through a closure over `ntuple`, six times. Passing
  `d` as a `Val` through an `@inline` closure made both faster than the
  old ones (gradient pass 6.9 ms against 13.6, laptop, one thread). One
  launch per dimension was as fast for the gradient but cannot serve
  the Laplacian, which sums over `d`.

## Reflecting walls

- 2026-10-27: reflecting (solid-wall, Neumann `∂ₙu = 0`) boundaries
  were added by way of TreeAMR's reflection symmetries: a `reflecting`
  keyword on `wave_forest` marks every face reflecting, and `WaveProblem`
  derives the per-variable parities the wall imposes (`u`, `v` even;
  `∂_d u` odd in `d`) for the field sets' ghosts, so the wall needs no
  boundary hook. A standing `cos` mode, `reflect_exact`, exercises it and
  converges at order `P` with energy conserved to the integrator's own loss.
  Global (rotating-seam, partly reflecting or outflow) boundaries remain a
  non-goal.

## Multi-patch forests

- 2026-11-07: the four-patch annulus, TreeAMR's acceptance test for its
  M17 (forests of patches glued at their faces, uniform), whose
  `PLAN.md` names TreeWaveOLDG's scalar wave on the annulus, converging
  with its energy conserved across the glued faces, as what finishes
  it. Erik Schnetter decided the shape the same day:
  - **A real polar annulus, not the flat equation on the ring.** The flat
    one would only have been a periodic strip in logical coordinates.
    So the package gained a metric: `Geometry`, `PolarAnnulus`, the flux
    form `√g v̇ = Σ_d D_d(A_d D_d u)`, and the `√g`-weighted energy, norm
    and time step. TreeAMR leaves the map to the application.
  - **The metric in `src/`**, not in the test, so that the annulus is
    something the package does, and the spherical shells
    `INSPIRAL-PLAN.md` set aside as "a multi-patch outer region" have a
    place to go.
  - **The rotating Neumann–Neumann Bessel mode** as the exact solution,
    rather than self-convergence. It is rotating rather than standing,
    so that flux crosses every glued face.
  - **TreeAMR from a checkout of its branch beside this one**, through
    `[sources]` path entries, until it was pushed. TreeAMR 0.2.8, with
    M17, was registered the same day and replaced them before the work
    was committed (see "Dependencies").
  - **The truncation orthogonal in the `√g`-weighted inner product.**
    The alternatives were to refuse truncation on a geometry, or to keep
    the element's projector and let the energy drift by the truncation's
    error. The element's projector is self-adjoint in the flat weights
    only, which is what the energy invariant needs ("Truncation" in
    `CODE.md`). Because `√g` factorizes on the annulus, the projector
    stays a rank-one update per line, with a pair of vectors per block,
    dimension and element index along it, built in `BigFloat`.
- 2026-11-07: the gradient sets hold the flux `A_d D_d u` rather than
  `D_d u`, multiplied in the gradient pass. That way the ghost across a
  face is the neighbour's own flux, and the energy's rate is
  `Σ_d Σ w (v D_d F_d + F_d D_d v)`, zero by summation by parts as on
  the flat box. The metric's coefficients come from a coefficient set
  filled once at the nodes, and the energy and the norms evaluate the
  geometry inside their folds, with the patch as each block's tag.
  Without a geometry the kernels compile to what they were.
- 2026-11-07: measured on Symmetry (job 593191, cn095, EPYC 7543):
  with truncation the rotating mode converges at order `P` for
  `P = 2 … 5` (L2 rates 1.93, 3.01, 3.82, 4.88), and the energy's loss
  falls 32 to 36 times per halving of the step, RK4's own. Each of
  three deliberate breaks fails the testset written for it: dropping
  the division by `√g`, the flux's `A_d`, or the `√g` weight of the
  projector. See "The annulus" under "Measured
  results" in `CODE.md`.
- 2026-11-07: `bin/visualize_annulus.jl` added, on Erik Schnetter's
  request for a 2D showcase of the annulus, and rendered by CI beside
  the blast wave. Its parameters were chosen to show the glue rather
  than to converge (`P = 4`, `(4, 2)` roots per patch, a pulse of
  `σ = 0.12` off-centre in patch 1, the mode `m = 3`, which does not
  share the ring's four-fold symmetry). Two things it showed on the way:
  - The first error map was ten times the nodal error everywhere,
    because the exact mode was taken at a quarter turn while the
    snapshot was at the nearest step, half a step away. The exact mode
    is now taken at the snapshot's own time.
  - With truncation the mode's error swells and shrinks with the period
    `2π/k`, to ten times its level without truncation: the frozen top
    mode of `u` ("Truncation" in `CODE.md`). The pulse's energy loss,
    `5.9e-7`, falls 32 times per halving of the step, so it is RK4's
    and not the glue's.
- 2026-11-08: the seven-patch ball, a cube and six shells, on Erik
  Schnetter's request for a seven-patch cubed sphere. TreeAMR needed no
  change. A scratch probe against TreeAMR 0.2.8 showed that its forest
  builds (42 patch faces), that an element set with `G = 1` fills
  exactly across every glue, and that two things are refused:
  - the gradient sets, `G = e_d`, because the shell–shell glues swap
    axes, which no labelling of a cubed sphere's axes avoids;
  - a reflecting or outflow outer sphere, because TreeAMR's face kinds
    are per `(dim, side)` and the cube is glued on all six faces.

  Three ways to carry the gradient across the swapped glues were weighed:
  1. **Cartesian components** in one set with ghosts on every face, the
     flux formed with the reading block's own frame. No TreeAMR change.
  2. **A variable map per glue** in TreeAMR, which would permute and
     sign one set's logical components. The same traffic as the first,
     saving only the contraction at the ghost.
  3. **A family of `D` sets** in TreeAMR, generalising its
     `RotationPair`, which would keep a set per dimension and its
     traffic. This is the largest change.

  Erik chose the first, with Dirichlet data on the outer sphere for now,
  and a per-element `√g`-orthogonal truncation, so that the energy stays
  invariant though the shells' `√g` does not factorize. The first costs
  twice the ghost values per right-hand side in half the exchanges, a
  trade not yet measured against the third. The design is "The
  seven-patch ball" in `CODE.md`. The annulus keeps its path, bit for
  bit.
- 2026-11-08: the per-element truncation was first built as
  `Λ_e = (Ψᵀ W_e Ψ)⁻¹ Ψᵀ W_e`, a dense `BigFloat` product and solve of
  about 25 ms per element: 9.4 s for the 56 blocks of `P = 3` with
  `(2, 2)` roots, and minutes at `(6, 6)`. Building the Gram matrix by sum
  factorization and keeping its Cholesky factor, solved in the kernel,
  took it to 1.5 s and halved the tables.
- 2026-11-08: a radiative outer sphere, on Erik Schnetter's request
  for maximally dissipative boundary conditions. He asked whether a
  13-patch ball, whose outer shell has exactly radial coordinate planes
  and spherical `r = const` surfaces, would be needed. It was not: the
  seven-patch map `x = f·[(1 − ρ) + ρR1/|f|]` already has radial lines
  as coordinate lines and the sphere as its outer face, so `S[k]` is
  radial there, and the energy argument reads nothing else. A 13-patch
  ball would still help accuracy (no shear near the sphere, the same
  radial spacing at every angle, a cheap larger `R1`), and would be the
  place for a higher-order absorbing condition.

  `CODE.md` had said an outgoing condition needed a TreeAMR hook that
  reads the interior, or per-patch face kinds. Neither was needed: the
  two exchanges run without a hook, and two kernels of this package
  rewrite the outer ghosts afterwards, `u`'s as the face node's own and
  the gradient's so that the face's central flux is
  `−σ |S| (κ v + u/R)`. The discrete energy rate then equals that flux's
  boundary term to rounding, which the test asserts.
- 2026-11-08: measured on the laptop at `P = 3`: an outgoing pulse of
  width 1/2 leaves 3.5e-3 of its energy behind at `(2, 2)` roots and
  1.8e-6 at `(4, 4)` under Sommerfeld's condition, and 6e-3 at both
  without its `u/R` term; a pulse of width 1 converges at order 2.9
  through the sphere. See "The seven-patch ball" under "Measured
  results" in `CODE.md`.
- 2026-11-09: `INSPIRAL-PLAN.md`'s M8b, refinement inside the seven-patch
  ball's cube, at Erik Schnetter's request, while M8a ran in another
  session. Until then every geometry refused a refined forest, because
  the coarse-fine coupling is energy-compatible in the flat weights. The
  refusal now names the patches it applies to: a geometry says which of
  its patches are affine (`affine_patch`), and refinement is refused
  only in the others. On the cube `√g` and the frame are constants, so
  the flat argument holds unchanged; the right-hand side on a refined
  brick under the identity map is the flat one's bit for bit, which is
  the test that TreeAMR treats the Cartesian gradient's components as it
  treats the flat path's gradient sets. Three changes came with it:
  - **The cube's truncation is the element's.** `ElementTruncation`
    holds its tables for the shells' blocks only, and its build runs on
    all threads; serial, it took about a minute for 192 shell blocks of
    `16³`, and would have taken most of an hour at M3b's scale.
  - **The time step is per block,** the smallest of each block's
    spacing over its own largest speed, where it had been the finest
    spacing over the fastest speed anywhere, which on a refined ball lie
    in different blocks (the owner chose this over the plan's
    "stays conservative"). On the test's ball it is 1.8 times larger.
  - **The shells beside the cube's corners limit the step** more than
    the plan's estimate assumed: their elements are thinner and sheared
    there, and their own step is about 0.3 of the cube's root level's at
    `R1 = 3Lc`, so the cube needs about three levels before its finest
    sets the step.
- 2026-11-09: `PLAN.md` started, at Erik Schnetter's request, with the
  savings measured after M8b: a flat path for the cube's blocks and a
  cheaper full-geometry Laplacian (the full geometry's right-hand side
  costs 7.7 times the flat one's, nearly all of it in the two kernels),
  a faster build of the shells' truncation tables (`Float64x2` gives
  `BigFloat`'s rounded tables bit for bit at a sixth of the time, and
  transposing one shell's to the other five saves two thirds more), and
  smaller tables and coefficients. A layer of 26 barrel-distorted
  patches around a Cartesian cube was considered for the shells' time
  step and set aside: with three or more levels in the cube the cube's
  finest level sets the step.
- 2026-11-09: the cube's flat path, the first of `PLAN.md`'s items, at
  Erik Schnetter's request. A block whose frame is one constant diagonal
  skips the frame's zeros in both Cartesian kernels, bit for bit the
  general path's result. Before it, on the same nested-box forest, the
  full geometry with the identity map cost 7.7 times the flat path per
  right-hand side (11.42 ms against 1.58 ms; gradient kernel 2.59 ms,
  Laplacian kernel 6.62 ms), and the local ball 23.7 ns per node. After
  it, 1.9 times (2.92 ms) and 13.6 ns per node. The plan had expected
  about 1.1 times from the breakdown; the Laplacian kernel stays at 2.7
  times the flat one for the same arithmetic, and the gradient's exchange
  at 2.5 times, which `PLAN.md` keeps as an item. The suite's seven-patch
  tests went from about 106 s to 53 s with this and the threaded table
  build.
- 2026-11-10: the shells' truncation tables built in `Float64x2`, the
  second of `PLAN.md`'s items, at Erik Schnetter's request. The rule had
  been that every table is built in `BigFloat` and rounded once; it now
  has one exception, the full geometry's per-element tables, built in
  MultiFloats' `Float64x2` when `T` is at most `Float64`, which a test
  holds to `BigFloat`'s rounded tables bit for bit. MultiFloats became a
  dependency. On the laptop the build is 7.5 times faster. At M3b's
  scale on 64 threads the `BigFloat` build had taken 1051 s, 884 s of it
  in the garbage collector, where the plan had estimated 23 s from the
  arithmetic; in `Float64x2` it takes 93 s, still mostly collection,
  which `PLAN.md`'s non-allocating loops would address. With M8a merged
  in below this, its "the setup is serial" and its proposed threaded
  build in `CODE.md` became "was serial" and were dropped: M8b builds
  the tables by block on all threads and only for the shells.

## Refined forests

- 2026-10-30: decided with Erik Schnetter, after the prototype had run
  on uniform forests: the mesh machinery a refined forest needs goes
  upstream as an **element centering** in TreeAMR (its milestone M16),
  with `P` a property of the field set, rather than an element-local
  operator family on cell-centered sets, which would have left
  positions and norms at cell centres; the coarse side's ghost is the
  **energy-compatible** restriction `R = M_c⁻¹ Iᵀ M_f`, the adjoint of
  the fine ghost in the quadrature, so that the discrete energy stays an
  exact invariant across coarse-fine faces, at the price of one degree
  tangentially, rather than an interpolation or the exact `L²`
  projection, which keep degree `P` and break the energy argument;
  refinement comes **static first, then adaptive** (a modal indicator, a
  regrid driver and an adaptive blast wave); and the particle code
  **keeps refusing refined forests**. A regrid coarsens by the exact
  projection, since energy is not conserved across a regrid anyway.
  Whether the degree loss costs accuracy in `D ≥ 2` was left to be
  measured here.
- 2026-10-31: built on TreeAMR 0.2.7 (eschnett/TreeAMR.jl#32): every
  field set element-centered with `Operators(family = ElementLocal)`;
  `node_coordinates` and `fill_at_nodes!` became TreeAMR's
  `coordinates` and `fill_by_coordinates!`, and `ball_norm` reads the
  node position from TreeAMR's fold, so the package's own position code
  went; `check_layout` dropped its refusal of a refined forest and
  requires `E` even, which is stricter than before (`P = 3`, `N = 12`
  had been legal), so the tests with one element per block now use two;
  `wave_forest(…; refined)` makes a static two-level forest as TreeWave
  does; `InspiralProblem` and `project_source!` refuse a refined forest.
  The energy question the prototype had left **(open)** is answered:
  the energy is an exact invariant, the skew-adjointness residual about
  `1e-18` relative to its terms in 1D to 3D.
- 2026-10-31: a first smoke run looked wrong: in 2D at `P = 3` the
  two-level forest's L2 error over the box was barely below the uniform
  coarse run's (`0.0096` against `0.0111`, fine `0.0013`), and in 1D at
  `P = 2` its L∞ error exceeded it (`0.310` against `0.275`). Measured
  out, the first is norm locality (the coarse three quarters of the box
  set the norm, `√(3/4)` of the coarse error), and inside the refined
  region the error lies between the uniform runs'. The second is not a
  degree loss, which 1D, where every transfer is injection, cannot have:
  the coarse part's error is a fixed `1.11–1.15` times the uniform coarse
  run's at every resolution, what the coarse-fine faces scatter back into
  it, converging at order `P` with the rest.
- 2026-10-31: measured, with the tables in `CODE.md` ("Refined forests"
  under "Measured results", and "The RK4 limit"): the L2 error over the
  box converges at order `P` for `P = 2, 3, 4` in 1D and 2D; RK4's limit
  at the finest spacing is the uniform one, within 2% in 1D and 2D and
  7% in 3D; the energy is lost to RK4 only, falling by `2^6` per halving
  of the step; a reflecting box refined at its walls keeps order `P` and
  its energy. In 2D the L∞ error falls short of order `P` (`2.3–2.7` at
  `P = 3`), with its maximum at fine nodes in the refined region, outside
  the coarse–fine bracket from `roots = 16` on, and so does the L2 error
  at a full period at `P = 3` (`2.55` at the finest pair, 45 times the
  uniform coarse run's, which converges faster than order `P` there). A
  refined stripe without corners shows the same. To isolate the cause,
  a scratch copy of TreeAMR with the exact `L²` projection in `R`'s place
  (not kept) ran the same cases: order `P` in L∞ (`3.06` at `P = 3`), and
  energy no longer conserved, gaining up to `3e-5` of it in a quarter
  period. So the coarse ghost's degree loss is the cause. The scheme was
  left as decided: Erik decided on 2026-10-31 to keep `R` for the
  invariant, and the item is **(open)** in `CODE.md`: a
  degree-preserving compatible operator through a finer mortar would be
  upstream, and the owner's decision.

- 2026-11-01: `wave_forest(…; refined)` also takes a vector of regions,
  one per level above the root level, each refining only the leaves of
  the level below whose centres lie in it, with the forest balanced
  after each, and `nested_cubes` builds the nested boxes of
  `INSPIRAL-PLAN.md`'s M3b from `w₀` and the number of levels.
  `refined = true` and a single region keep their two-level meaning (a
  test). See "Particles" below.

## Adaptive refinement

- 2026-10-31: built as decided above, on TreeAMR's `regrid!` of element
  sets: `WaveProblem(fs, el)` rebuilds a problem from a regridded state
  set, `modal_indicator` and `refine_flags` in `src/refinement.jl`, and
  `adaptive_blast`, TreeWave's `track_blast` cycle in chunks. The plan
  had the indicator read `u`'s top Legendre mode. With truncation that
  mode is constant in time, since `∂ₜu = v` and `v` has none: on the
  blast wave a level-1 block the ring had crossed held `7.03e-6` in it at
  `t = 0.1` and `0.4`, under a ring of amplitude `0.14`, and refining on
  it left the ring coarse. The truncated scheme reads mode `P - 1`
  instead, and so cannot adapt at `P = 1`. With provisional thresholds
  (`1e-2`, `1e-3`) the run matched the uniform one at its cap; at
  `refine_tol = 3e-2` it passed the root mesh as resolved, because mode
  2 of the Gaussian, centred on an element corner, is `2.8e-2` at
  `hₑ = 1.56σ` and `5.7e-2` at half that, and its error reached 7.4.
- 2026-10-31: the indicator made robust before the mesh resolves the
  data. Measured against each other (tables in `CODE.md`, "Adaptive
  refinement" under "Measured results"):
  - the largest of modes `P - 1` and `P - 2`, or the root of the sum of
    their squares: at `P = 3` that is mode 1, which falls only as `hₑ`
    (`0.51` to `0.042` from `roots = 2` to 64), so refinement stops at
    the cap whatever the threshold below `4e-2`. Dropped;
  - every moving mode raised to `(P-1)/m`, unweighted: robust, but it
    overstates smooth data by `(2M - 1)!!`, measured 3.0, 15 and 105 on a
    sine. Weighted by that factor it is the chosen indicator;
  - `u`'s top mode at `t = 0` only, where it is still the data's: fixes
    the truncated scheme's start (worst dip 1.2 at `P = 3`, 1.9 at
    `P = 4`, against 3.3 and 65 for the single mode), but the plain
    scheme has no mode above `P` (its dip stays 35), and the start would
    use another indicator than the run. Dropped;
  - a minimum level: works where the caller knows the feature's width,
    and leaves the indicator wrong on anything it did not anticipate.
    Not built;
  - TreeWave's Löhner indicator fires at every element face. Its
    refreshed global scale is kept.
- 2026-10-31: the thresholds measured, over `P = 3, 4`, caps 2 and 3, to
  `t = 0.2` and `0.4`. The provisional pair did not generalize: at cap 3
  `coarsen_tol = 1e-3` gave 5 to 8 times the uniform run's error, and at
  `P = 4` cap 3 with `refine_tol = 3e-3`, 290 times. Traced, the cause is
  refining data already evolved on a coarser level (the start had
  stopped at level 2, and the decaying peak later lifted the centre's
  score): the transfer is exact at the nodes, yet one chunk later `v`'s
  error is 15 times larger, and the same mesh started from exact data is
  fine. Freezing the indicator's peak at its initial value was tried,
  so that a decaying ring would not cross `refine_tol`; it coarsens the
  blocks around the ring instead and measured worse (1.04 to 1.90, 1.5
  to 7.7). What works is a start that reaches the cap and `coarsen_tol
  = refine_tol/1000`: within 1.003 in L∞ in every case, at 61–90% of the
  node-steps. TreeWave's band is a factor 4.
- 2026-10-31: the first runs read the wake behind the ring as noise
  of `τ ≈ 5e-3` left by an under-resolved start. The exact solution's
  indicator there is `2.1e-3` at level 2 (`P = 3`, `t = 0.4`): the wake
  is the 2D solution's, and the criterion keeps it at the cap, rightly.
- 2026-10-31: `bin/visualize2d.jl --adaptive` added, with the block
  faces coloured by level as in TreeWave's viewer.

## Truncation

- 2026-10-26: truncation was first applied to `∂ₜv` rather than to the
  state at each substep, relying on the linearity of the Runge–Kutta
  stages to keep `v` truncated. Erik Schnetter asked for the mode to be
  set to zero instead, through the integrator's limiter API, as hydro
  codes project back into the valid region of the state, and for the
  initial conditions too. It now is (`truncate_stage!`), and the
  right-hand side no longer projects. The convergence table did not
  change in any printed digit.
- 2026-10-26: in several dimensions the projector is the tensor product,
  removing every mode of degree `P` in any dimension; confirmed by Erik
  Schnetter, as the paper describes one dimension only.
- 2026-10-26: the default is the paper's truncated scheme. Measured
  against the plain one, truncation gives order `P` for `u` and `v`
  alike, while the plain scheme gives a smaller and faster-converging
  `u` and an irregular `v` (tables in `CODE.md`, "Convergence").

- 2026-10-30: the truncation kernel applied the full tensor-product
  matrix, `(P+1)^D` multiply-adds per node, from a copy of `v` made by a
  serial broadcast, whose first write to its scratch array so lay on one
  NUMA domain, and cost more than half a right-hand side (41 ms on
  the laptop, as above). It now works in place, one element per work
  item, as a rank-one update per dimension (`topmode`, `topdual` in
  `Element`), about `2D` multiply-adds per node: 3 ms. The matrix
  `truncate` stays in `Element` for the tests. This was the "kernel per
  element" of "Possible extensions".

## The blast wave and the viewer

- 2026-10-26: Erik Schnetter asked for a 2D visualization like
  TreeWave's radial blast wave, leaving out what this package does not
  support (refinement, backend, centering, type). The blast wave, its
  exact solution and TreeWave's 2D viewer were ported to the nodes; the
  refinement panel (block count and indicator over time) became the
  error and energy over time, and the refinement levels in the radial
  panel became face nodes against interior nodes. SpecialFunctions
  became a dependency, for `besselj0`.

## Threads and the integrator

- 2026-10-30: Erik Schnetter asked what it would take to run efficiently
  on many threads on Symmetry's AMD nodes, and whether IMEXRungeKutta,
  which TreeHydro, TreeGeneralizedHarmonic and TreeGRRMHD use, would
  help. An investigation on the laptop found three costs: OrdinaryDiffEq's
  serial stage broadcasts (79% of a 64-thread step in TreeWave) and its
  serially first-touched cache; with a step limiter, a fifth right-hand
  side and a fifth truncation per RK4 step, as OrdinaryDiffEq refills
  its first-same-as-last slot after the limiter changes the result; and
  the truncation and exchange costs above. He asked for all three
  fixes, then a benchmark on Symmetry.
- 2026-10-30: the integrator is now IMEXRungeKutta's `RK4()` through
  `wave_integrator`, with `truncate_stage!` as both limiters and the
  stage arithmetic split by block owner (`state_partition`, TreeHydro's
  helper). `wave_errors` and `uniform_blast` take a tableau as `alg`
  instead of an OrdinaryDiffEq constructor, and `uniform_blast` calls its
  observer after the steps nearest the snapshot times instead of at
  interpolated ones, so that an observed run takes the same steps as an
  unobserved one. The measured errors agree with the OrdinaryDiffEq runs
  to roundoff (the last blast-wave L∞ at `P = 2`, 1.2646, had been
  printed as 1.27 and is now 1.26). On the laptop a step went from 497
  to 118 ms at one thread and from 155 to 40 ms at six, all three
  changes together.
- 2026-10-30: measured on Symmetry (job 586013, cn085, EPYC 7532, 64
  pinned threads; 3D, `P = 3`, 16.8M nodes), against `main` in the same
  job: a step went from 714 to 72 ms with `16³` blocks and from 688 to
  64 ms with `32³`, and its speedup over one thread from 12–14× to
  30–33×. Tables in `CODE.md`, "Threads on Symmetry". A first
  submission (job 586012) failed at once, because the scripts loaded
  TreeAMR by name from an environment that only developed the package.
- 2026-10-30: two changes to IMEXRungeKutta were considered and kept as
  ideas: partitioning an `ArrayPartition` state, which `INSPIRAL-PLAN.md`
  plans for the inspiral with OrdinaryDiffEq, and a low-storage method.
  `INSPIRAL-PLAN.md` was left as it is at Erik Schnetter's request, so
  its `ArrayPartition`/OrdinaryDiffEq state in M3 is now at odds with
  the package's integrator; "Possible extensions" in `CODE.md` has the
  two ways out.

## Particles

- 2026-10-30: M1 of `INSPIRAL-PLAN.md`, the analytic particle, on branch
  `inspiral-m1-particle`: the derivations in `derivations/`, the
  generated `src/particle_forms.jl`, `src/particle.jl`, and
  `test/particle_tests.jl`. It settles these **(open)** items of
  `INSPIRAL.md`:
  - The interiors of `B₃` and `B₅` are derived (in `CODE.md`, "The
    particle").
  - The near field of orders 1–3, derived again from the retardation
    series without using `INSPIRAL.md`'s results, equals them inside
    and outside the particle. Its interior pieces are confirmed.
  - The `a² a` part of the order-3 self-force is
    `-(15246/482885π) Q²ℓ a² a ≈ -0.01005 Q²ℓ a² a`, 6/5 of the `ä`
    coefficient.
  - The window's step is the polynomial of degree `2m + 1` with `m`
    vanishing derivatives at both ends, `m = P + 1` (`INSPIRAL-PLAN.md`;
    "Settled later" below).
- 2026-10-30: found in M1, and not in `INSPIRAL.md`: order 2 has a
  nonzero potential average, `(2541/193154π) Qℓ a²`, which enters the
  mass `M₀ + QΦ̄ₑ`.
- 2026-10-30: found in M1, and at odds with `INSPIRAL.md`'s "Moving
  particles and the lab slice", which expects the missing lab-slice
  re-expansion to cost only terms suppressed by powers of `v`. The
  bracket takes `∂ₜ²`, which turns the re-expansion's `t′²` term,
  `½(v·r)²(-a·∇Φ₀)`, into `(a·r)²(a·∇Φ₀)` at rest. So until M5 the
  order-3 residual keeps an `a³` part that behaves like `r⁰`, plus
  finite-size terms in `(ℓ/r)²`, and carrying order 3 removes only its
  `ä` part. On a circle that part is `v²` smaller and the expectation
  holds at small `v`; for hyperbolic motion at `t = 0` (M4) it does not.
  This is a gate of `INSPIRAL-PLAN.md` ("Gates of the finished
  milestones" below) and goes
  to Erik Schnetter before M4; `CODE.md` keeps it under "Open
  questions".
- 2026-10-30: the bracket is assembled without cancellation, as the plan
  asks. Measured, the naive difference loses more than the plan's
  estimate of three digits: eleven at `v = 0.2` and all of them at
  `v = 0.05` (`CODE.md`, "The source bracket"), because `A`'s order-1
  part also cancels against `∇²Φ⁽¹⁾`.
- 2026-10-30: the generator writes each interior polynomial in `σ²`, or
  in `1 - σ²` where that is better conditioned. Written in `σ²`, `ρ₀`
  lost 12 digits near the particle's edge.
- 2026-10-30: M2 of `INSPIRAL-PLAN.md`, the transfer, on branch
  `inspiral-m2-transfer`: `src/transfer.jl`, `test/transfer_tests.jl` and
  `bin/projection.jl`. It departs from `INSPIRAL.md`'s "Quadrature" and
  the plan's M2 in three places, all measured first:
  - **The near rule is split at the particle's support.** The plan's
    tensor sinh rule cannot integrate the profile to `10⁻¹²`: `ρ₀` is
    only `C³` at `σ = 1`, and across that edge Gauss rules converge
    algebraically, with or without the sinh transformation (in 1D at
    `ℓ/hₑ = 10⁻³`, `5·10⁻⁵` at 128 points, against `2·10⁻²³` for a
    Lorentzian of the same width). The same edge is in `∇ρ` (`C²`) and,
    weaker, in `S`. Erik Schnetter chose, of three options put to him,
    a rule that splits every line at the support; "Quadrature" in
    `CODE.md` has it.
  - **The near rule is nested, not a tensor product,** also on the near
    elements that the support does not meet. One sinh scale per
    dimension does not fit every line of an element: with the particle
    just outside a face, `1/r²` needed 64 points per dimension at
    `ℓ/hₑ = 10⁻³` and `1/r³` did not converge by 64. With each line's
    scale its own distance from the particle, as Johnston and Elliott
    do in 2D, the points grow like `log(hₑ/ℓ)`.
  - **`r_near` is `5hₑ`, not `2hₑ`.** On the first run's source the far
    rule (`q = P + 3`) is off by `5·10⁻⁹` relative at `2hₑ`, and reaches
    `10⁻¹²` near `4.5hₑ`, at every order of the near field.
  The plan's derivative table at the far rule's points is not built:
  nothing reads a derivative there, and `reconstruct_gradient` takes the
  Lagrange polynomials' derivatives at arbitrary points from the product
  form.
- 2026-10-30: the cost gate of `INSPIRAL-PLAN.md` ("Gates of the
  finished milestones" below) fired: one projection costs far more than ten times the
  wave equation's work on the window's elements (`CODE.md`, "The
  transfer's cost"). A source evaluation is 2.9 µs, where
  `INSPIRAL.md`'s estimate assumed a tenth of that. Erik Schnetter chose
  to measure only in M2: `q` stays `P + 3` everywhere, and neither the
  plan's first fallback (`q = P + 1` far out) nor tuning the source is
  done yet.
- 2026-10-30: the source is tuned, on Erik Schnetter's request, after
  the cost gate. The bracket was nested dual numbers through the
  trajectory, the boost and `α′` at every point, 16 components deep for
  `Φ_C`'s derivatives in the deformation parameter `s`, which were
  evaluated twice, and a further dual pass for `∇Ψ̂` in the window's
  transition. It is now one evaluation of `Ψ` through its four
  invariants on nested dual numbers, with the invariants' `x`-independent
  factors and their time derivatives computed once per particle
  (`ParticleSource`, `ParticleJets`; `CODE.md`, "The split"), and the
  deformation parameter is gone. This also folds each near-field term's
  `(v·∂_z)²` into the acceleration's part instead of subtracting it. The
  generator now also emits the near-field terms as functions of the
  invariants, and the window's step sums its binomials by a recurrence.
  A source evaluation went from 2.9 µs to 0.16 µs, and one projection
  from 20.3 s (far rule 8.3, near rule 11.6) to 2.08 s at `r_near = 5hₑ`
  and from 11.5 s (8.2, 3.4) to 0.95 s at `2hₑ` (`CODE.md`, "The
  transfer's cost"), 210 and 100 times the wave equation's work on the
  window's elements, where they were 2000 and 1100. The `Float64`
  bracket's error against the naive difference in `BigFloat`, on twelve
  points at `σ = 0.3, 0.9, 1.5, 4` with `ä = -4va`, is `3.4·10⁻¹⁴`
  against the old `3.5·10⁻¹⁴` at `v = 0.2`, and `6.7·10⁻¹³` against
  `1.1·10⁻¹²` at `v = 0.05`.
- 2026-10-30: a graded rule between `r_near` and the far rule, on Erik
  Schnetter's request ("a better far rule, which would let `r_near`
  shrink and cut the near work", the extension `CODE.md` proposed).
  Away from the support the source is analytic and a tensor Gauss rule
  converges geometrically in the element's distance from the particle, so
  the points per dimension now grow toward the particle (14, 10, 8, 7 from
  `1hₑ`, `2hₑ`, `3hₑ`, `4hₑ`), the far rule's 6 apply from `9hₑ`, and the
  nested rule only within `hₑ`. The far rule on subdivided elements, the
  other option, was not tried. Sampling elements in random directions
  showed that the old `r_near = 5hₑ` was optimistic: it came from
  elements on the axes, offset `0.3hₑ` across, where the far rule's
  largest error relative to `∫|S|` was

  | `δ` | `2hₑ` | `3hₑ` | `4hₑ` | `4.5hₑ` |
  |---|---|---|---|---|
  | order 0 | 9.3e-10 | 3.4e-11 | 3.0e-12 | 1.1e-12 |
  | order 1 | 3.4e-9 | 7.7e-11 | 4.4e-12 | 1.3e-12 |
  | order 2 | 4.2e-9 | 9.1e-11 | 5.0e-12 | 1.5e-12 |
  | order 3 | 5.1e-9 | 1.3e-10 | 7.6e-12 | 2.4e-12 |

  while at random it is `1.7·10⁻¹¹` at `5hₑ` and reaches `10⁻¹²` only at
  `9hₑ` (`CODE.md`, "Quadrature"). With the old rules, elements between
  the particles had also come out within `3·10⁻¹⁵` of a 24-point Gauss
  rule clustered toward either particle. One projection at `hₑ = d/8`
  went from 2.08 s to 0.87 s, of which the near work from 1.49 s on
  1 368 elements to 0.27 s on 16 (graded rule 0.12 s on 5 936), and now
  meets `10⁻¹²` where it did not; about 90 times the wave equation's work
  on the window's elements. `bin/projection.jl` reports the graded rule
  and takes `--offgrid`.
- 2026-10-30: Erik Schnetter accepted the projection's cost, about 90
  times the wave equation's work where the gate asks for ten, for now;
  it is to be revisited later, probably together with threading or a
  device. The gate's fallbacks (`q = P + 1` far out, the exterior and
  core split) are not taken.
- 2026-10-30: found in M2: elements cut by the window's spheres `R₁`
  and `R₂` carry a quadrature error of `10⁻⁸`–`3·10⁻⁷` relative at
  `hₑ = d/8`, since the window is only `C^{P+1}` there. It is static,
  and `INSPIRAL.md`'s tolerance of `10⁻¹²` does not hold on those
  elements.
- 2026-11-06: the owner asked to try the plan's first fallback, a far
  rule of `q = P + 1` away from the particles. Measured per element on
  the first run off the grid, with the M5 source, it does not pay: of
  the far rule's 31 102 elements 24 lie inside `R₁`, and the others in
  the window's transition or cut by its spheres, where `q = 4` errs by
  `10⁻⁵`–`4·10⁻⁴` of the element's largest `|PS|` against `q = 6`'s
  `10⁻¹¹`–`10⁻⁹`: the window is a polynomial of degree nine in `r` over
  ten elements. Not implemented. The same measurement put the cut
  elements' error at about `10⁻⁴` at `q = 6`, also on `main`, not M2's
  `10⁻⁸`–`3·10⁻⁷`, whose measure is not recorded.
- 2026-10-30: the pairing reads the state on the host, over the support's
  elements found from `z` (`INSPIRAL-PLAN.md`; "Settled later"
  below),
  measured against the `mesh_fold` it replaces (`CODE.md`, "The
  transfer's cost").
- 2026-10-30: M3 of `INSPIRAL-PLAN.md`, a particle at rest and in uniform
  motion, on branch `inspiral-m3-uniform`, started at Erik Schnetter's
  request: `src/inspiral.jl`, `test/inspiral_tests.jl`,
  `bin/uniform_motion.jl` and `bin/symmetry/uniform_motion.sh`. It
  departs from the plan in two places:
  - **The state is a plain vector, not an `ArrayPartition`.** The
    decision of 2026-10-30 (`INSPIRAL-PLAN.md`; "The plan's status and decisions" below) was made
    for OrdinaryDiffEq, before the integrator became IMEXRungeKutta, whose
    stage arithmetic by block owner takes a plain `Array` only and would
    fall back to its broadcast for an `ArrayPartition`. The particles'
    six numbers each go after the field's state vector, owned by thread 1
    in the partition, the first of the two options `CODE.md` listed
    under "Possible extensions"; IMEXRungeKutta needs no change. This
    also settles the plan's "`SVector` or `Vector`" for the particles'
    part (`INSPIRAL-PLAN.md`; "Settled later" below).
  - **A prescribed motion drives the source, not the state.** The source
    and the forces take the trajectories from the path at `t`; the
    particles' part of the state integrates `ż = v`, `v̇ = a` from the
    path alongside, as the plan says, and the tests check that it follows
    the path.
- 2026-10-30: found in M3: the error of the box's faces travels inward
  at about twice the speed of light, independent of the resolution, so
  a point at distance `r` from the centre of a box of side `L` is clean
  until about `(L/2 - r)/2`, not `L/2 - r` (`CODE.md`, "Uniform motion").
  Once there, it is the size of the remainder itself, and the force it
  exerts grows to ten times the first run's radiation-reaction force.
  This fires the gate "the clean time is too short" of
  `INSPIRAL-PLAN.md` ("Gates of the finished milestones" below) for M4's
  circle, measured
  at `t ≈ 5`, and goes to Erik Schnetter before M4.
- 2026-10-30: found in M3: the nested rule was compiled without
  specializing on its integrand, which it only passes on, and inference
  widened its recursion from one dimension to the next, so every
  accumulator, the `(P+1)^D` values of an element's projection, was boxed
  and every call below the top dispatched dynamically, and
  `element_basis` allocated its partial Kronecker products. With the
  integrand specialized, a method per level, and the products by
  recursion, the nested rule on the 20 near elements of a coarse box
  (`hₑ = 1/2`) went from 0.126 s and 630 MB allocated to 0.080 s and
  1.7 MB on the laptop at four threads. The results are unchanged: the
  suite's transfer tests pass as before.
- 2026-10-30: found in M3, on Symmetry: at 64 threads the source of one
  uniformly moving particle took 0.12 s at `hₑ = d/4`, barely less than
  on the laptop's four threads, because the near pass was threaded over
  elements and the element that holds the particle is about a third of
  its work (69 ms of 237 ms serially on the laptop, 342 000 points). The
  near pass is now threaded over the points of each element's outermost
  line, and summed afterwards in the nested rule's order, so it still
  does not depend on the thread count; it equals the serial nested rule to
  an ulp, where the compiler fuses a `muladd` in one and not the other
  (a test). Each line of the rule is now its points and weights
  (`line_fold`), shared by both, and each level of the recursion has its
  own methods throughout. On the laptop the near pass of that box takes
  0.229 s at one thread, against the 0.237 s its elements took one after
  another, 0.069 s at four and 0.052 s at eight; on Symmetry, see
  `CODE.md`, "Uniform motion".
- 2026-11-01: Erik Schnetter answered M3's clean-time gate: from M4 on,
  the inspiral runs on a static forest of nested boxes about the centre,
  fixed before the run, not adaptive. TreeAMR 0.2.7's element centering
  and the refined forests here already carry the field. `INSPIRAL-PLAN.md`
  gets M3b, which builds the nested boxes and lets the transfer run on
  their finest level, and the binary's warm-up and orbits, deferred on
  2026-10-30 until mesh refinement or a better outer boundary, become its
  M7. The gate's own fallbacks, a smaller circle radius or `P = 4` at a
  coarser `hₑ`, are not taken. Neither are initial data without net
  charge or periodic, nor an outflow boundary, nor adaptive refinement.
  The region that needs resolution, the window's ball about the orbit,
  is known before the run. Regrids would change the operator and the
  energy, and refining evolved data costs accuracy (`CODE.md`, "The
  verdict"). The plan's estimate is seven levels, about `10⁸` nodes at
  `d/8` and 64 GB at `d/12`, with the outer faces' error clean at the
  window's edge to `t ≈ 126`, against `8.6·10⁹` nodes for a uniform box
  of side 64 at `d/8`.

- 2026-11-01: M3b of `INSPIRAL-PLAN.md`, nested boxes, on branch
  `inspiral-m3b-nested`, started at Erik Schnetter's request:
  `nested_cubes` and the region list of `wave_forest`, the transfer on
  the finest level, `test/nested_tests.jl`, `--levels` and `--w0` in
  `bin/uniform_motion.jl`, `bin/nested_waves.jl`, and
  `bin/symmetry/nested_motion.sh` and `nested_waves.sh`. The refusal of
  a refined forest (`check_uniform`, 2026-10-31) gave way to
  `check_finest`: every block the source's ball meets, and every block
  holding part of a particle's support, must be on the finest level;
  `InspiralProblem` checks the window's ball and the supports at
  `t = 0`, every projection its ball and the supports, and every pairing
  its support. The refusal names the block's level and its nearest
  point's distance from the ball's centre, which is how far it reaches
  into the ball. Settled along the way, where the plan left it open:
  - **`roots` a multiple of 4.** With the cubes about the box's centre
    and `roots·E` elements across each level, the faces of every cube lie
    on block faces of the level below only then, so `bin/` requires
    `2w₀·hdiv` to be a multiple of `4E`. Around M3's box (`w₀ = 3`) that
    leaves `d/4`, `d/8` and `d/12` of M3's series, not `d/6` or `d/10`.
  - **The uniform reference of the suite is the finest cube alone,** a
    uniform periodic box over it with the same coordinates, rather than
    the whole box at the finest spacing (2.1M nodes for three levels at
    `hₑ = 1/2`): the transfer only reads the elements in the ball, and
    with the same coordinates the element boxes are the same bit for bit.
  - **The outgoing pulse** of `bin/nested_waves.jl` is
    `g(s) = exp(-(s/σ)²) cos(2πs/λ)` with `σ = λ/2`, even, so that
    `[g(r - t) - g(r + t)]/r` is regular at the centre, where a Taylor
    series replaces the quotient.
  On the laptop the suite's coarse three-level forest leaves the error
  in M3's clean ball and the force as M3's uniform box does, and is
  cleaner in the `R₂` ball once the uniform box's faces' error has
  arrived (`CODE.md`, "Nested boxes" under "Measured results").
- 2026-11-02: M3b's Symmetry runs, jobs 587559 and 587560. The series
  on four levels gives M3's errors and force to the digits printed; the
  coarse-fine face sends back no start-up error, where M3's periodic
  face sent one at twice the speed of light; an outgoing wave resolved
  at 8 elements per wavelength across a face comes back at about `10⁻³`,
  one at 4 at about a tenth. The clean-time runs as the plan set them,
  at `v = 0.2` to `t = 20` on 4 to 7 levels, could not measure the
  faces: the window is fixed about the centre, the particle leaves its
  `R₁` ball at `t ≈ 5.7`, which then spoils the force, and the finest
  cube at `t = 15`, where the transfer refused the forest and the job
  stopped after the first of them. `nested_motion.sh` now runs them at
  rest on four and five levels. The source turned out to cost as the
  whole forest rather than the window's ball. The `[sources]` entries
  for TreeAMR moved from the branch of eschnett/TreeAMR.jl#32, merged
  and deleted, to its tag `v0.2.7`, the same tree.
- 2026-11-02: M3b's clean time at rest, jobs 587798 and 587799, four and
  five levels at `d/8` to `t = 20`. The coarse-fine faces send nothing
  back above the discretization error; the two runs agree to every digit
  until the four-level box's outer faces arrive, at the `R₂` sphere at
  `t ≈ 14`, and the five-level one stays clean to the end. The outer
  faces' error crosses the coarse levels at about 1.5 times the speed of
  light, not M3's twice, which superseded the plan's clean times for the
  layouts (`(L/2 - r)/2`, 14 to 126) with about a third more (19 to
  169). M3b's Done-when items hold; it was marked done. The drift of the
  net charge, and six and seven levels, were not run apart: at rest the
  drift arrives with the outer faces' error, and five levels already
  outlast the run.
- 2026-11-02: the owner chose the proposed layout for M4 to M7: `w₀ = 4`
  about the centre of mass, five levels, 64 elements across each at
  `d/8` (`E = 4`, `roots = 16`), an outer box of side 128 at `7.5·10⁷`
  nodes. How many levels M7's five orbits need is left to M7.
- 2026-11-02: M4 of `INSPIRAL-PLAN.md`, prescribed paths, on branch
  `inspiral-m4-prescribed`, started at Erik Schnetter's request:
  `src/prescribed.jl` (`circular_motion`, `hyperbolic_motion`,
  `retarded_field` by root-finding, `point_model`), the retarded field
  as initial data and reference (`inspiral_initial_state(p; retarded)`),
  `ball_norm`'s hole, `test/prescribed_tests.jl`, `bin/prescribed.jl` and
  `bin/symmetry/prescribed.sh`. Chosen here, where the plan left them
  open ("Settled later" below), for the owner to review:
  - **The circle:** `r_c = 0.5`, the first run's, at `v = 0.05`, so
    `Ω = 0.1` and radiation reaction `6.63·10⁻⁸`; the force recorded
    every step to `t = 8` and averaged over `t = 2` to 8, leaving out
    0.25 about each face crossing. A smaller radius would raise the
    signal as `1/r_c²` and let orders 2 and 3 differ visibly; the plan's
    radius was kept.
  - **Hyperbolic motion:** `g = 0.1`, until `v = 0.05` at `t = 0.50`.
    `g` is the size of the first run's accelerations, and `1/g = 10`
    keeps the retarded field defined over the ball the initial data take
    it in (to 3.75 about the centre, the particle 0.5 from it).
  - **The initial data of hyperbolic motion:** the plan's
    `Φ_exact - Φ_p` near the particle and `(1 - W)·…` far out is the
    total field `W_xΦ_ref + (1 - W_x)Ψ̂`, `W_x` one to 3 and zero from
    3.75, with the remainder taken against the point limit of `Ψ̂`, so
    that it is regular at the particle and holds no `ℓ`-dependent
    cancellation (`CODE.md`, "Prescribed paths").
- 2026-11-02: departed from the plan in M4: the circle's series starts
  from its retarded field, not only from the instantaneous data the plan
  names. From instantaneous data the tangential force is about zero to
  `t = 8` and then turns negative, to -2.9 times radiation reaction at
  `t = 30` (job 587929, `d/4`): the junk does not leave the near zone,
  of radius `1/Ω = 10`, by `t ≈ 5`, as the plan assumed from the window's
  size. The instantaneous runs are kept beside it, to record that.
- 2026-11-02: in M4, `project_source!`'s far kernels range over the
  blocks the source's ball meets and split them evenly over the threads,
  instead of every block on its owner, the **(open)** item M3b left. On
  the layout for M4 to M7 the source fell from 0.269 s to 0.087 s at
  `d/8` and from 0.892 s to 0.189 s at `d/12`, a step from 1.40 s to
  0.68 s and from 4.64 s to 1.83 s (jobs 587903 against 587926 on the
  same node, and 587929). M3b's measurement of the source costing as the
  whole forest is superseded; its table in `CODE.md` is kept as M3b's.
- 2026-11-02: found in M4: `retarded_field`'s closures captured variables
  they reassigned, which boxed them, and it allocated on every call; and
  its tolerance did not scale with the positions, so on the layout,
  whose centre is at 64, it refused points it had converged on. Both
  fixed before the series.
- 2026-11-02: M4's go/no-go gate fired, both ways, and goes to Erik
  Schnetter (`INSPIRAL-PLAN.md`; "Gates of the finished milestones"
  below; `CODE.md`,
  "Prescribed paths" under "Measured results"). On the circle the
  tangential force is radiation reaction to a few percent from `d/4` to
  `d/8`, but the radial force stays at 0.2–0.3 of it, and every face
  crossing gives an impulse of tens to hundreds of times it, still 11
  to 56 at `d/12`. Reading the exact remainder's force without evolving
  it shows that the radial force, and the whole of hyperbolic motion's,
  is the `t′²` term of `Φ_C`'s lab-slice re-expansion, the `r⁰` part
  proportional to `v²a` that `INSPIRAL.md` ("Moving particles and the lab
  slice") expected to be suppressed enough at small `v`, and the
  extension `CODE.md` proposed after M1. At `v = 0.05` it is not: the grid
  reads it as 0.14 to 0.55 of radiation reaction, more the finer the
  elements. Taken out, the readings fall by two to three orders, and the
  tangential force converges to radiation reaction.
  Orders 2 and 3 agree to 1% everywhere: at `v = 0.05` and `r_c = 0.5`
  the order-3 terms are too small to see, so the rates 1 and 2 of
  "Why the near field must be analytic" were not measured. The M1
  question whether hyperbolic motion can show second order is answered
  only as far as this: its force is the `t′²` term's. Also found: the
  bracket's omission of `d⁴a·∂_äΦ⁽³⁾` leaves 3.5% of the remainder in
  hyperbolic motion's field at every resolution. The SSH tunnel to
  Symmetry closed at about 13:20 while jobs 587927–587929 and 587933 ran,
  and was reopened later the same day.
- 2026-11-02: Erik Schnetter answered M4's gate: carry the `t′²` term of
  `Φ_C`'s lab-slice re-expansion now, ahead of M5, and re-measure the
  circle and the face-crossing impulses. `derivations/labslice.wl`
  derives `T₂ = -½t′²(α′·∇Φ₀) = -(v·y)²(α′·y)Φ₀′(q)` with the extended
  `Φ₀`, its Laplacian through `∇²Φ₀ = ρ₀` differentiated in `q`, and its
  own force at rest, `(11/60)δm v²α′ + (11/30)δm (v·α′)v`; `generate.wl`
  emits them into `particle_forms.jl`. `lab_slice` evaluates it,
  `particle_field` adds it from order 1 on, and the bracket gets a fifth
  invariant, `v·y = w·r` with `w = γv`, lifted like the others, and
  `∇²_y T₂` beside the near field's. Decided here: it is gated with
  order 1, not always on, since it holds one acceleration and
  `INSPIRAL.md` pairs each order carried with its lab-slice corrections;
  order 0 stays the particle paper's `Φ_C`. This settles the extension
  `CODE.md` proposed after M1 and the M1 finding that the bracket at rest
  keeps `(a·r)²(a·∇Φ₀)` at order 3: a point particle's bracket has no
  order-3 part along a trajectory through rest now. The rest of the
  re-expansion stays M5's. The bracket's error against the naive
  `BigFloat` difference stays at `10⁻¹⁶` of `Qa/ℓ²`, but the bracket
  itself is smaller now, along the acceleration down to `10⁻⁸` to
  `10⁻¹¹` of that, so the relative errors `CODE.md` quoted, `1.7·10⁻¹⁴`
  at `v = 0.2` and `5.0·10⁻¹³` at `v = 0.05`, became `3.1·10⁻⁹` and
  `2.9·10⁻⁶` there; the suite's check of it is now absolute.
- 2026-11-02: M4 re-measured with the `t′²` term, jobs 587989 to 587994
  (and the first series' 587927–587929 and 587933 collected after the
  SSH tunnel was reopened; over the full `t = 2` to 8 the first series'
  `d/12` reads 1.06 and 0.33, not the 1.22 and 0.30 of its first three
  time units). The gate's conditions now hold at order 3: on the circle
  the tangential force is 0.949, 0.975, 0.993 and 1.003 `F_RR` from
  `d/4` to `d/12`, and 1.003 at `d/16` over the 3.3 time units
  collected, against `γ⁴ = 1.005`, and the radial force 0.14, 0.001,
  -0.004 and 0.000. Hyperbolic motion's force falls from `10⁻⁶` to a few
  `10⁻⁹`, a `v²` force the same at every resolution, and at `t = 0`
  falls at first order with order 2 and second with order 3, the first
  measurement of the rates of "Why the near field must be analytic".
  Face crossings still give impulses, 14–19 `F_RR` at `d/12` and 4.4 at
  `d/16`'s slow crossing, falling at about third to fourth order from
  `d/8`, where in the first series the slow crossing's radial impulse
  had stopped falling at 56. Whether that passes the gate, and the
  smooth reading kernel, are the owner's; M4 waits for that answer. The
  term costs the source 18–20%. The tunnel closed again at about 16:40
  while both `d/16` circles ran.
- 2026-11-02: the two `d/16` circles collected whole once the tunnel was
  reopened. With the term `F_t/F_RR` is 1.002 and `F_r/F_RR` 0.000 over
  `t = 2` to 8 (1.003 and 0.004 over the first 3.3 time units), without
  it 1.06 and 0.42 (1.10 and 0.47 over the first 4.4). Away from the
  slow `x` crossing, whose window at `d/16` also holds a `y` crossing,
  every crossing with the term gives at most 1.6 `F_RR`, without it up
  to 27. `CODE.md` had called its radial force outward while dividing
  by the signed radiation-reaction force, which is negative; its
  wording now says that a positive `F_r/F_RR` points inward.
- 2026-11-02 and 2026-11-03: the owner asked whether OLDG's
  reconstruction of a continuous solution would help against the
  impulses, and to test it as a reading first.
  `pairing(…; continuous = true)` averages the nodes elements share
  before reading, and `bin/prescribed.jl --continuous` reads the force
  both ways every step, with the source and the evolution unchanged
  (jobs 588208 to 588210, `d/4` to `d/12` to `t = 8`, `d/16` to `t = 4`;
  the plain reading is the earlier runs' to the bit). The impulses fall
  20 to 100 times, 581 to 8 `F_RR` at `d/4`'s slow crossing and 81 to
  2.6 at `d/8`'s `y` crossing, and from `d/12` on are within radiation
  reaction; the means do not change beyond the scatter. The kernel of
  width `hₑ` was not built. A first version of the averaging took an
  element on the domain's face for its periodic neighbour, through
  `locate_element`'s clamp; it now refuses such an element, as the
  transfer looks for no periodic images.
- 2026-11-03: the owner made the averaged reading the force, with the
  source written through the same averaging, so that the force stays
  the gradient of the discrete interaction energy, and asked that M4's
  gate count as passed if the impulses stay within radiation reaction.
  `average_shared!` averages the projected source over the source
  ball's blocks one dimension at a time, in `du`'s first variable as
  scratch, and `InspiralProblem(…; continuous = true)` is the default;
  `--continuous` became `--plain` and `--both`. The uniform-motion test's
  force at `t = 0`, exactly zero before, is `2·10⁻¹⁰` on the coarse box,
  where the averaging reaches nodes in the window's transition that
  truncation leaves apart; its bound is now `10⁻⁹`. Jobs 588573 to
  588575: near every crossing the force is within radiation reaction
  from `d/12` on, the slow crossing's radial impulse 7.4, 2.5, 1.7, 0.56
  and 0.17 `F_RR` from `d/4` to `d/16`, 5–15% below the read-only
  runs', at 2% of a step or less. The gate passes, and M4 is done on its
  branch.
- 2026-11-05: M5 of `INSPIRAL-PLAN.md`, the lab slice, started on branch
  `inspiral-m5-labslice`, with its derivation and the exact check first.
  Found, and at odds with `INSPIRAL.md`'s "Moving particles and the lab
  slice": the re-expansion of the rest-frame near field at `t′` describes
  a particle whose density is rigid on the rest frame's slices, but the
  model's `ρ₀(|Λ(v)(x - z)|)` is rigid on the lab's. They differ inside
  the particle at order 1 by `½(v·y)²(α′·∇ρ₀)`, so that with `T₂` alone
  the bracket keeps `Q(v·y)²(α′·y)ρ₀′`, linear in the acceleration; on
  `main` the bracket divided by `|a|` tends to `8.4·10⁻⁴` as `a → 0` at a
  point inside, at `v = (0.2, 0.1, 0)`, against `8.42·10⁻⁴` for that
  term, where at rest it tends to zero. It is also what M1 found an extended particle's bracket to keep
  at rest, in `(ℓ/r)²`. `labslice.wl` therefore takes the time-symmetric
  field of the lab-sliced density's history in the rest frame, from the
  same retardation series at any `t′`, at `t′ = -v·y`; its order 1 is
  `T₂` plus the field of that density. The point limits are the same
  either way, and the Born field of uniform acceleration on the tilted
  slice agrees with them through order 3. Decided here: the rest-frame
  `ä` of the near field is the worldline's fourth derivative in the rest
  frame's coordinate time, `C`, not the lab's `ä`, and the lab slice
  takes the third, `B`, which orders 1–3 at rest did not need;
  `rest_frame_derivatives` gives both. Also found: `Λ₁` exerts no force
  on the particle at rest, where `T₂` alone exerts
  `(11/60)δm v²α′ + (11/30)δm (v·α′)v`. `generate.wl` emits the lab slice
  as 98 invariant terms and 98 Laplacian terms, about 2400 lines of
  `particle_forms.jl`, and takes the exterior power of `r` from the
  homogeneity degree, so that a finite-size term without a point limit
  has one; the existing forms are unchanged byte for byte. The particle
  field and the bracket still carry `T₂` and the lab's `ä`.
- 2026-11-05: the owner asked for the lab slice to be wired in and
  hyperbolic motion studied before the circle at `v = 0.2`, and for large
  tests and benchmarks to run on Symmetry. The particle field carries
  `Λ_N` with order `N`, the near field takes `C` for `ä`, and the
  bracket's invariants are now eleven: `q`, the four `u·y` and the six
  dot products of the rest-frame vectors. `T₂` is no longer generated on
  its own, nor its self-force coefficients. Along a motion at `v = 0.19`
  the bracket with orders up to `N` carried now falls as `ε^{N+1}`,
  inside the particle and out. Found on the way, each in the generated
  code: a sum of 74 terms written as one expression allocates, since
  Julia makes it one call of `+`; a `BigFloat` Hessian of order 3 with
  every radial function inlined takes 178 s to compile, 17 s without;
  and wrapping the coefficient tables for every point copied about 10 KB
  per evaluation. The source's cost per point at order 3 went from
  102 ns to 4.4 µs with each term on its own, 1.9 µs with one function
  per order, and 1.0 µs with the terms grouped by their monomial in the
  invariants linear in `y` and the other invariants folded into
  coefficients computed once per particle and step. The particle tests
  take 5m46 on the laptop, against 43 s before M5, mostly `BigFloat`
  compilation.
- 2026-11-05: hyperbolic motion with the lab slice, on Symmetry against
  `main`. The exact remainder's force read at `v = 0.05`, 0.1 and 0.2 is
  a few `10⁻¹²` at order 3 from `d/4` to `d/12`, where `main`'s grows
  with `v` to `10⁻⁸`; M4's `v²` force was the lab slice, not finite-size
  terms. The evolutions to `v = 0.2` match the reading to `t ≈ 1/4` and
  are then covered by the evolved field's own discretization error,
  the same as `main`'s. The projection costs 8.2 times `main`'s at order
  3 and a step 4.9 times. The full suite took 60 minutes on one thread
  of an EPYC 7543. The circle at `v = 0.2` waits for the owner.
- 2026-11-06: the owner asked where the lab slice's projection goes
  before choosing how to speed it up. `bin/projection.jl` now counts the
  nested rule's points inside a particle and times the source there.
  Found: 0.25–1.9% of the points lie inside a particle; a quarter of the
  order-3 source per point is `ParticleSource` copying its 12.9 KB jets
  for each point, which inlining with constant particle indices removes;
  and the nested rule runs on few threads off the grid, 0.81 s where its
  evaluations would take 0.14 s spread evenly.
- 2026-11-06: the owner had both fixes made. `ParticleSource` sums its
  particles by constant index, with `s` passed rather than captured; a
  first version that also inlined the call made the inspiral's
  right-hand side differ from a separate projection by an ulp, since the
  compiler fused a `muladd` differently in the two compiled projections,
  so the call stays a function of its own. For the nested rule, its
  items were first split over the first two lines; that allocated
  3.2 GiB per call and gained nothing, and counting the items showed
  they were fine enough already (1 664, the largest 0.46% of the work).
  The cause was `Threads.@threads :dynamic`, which cuts the list into one
  contiguous chunk per thread, where the heavy items near a particle lie
  together; `:greedy` hands them out one at a time. On one EPYC 7543
  the order-3 projection went from 2.18 to 1.34 s (`main` 0.21 s), its
  nested rule from 0.82 to 0.17 s, and a hyperbolic step at `d/8` from
  3.74 to 2.85 s. The suite on four threads went from 28 to 21 minutes.
- 2026-11-06: the owner asked for the lab slice's closed form outside
  the particles. Outside, each group is already `r^p` times a polynomial
  in `ℓ²/q`, so there was no form to derive; timing the parts showed the
  order-3 term on the bracket's dual numbers at half the source, and the
  powers of `v·y`, `α′·y` and `r` formed in every group at as much as
  its polynomials. The generator now forms them once per evaluation. On
  the laptop one particle's source outside it went from 940 to 540 ns at
  order 3; on one EPYC 7532 the projection from 1.67 to 1.24 s and a
  hyperbolic step at `d/8` from 3.47 to 2.71 s. Dropping the
  polynomials' later coefficients far out would take the term from 274
  to 190 ns, but needs a bound relative to the whole field, and is not
  done. The three suites pass on one EPYC 7532, in 67, 67 and 25
  minutes; that node is slower than the 7543 of the earlier times.
- 2026-11-06: CI failed on main from M4 on, on every job, on one
  assertion: the retarded field's `∂ₜΦ` at `t = 0` of hyperbolic motion,
  bounded by `10⁻¹⁵|Φ|`, is `1.2·10⁻¹⁵|Φ|` at `r = 3`. It vanishes
  there, so the value is roundoff, mostly of the retarded time, which
  lies about `r` back; it is under `3·10⁻¹⁷|Φ|` at `r ≤ 1`. The bound
  is now `10⁻¹⁵ max(r, 1)|Φ|`.
- 2026-11-07: the owner started the circle at `v = 0.2`, orders 2 and 3
  from `d/4` to `d/16`, on Symmetry. The tangential force was 1.0417
  `F_RR` throughout, which is `γ²`, not the `γ⁴ = 1.005` that M4's
  blocks at `v = 0.05` had been set against: for the force the code
  reads the `a²u_μ` parts of the regular field cancel between `∇̄Φ_r`
  and `vΠ̄_r`, and M4's 1.002–1.003 agree with `γ² = 1.0025`. At
  order 2 the radial force fell from 0.0075 to 0.0029 `F_RR`; at order
  3 it went to `-0.024 F_RR` at every resolution, after a transient
  that did not depend on `hₑ` either. Evaluating the source along the
  exact circle showed it off by 3% of the order-3 correction, while it
  matched the residual of the path's Taylor polynomial to `10⁻⁷²`:
  `C`'s `ä` had no second time derivative, since the `Trajectory`
  stopped at `a⃛`. `CODE.md` had called that term, `d⁴a·∂_äΦ⁽³⁾`, of order
  5 and omitted it, but the source is the residual of `WΨ̂` along the
  Taylor polynomial, truncated at no order, so without it the source was
  that of another path. At `v = 0.05` the difference was `10⁻¹⁰` and
  invisible.
- 2026-11-08: the owner asked how a dynamic particle would supply a
  sixth derivative. Not from its equations of motion differentiated
  four more times: in M6 the order-3 `ä`, and `C`'s `ȧ`, are to come
  from the predictor, whose time derivatives the chain rule gives from
  the state and its first two derivatives (`CODE.md`, "The near field",
  **(proposed)**). The owner then had the sixth derivative added:
  `Trajectory` holds `a⃜`, the prescribed paths give it in closed form,
  and a test checks the order-3 source along the exact circle against
  the exact residual to `10⁻⁴⁰`. The circle at `d/8`, order 3, rerun
  with it, gives `F_t = 1.0415 F_RR` and `F_r = -0.0001 F_RR`, with no
  transient, and the field's error falls from `2.5·10⁻⁶` to order 2's
  `1.3·10⁻⁸`. The owner had the other order-3 resolutions rerun with
  it, finished on 2026-11-09: the radial force is `-0.0014 F_RR` at `d/4`
  and below `10⁻⁴` from `d/6` on, the tangential force 1.0417 from `d/12`
  on, and the field's error the same as order 2's at every resolution.
  With that M5's gate holds as M4's did at `v = 0.05`. Two open questions
  in `CODE.md` are settled by it and dropped: whether the bracket should
  hold `d⁴a·∂_äΦ⁽³⁾` (it does, through `a⃜`), and whether a circle can
  tell orders 2 and 3 apart (at `v = 0.2` it can: 0.0050 against
  -0.0001 `F_RR` radially at `d/8`).
- 2026-11-09: `INSPIRAL-PLAN.md` gained M8, proposed at Erik
  Schnetter's request: the particles in the seven-patch ball's cube,
  whose map is affine, with the nested cubes inside it and the shells
  out to a radiative sphere. M8a measures what the sphere sends back
  against the nested boxes and is a gate; M8b lifts a geometry's refusal
  of refined forests where the refinement stays in affine patches; M8c
  moves the transfer into physical coordinates through the cube's map and
  needs a patch-aware `locate_point` from TreeAMR. M7's paragraph on the
  ways out had said:

  > - **A multi-patch outer region,** for example spherical shells
  >   whose radial spacing grows outward. It is mesh machinery, which
  >   belongs upstream ("Scope and non-goals"), and OLDG would need an
  >   interface between patches that keeps summation by parts.
  > - **An outflow or radiative boundary,** the paper's singleton
  >   elements `Ω₀` and `Ω_{K+1}`, which `CODE.md` lists as not
  >   implemented. Even a perfect one at side 24 leaves uniform `d/12`
  >   above a node's memory (below).

  with the first marked inconsistent with `CODE.md`, which has both
  since TreeAMR's M17 and the seven-patch ball. Its decisions row on the
  outer boundary had said "no sponge or outflow boundary in this plan".
- 2026-11-09 to 11-10: M8a's runs, at Erik Schnetter's request, on
  Symmetry (`bin/sphere_waves.jl`, `CODE.md`, "The radiative sphere").
  Two balls of one spacing, `R₁ = 25` and 50, at `λ = 16` for the
  monopole and a centred dipole and at `λ = 4` for the monopole; the
  plan had asked for an off-centre dipole, and the centred one was taken
  as the binary's `ℓ = 1` without the mixture of orders an offset brings.
  The nested boxes' side is M3b's existing runs, not rerun. The first
  reading, at `R₁ = 25`, took the error of about `10⁻⁴` before anything
  could return from the sphere for the cube's own error at `hₑ = 1/2`,
  too large to see a reflection under; the `R₁ = 50` runs showed it was
  the cube–shell faces' at 8, six times smaller where they are at 16. In
  between, a difference against a second problem with the exact pulse as
  Dirichlet data on the sphere was added to the script and four jobs
  started (594081, 594082, 594086, 594088); a laptop run showed that
  the Dirichlet sphere reflects the outgoing discretization error whole,
  more than the radiative one sends back, so the jobs were cancelled
  within half an hour and the option taken out again. The `R₁ = 50` jobs
  also showed the serial `element_truncation`, 1¾ hours of each job.

### What the plan said of its finished milestones

`INSPIRAL-PLAN.md` keeps one line for each finished milestone. Its
earlier text for M0 to M5 is here, as it stood when M5 was done, with
the parts of its status, decisions, gates and items left to settle that
only those milestones used. Why nested boxes, and the estimate for their
layouts, went to `CODE.md` ("Nested boxes").

#### The plan's status and decisions

The plan's status paragraph read:

> **Status.** M0 to M4 are done. M2's cost gate fired, and the cost is
> accepted for now. M3's clean-time gate fired, and the owner answered it
> on 2026-11-01 with a static forest of nested boxes, which is M3b; it
> also brings the binary's warm-up and orbits back as M7. M4's go/no-go
> gate fired on 2026-11-02, and passes since the owner had the `t′²` term
> of `Φ_C`'s lab slice carried ahead of M5 and the source written and the
> forces read through the averaging over shared nodes (2026-11-03).
> M5 is started (2026-11-05). Nothing else is started. The decisions below were settled with the
> owner on 2026-10-30, 2026-11-01, 2026-11-02 and 2026-11-03. A milestone's status line changes here when its work starts
> and when its pull request merges.

**Inconsistent:** this status has M5 started, where M5's own status line
below has its work done.

Its decisions were introduced as "Settled with the owner on 2026-10-30;
the mesh, and with it the scope, on 2026-11-01." Beside the rows the plan
keeps, the table held

| question | decision |
|---|---|
| state vector | `RecursiveArrayTools.ArrayPartition(field, particles)` |

which M3 departed from (above). Its scope row ended "the warm-up and
orbits waited for mesh refinement or a better outer boundary, and are M7
on nested boxes since 2026-11-01", its mesh row was dated 2026-11-01, and
its outer-boundary row said "Short runs only until 2026-11-01". M7's
status line said "It was deferred on 2026-10-30 until mesh refinement or
a better outer boundary. The owner chose static nested boxes on
2026-11-01 (M3b), which makes it a milestone."

Defaults chosen here, to be changed if the owner objects:

- **`StaticArrays` is a new dependency** besides `ForwardDiff` and
  `RecursiveArrayTools`, for 3-vectors, 3×3 solves and `isbits` particle
  states in kernels.
- **The transfer operator is generic in `D`** and tested in 1D and 2D,
  where it is cheap. The particle physics is three-dimensional only.

Its other two defaults, generic in `T` and checkpoints as raw arrays
with a TOML header, are in `CODE.md` ("Particles") and the plan's M7.

#### M0: Documents and tooling

**Status:** done; its documents are on `main`.

- Commit `INSPIRAL.md` (unchanged), the task it answers
  (`INSPIRAL-PROMPT.md`, unchanged) and this file.
- `CLAUDE.md`, "Where things go": a row for `INSPIRAL.md` (the design
  proposal for the inspiral, frozen, for reference), one for
  `INSPIRAL-PROMPT.md`, and one for `INSPIRAL-PLAN.md` (the milestones
  and how each is known to be done). The `PLAN.md` row points at it.
- `CLAUDE.md`, "Commands": the generator command (M1), with the
  `wolframscript` path on the owner's machine,
  `/Applications/Wolfram.app/Contents/MacOS/wolframscript`, and the
  environment variable `TREEWAVEOLDG_WOLFRAMSCRIPT` that the test of M1
  reads.
- `HISTORY.md`: the adoption of the design and the plan, and the
  decisions of 2026-10-30.

**Done when** the pull request has merged and the three documents are
on `main`.

#### M1: The analytic particle

**Status:** done; on `main`.
`INSPIRAL.md`: "The particle profile", "The
near field of an accelerated particle", "The particle field", "The
remainder equations", "The window"; validation step 1.

##### M1 derivations

Wolfram Language scripts in `derivations/`, each runnable alone with
`wolframscript -file`, and each asserting its own identities so that a
wrong derivation fails loudly:

- `profile.wl`: `ρ₀`, `Φ₀`, the moments `⟨s²⟩`, `⟨s⁴⟩`, `⟨s⁶⟩`; `B₁`,
  `B₃`, `B₅` inside and outside, which settles the **(open)** interiors of
  `B₃` and `B₅`. Asserts: `∇²B_k = k(k+1)B_{k-2}`; the interior and
  exterior forms match to the profile's smoothness at `σ = 1`; `δm` and
  `∬ρρ′|x - x′|` take the values `INSPIRAL.md` gives.
- `nearfield.wl`: orders 1–3 for the extended particle from the three
  steps of "How the terms are computed", derived independently of
  `INSPIRAL.md`'s results and compared with them (the **(open)** check of
  the interiors). Asserts: the hierarchy `∇²Φₙ = ∂ₜ²Φₙ₋₂` for every
  term; the point limits; the hyperbolic-motion expansion for
  `k = 1, 2, 3`.
- `selfforce.wl`: the charge-weighted averages of the particle's own
  orders 1–3. These are `δm a`, zero at order 2, the `ä` force, and the
  **(open)** `a² a` force.
- `generate.wl`: loads the others and writes `src/particle_forms.jl`:
  the radial functions as Horner polynomials in `σ²` (interior) and
  closed forms (exterior), the near-field terms and their Laplacians from
  the hierarchy, and the self-force coefficients. The file begins with a
  "generated by `derivations/generate.wl`, do not edit" header and the
  command that regenerates it.

Interior and exterior forms are selected by a branch on `σ < 1`. Each
emitted expression is generic in its argument type, so that it works on
dual numbers.

##### M1 code

`src/particle.jl` (hand-written) on top of `src/particle_forms.jl`
(generated):

- `ParticleModel{T}`: `Q`, `ℓ`, the physical mass `M`, the bare mass
  `M₀ = M - δm(ℓ)`, with a constructor that refuses `M₀ ≤ 0` and gives
  the reason (the classical-radius bound).
- `Trajectory{T}`: `z`, `v`, `a`, `ȧ`, `ä`, `d³a/dt³` as `SVector{3}`s,
  `isbits`. It is the Taylor polynomial in `t` that the time derivatives
  act on.
- `boosted_static(m, x, z, v)`: `Φ_C`, through `y = Λ(v) r`.
- `near_field(m, Val(N), x, traj)` for `N = 1, 2, 3`. Until M5 these are
  the rest-frame terms, evaluated at `y` with `α′ = γ²Λ(v)a`.
- `particle_field(m, x, traj; order)`: `Ψ`, and `particle_rate`: `𝒟Ψ`.
  The rate is a `ForwardDiff` derivative in `t` of `Ψ` along the
  trajectory's Taylor polynomial. `order` (0–3) is a parameter, so that
  M4 can measure what each order buys.
- `source_bracket(m, x, traj; order)`: `𝒟²Ψ - ∇²Ψ + ρ`, assembled
  without cancellation, as "The remainder equations" requires. `ρ`
  cancels against `□_vΦ_C` analytically. The Laplacians of the
  near-field terms come from the generated hierarchy. The
  acceleration-dependent part of `𝒟²Φ_C` is computed as derivatives in a
  deformation parameter `s`, of the motion `z + vt + s·δz(t)`,
  `v + s·δv(t)` with `δz = O(t²)` and `δv = O(t)`, and not as the
  difference of two second derivatives. That difference would lose about
  `log₁₀(v²/aℓ)` digits, three for the first run. **Sharp edge:** `S` is
  never formed by subtracting the large terms.
- `Window{T}`: centre, `R₁`, `R₂`, and a polynomial step of continuity
  `C^{P+1}` or better. Closed forms for `W`, `∇W` and `∇²W`.
- `particle_source(…)`: `S = W·bracket - 2∇W·∇Ψ̂ - Ψ̂∇²W` for a pair of
  trajectories.

##### M1 tests

`test/particle_tests.jl`:

- The closed forms equal `BigFloat` quadrature of their definitions
  (`Φ₀` and `B_k` at radii inside and outside, `δm`, `∬ρρ′|x - x′|`) to
  `10⁻¹⁴` relative in `Float64`.
- `□_vΦ_C = ρ` at `v = 0.2`, in all directions and inside and outside
  the particle, through `BigFloat` derivatives.
- The hierarchy holds for each order to roundoff, through nested
  `ForwardDiff` in `BigFloat`.
- The point limits of orders 1–3 reproduce the expansion of the
  hyperbolic-motion field, and the residual after order 3 falls as `r³`
  (a fitted slope of `3 ± 0.1` over a decade of `r`).
- `𝒟Ψ` and the bracket equal `BigFloat` finite differences in `t`.
- The cancellation-free bracket equals a `BigFloat` evaluation of the
  naive difference to `10⁻¹²` relative, in `Float64`.
- With `a = ȧ = ä = 0` the bracket is zero, to `10⁻¹⁵` of `Q/ℓ³`.
- `W` is 1 inside `R₁` and 0 outside `R₂`, and is `C^{P+1}`.
- Where `wolframscript` is available (`TREEWAVEOLDG_WOLFRAMSCRIPT` is
  set), regenerating `src/particle_forms.jl` reproduces the checked-in
  file byte for byte. Otherwise the test is skipped with an `@info`. CI
  skips it.

**Done when** the derivations run clean, the tests pass, and `CODE.md`
has "The particle", "The near field" and "The window" with the values of
the interiors of `B₃` and `B₅` and the `a² a` coefficient. `HISTORY.md`
records which of `INSPIRAL.md`'s open items M1 settles, and any
disagreement with `INSPIRAL.md`'s results.

#### M2: The transfer operator

**Status:** done; on `main`. The
cost gate of [the gates below](#gates-of-the-finished-milestones) fired, and after
tuning the source and grading the far rule the owner accepted the cost
for now (`HISTORY.md`, "Particles"); the near rule and `r_near` depart
from what is planned here. `INSPIRAL.md`: "The transfer operator", "Quadrature";
validation step 2.

##### M2 code

`src/transfer.jl`, generic in `D`:

- `TransferTables{T,…}`, `isbits`, beside `Element` and leaving it
  unchanged: Gauss–Legendre nodes and weights for `q = P + 3`, the
  `q × (P+1)` Lagrange table on the Gauss–Lobatto nodes, its derivative,
  and the reference Gauss nodes of the sinh rule. All built in
  `BigFloat` and rounded once.
- `sinh_rule(ξc, ε, n)`: Johnston and Elliott's transform on `[-1, 1]`
  for a near-singularity at reference coordinate `ξc` with scale `ε`
  (here `2ℓ/hₑ`). It is smooth in `ξc`, including where `ξc` leaves the
  element, so that the transfer stays smooth in `z`.
- **The projection** `P S`, in two kernels. One evaluates `S` at the
  quadrature points of each element in the window into a scratch array
  (range: `E·q` points per dimension and block). The other contracts it
  against the Lagrange table one dimension at a time, and divides by
  `wₙ`, into the owned nodes of `dv`. Elements whose bounding box misses
  the window ball are skipped by a per-block mask and a per-element
  test. Elements within `r_near ≈ 2hₑ` of a particle use the tensor
  sinh rule, with the point count from a table built once by the
  convergence test of "Quadrature".
- **The adjoint pairing** `⟨P K, g⟩_w = ∫ K g_h` for `K = ρₐ` and
  `K = ∇ρₐ`, over the at most eight elements that meet the particle's
  support. They are found from `z` through `TreeAMR.locate_point` and
  the element index arithmetic of `node_coordinates`, and the state is
  read directly on the host (CPU). A `mesh_fold` over every point would
  cost a pass over the mesh per particle and kernel; it is kept only as
  the reference in the tests. **(Decide in M2 by measuring.)**
- `reconstruct(p, u, x)` and `reconstruct_gradient`: `g_h` and `∇g_h`
  at an arbitrary point, for the flux spheres (M6b) and for tests.
- `ball_rule(n_r, n_θ)`: Gauss in radius and a product Gauss rule on
  the sphere, for the analytic averages over a particle's ball in its
  rest frame (M3 on).

##### M2 tests

`test/transfer_tests.jl`:

- Adjointness: `⟨P S, g⟩_w = ∫ S g_h`, the right side by a much finer
  independent quadrature, for random `g`, to `10⁻¹²`, in 1D, 2D and 3D.
- Consistency: for smooth `S`, `(P S)ₙ - S(xₙ)` falls at order `P` for
  `P = 2, 3, 4` (1D and 2D; one coarse 3D check).
- Near rule: `∫ ρ φₙ` over an element converges to `10⁻¹²` relative for
  `ℓ/hₑ` from `10⁻¹` to `10⁻⁴`, with a point count that grows like
  `log(hₑ/ℓ)`, and `Σₙ wₙ (P ρ)ₙ = Q/γ`.
- Smoothness in `z`: second differences of `⟨P ρ(z), g⟩_w` and of the
  far/near switch, as `z` crosses an element face and `r_near`, show no
  jump above the tolerance.
- The pairing over the support elements equals the `mesh_fold`
  reference.
- `reconstruct` is exact on `Q_P` and equals the stored values at the
  nodes.

**Done when** the tests pass, the timing of one projection over a window
ball of radius `2.5d` at `hₑ = d/8` is measured and recorded (the
**(open)** cost in "Quadrature"), and `CODE.md` has "The transfer" and
"Quadrature".

#### M3: A particle at rest and in uniform motion

**Status:** done; on `main`. The state is a plain vector rather than an
`ArrayPartition`, and a prescribed motion drives the source from its
path (`HISTORY.md`, "Particles"). The clean time is about half what a
light-speed estimate gives, which fired the gate "the clean time is too
short" below. The owner answered it on 2026-11-01 with nested boxes,
M3b. `INSPIRAL.md`: "The remainder on OLDG", "Time integration";
validation step 3.

##### M3 code

`src/inspiral.jl`, first part:

- `InspiralProblem`: a `WaveProblem`, the particle models, the window,
  the transfer tables, the near-field `order`, and a motion: `Prescribed`
  (a function `t -> (Trajectory, Trajectory)`, `isbits`) now, `Dynamic`
  in M6a.
- The state is `ArrayPartition(field, particles)`, where `particles` is
  an `SVector{12}` or a 12-element `Vector` (whichever OrdinaryDiffEq's
  RK4 cache handles without allocating per step), and `field` is the
  usual `statevector`. `inspiral_stage!(u, integrator, p, t)` truncates
  `u.x[1]` and is the stage and step limiter.
- `inspiral_rhs!(du, u, p, t)` with the steps of "The right-hand side"
  that need no solve: `wave_rhs!` on the field part, `P S` subtracted
  from `dv`, `ż = v` and `v̇ = a` from the prescribed motion. The forces
  are computed as a diagnostic (`particle_forces`) and do not yet act.
- `inspiral_initial_state(p)`: `Φ_r = (1 - W)Ψ̂`, `Π_r = (1 - W)𝒟Ψ̂`
  through `fill_at_nodes!`, `Π_r` truncated.
- `ball_norm(p, u, center, radius)`: `quadrature_norm` restricted to a
  ball, through `mesh_fold` on node positions. **Sharp edge:** in a
  periodic box the field of a charge is not periodic, and every error
  norm of an inspiral run is taken in the causally clean ball only.

##### M3 tests and measurements

`test/inspiral_tests.jl`, coarse 3D (`P = 3`, few elements):

- The right-hand side neither writes its state nor remembers a call;
  every stage has `v` truncated; plain `RK4()` without the limiter
  leaves the mode in.
- A particle at rest: with `Φ_r(0) = (1 - W)Φ_C`, `Φ_r` stays there and
  the force stays zero, both to the window's discretization error, which
  falls between two coarse resolutions.
- Uniform motion, `v = 0.2`: `Φ_r(t)` tracks `(1 - W)Φ_C(z + vt)` in the
  clean ball, and the force stays near zero.

`bin/uniform_motion.jl`, a series `hₑ = d/4 … d/12`. Errors of `Φ_r`
and of the force fall at order `P`. Measured and recorded:

- **The clean time.** When the error from the box faces (the initial
  data is not periodic) reaches the centre, and how large it is compared
  with the window's discretization error. It sizes the boxes of M3b–M7.
- **The cost.** Seconds per right-hand side and bytes per node, wave part
  and source part separately, on one Symmetry AMD node at 64 threads.

**Done when** the tests pass, the series converges at order `P`, the
clean time and the cost are recorded, and `CODE.md` has "The right-hand
side with particles" and the measurements. README's status mentions the
particle runs.

#### M3b: Nested boxes

**Status:** done; on `main`. The owner chose the layout for M4 to M7 on
2026-11-02: `w₀ = 4`, five levels, 64 elements across each at `d/8`,
`E = 4` (`CODE.md`, "Nested boxes"). The forest, the transfer on its finest
level, the norms, the scripts and the suite's tests are written and
pass. On Symmetry the series, the cost and the outgoing waves are
measured (jobs 587559 and 587560), and the clean time at rest on four
and five levels (587798 and 587799), since the particle at `v = 0.2`
leaves the fixed window's `R₁` ball at `t ≈ 5.7`. Decided with the owner
on 2026-11-01, in answer to M3's clean-time gate: from M4 on, the
inspiral runs on a static forest of nested boxes about the centre, fixed
before the run and not adaptive. The field side exists (TreeAMR 0.2.7's
element centering; `CODE.md`, "Refined forests"). This milestone builds
the forest with more than two levels and lets the particles run on it.
It also measures what the coarse-fine faces send back into the window,
and what the layout costs.

##### M3b code

- **The forest.** `wave_forest(…; refined)` takes a list of regions,
  one per level above the root level. Each region refines the leaves of
  the level below whose centres lie in it, and the forest is balanced
  after each. `refined = true` and a single tuple keep their present
  two-level meaning. A helper builds the nested cubes from `w₀` and the
  number of levels.
- **The transfer on the finest level.** `check_uniform` gives way to a
  check that every block the source's ball meets is on the finest
  level, and so is every block that holds part of a particle's support.
  Anything else is refused, naming the block's level and its distance
  from the ball. `hₑ`, and with it `rnear`, `rfar` and the graded bands,
  stays the finest level's, which is then every level the transfer
  touches. `InspiralProblem` checks the window's ball and the initial
  positions when it is built. Each projection and each pairing checks
  again, since the particles move. **Sharp edge**, replacing "the field
  runs on refined forests; the particles do not": the particles need the
  finest level under their source's ball.
- **The norms and the initial data.** `ball_norm` and the initial
  remainder are checked on a refined forest, with each block's own
  spacing.
- **Scripts.** `bin/uniform_motion.jl` takes the number of levels and
  `w₀`. A new `bin/nested_waves.jl` sends outgoing waves through the
  levels (below). Each gets a job script in `bin/symmetry/`.

##### M3b tests and measurements

In the suite, coarse 3D:

- **The same elements give the same source.** On a three-level forest
  whose finest level covers the source's ball, `project_source!` and the
  pairing equal those on a uniform forest at the finest spacing, element
  for element in the ball, to an ulp.
- **The refusal.** A ball or a support that reaches a coarser block is
  refused; one that stays inside the finest level is not.
- **`ball_norm`** on a nested forest equals the uniform forest's for a
  ball inside the finest level.
- **A particle in uniform motion** on a coarse nested forest: as in M3,
  the error in the clean ball falls between two resolutions, and the
  force stays near zero.

On Symmetry:

- **Uniform motion on nested boxes.** M3's series and clean-time runs,
  around M3's box, with 4 to 7 levels:
  - Until M3's clean time, the errors in the `R₁` and `R₂` balls and the
    force equal M3's.
  - After it, record when anything arrives and from which face (the
    first coarse-fine face at `w₀`, each further one, the outer faces),
    against `(L/2 - r)/2`.
  - The net charge's drift falls as the box's volume grows, `10³` to
    `10⁶` times over M3's box.
- **Outgoing waves through the levels,** the field alone. The exact
  outgoing pulse `[g(r - t) - g(r + t)]/r`, with a smooth `g` of
  wavelength 16 (the first run's) and of wavelength 4 (shorter than any
  physical wave, standing in for the start-up junk). After the pulse has
  left the finest cube, whatever comes back into it is the coarse-fine
  faces' reflection. Record it for each face against the outgoing
  amplitude. The coarse ghost's degree loss (`CODE.md`, "Accuracy",
  **(open)**) shows here if anywhere.
- **The cost.** Nodes, bytes per node, `inspiral_rhs!` and a step, against
  M3's uniform box at the same finest spacing.
- **The layout for M4 to M7.** `w₀`, the number of levels, the elements
  across a level and `E`, chosen from the measurements above and
  recorded.

**Done when** all of these hold:

- the tests pass;
- on nested boxes, uniform motion converges at order `P` in the clean
  balls, with M3's errors;
- the time until the force exceeds ten times its clean level covers M4's
  circle (`t ≈ 5`, plus a time to measure over);
- the reflections are recorded;
- `CODE.md` has "Nested boxes" under "Particles", with the measurements,
  and `CLAUDE.md`'s sharp edge is changed.

#### M4: Prescribed paths

**Status:** done; on `main`. The
go/no-go gate fired on 2026-11-02: the radial force did not converge to
zero, and face crossings gave impulses far above the radiation-reaction
force. The owner chose to carry the `t′²` term of `Φ_C`'s lab slice
ahead of M5, which makes the forces converge, and then to write the
source and read the forces through the averaging over shared nodes, the
paper's post-processing, which keeps every crossing within radiation
reaction from `d/12` on (2026-11-03; `CODE.md`, "The averaging" and
"Prescribed paths" under "Measured results"). The circle was also
started from its retarded field, beside the instantaneous data the plan
names (`CODE.md`, "Prescribed paths"). `INSPIRAL.md`:
validation step 4, at `v ≤ 0.05`.

- **Hyperbolic motion.** The exact retarded field of a point charge, by
  root-finding for the retarded time (`bin/` and test helper). Initial
  data at `t = 0`: `Φ_r = Φ_exact - Φ_p` outside a few `ℓ` of the
  particle, and `(1 - W)·…` far out. Evolve over a time during which `v`
  stays below `0.05`. Check: `Φ_r + Φ_p` matches the exact field away
  from the particle, and the remainder's force converges to zero.
- **The slow circle.** `v = 0.05`, a radius `r_c ≈ 0.5`, the particle
  set up from instantaneous data, the force measured after the initial
  junk has left the near zone (`t ≈ 5`), on M3b's nested boxes. Check: the tangential force matches `(Q²/12π)ȧ`; the radial
  force converges to zero.
- **Orders.** Both runs at `order = 2` and `order = 3`. The expected
  rates are 1 and 2 ("Why the near field must be analytic"), and they
  are what is recorded.
- **Face crossings.** The circle crosses element faces. Record the force
  against time at the finest resolution, and the size of any impulse at
  a crossing against the radiation-reaction force.

The suite gets one coarse hyperbolic check and one coarse circle check
(the force has the right sign and order of magnitude). The series run
from `bin/prescribed.jl`, on Symmetry if the finest exceeds the laptop.

**Done when** the rates are measured and recorded. **Go/no-go:** with
`order = 3` the tangential force converges to `(Q²/12π)ȧ` and the radial
force to zero. If not, stop and take it to the owner before M5; see
[the gates below](#gates-of-the-finished-milestones).

#### M5: The lab slice

**Status:** started on 2026-11-05 on branch `inspiral-m5-labslice`.
The derivation, the regenerated `particle_forms.jl` and the exact
hyperbolic check are done, and the particle field and the bracket carry
the lab slice; in hyperbolic motion the remainder's force is a few
`10⁻¹²` at order 3 up to `v = 0.2` (`CODE.md`, "Prescribed paths" under
"Measured results"). The circle at `v = 0.2` ran from `d/4` to `d/16` on
2026-11-07: order 2 converges, and order 3 converged to a radial force
of `-0.024 F_RR` until the trajectory carried the sixth derivative of
`z`, `a⃜`. Rerun with it on 2026-11-08 and 09, order 3's radial force
converges to zero, below `10⁻⁴ F_RR` from `d/6` on, and its tangential
force to `γ²F_RR` (not `γ⁴F_RR`, which M4 had expected), as at
`v = 0.05`; order 2's radial force falls only from 0.0075 to 0.0029
`F_RR` (`CODE.md`; `HISTORY.md`, 2026-11-07 and 08). M5's work is done;
its pull request waits for the owner. The lab slice is that of the
lab-sliced density, not of `INSPIRAL.md`'s re-expansion of the
rest-frame near field (`CODE.md`, "The near field"; `HISTORY.md`,
2026-11-05). `derivations/labslice.wl` exists since M4, with the `t′²`
term of `Φ_C` that the owner had carried ahead of this milestone; M5
extends it. `INSPIRAL.md`: "Moving particles and the lab slice";
validation step 5.

- `derivations/labslice.wl`: the re-expansion
  `Ψ_lab(r) = Σⱼ (t′ʲ/j!) ∂ⱼ_{t′} Ψ′(0, y)` for orders 1–3, with
  proper-time derivatives and `α′ = γ²Λ(v)a`. Asserts: the order-1 term
  has the `v² a` part, of order `r⁰`, that the boosted form lacks; each
  order satisfies the lab-frame hierarchy. Regenerate
  `src/particle_forms.jl`. `near_field` switches to the lab-slice forms.
- **An independent check:** hyperbolic motion at `t ≠ 0`, where
  `v ≠ 0`. The lab-slice expansion must reproduce the expansion of the
  exact field about the particle, which is an exact known function.
  This goes into the suite as a `BigFloat` comparison.
- Re-run M4's circle at `v = 0.2`. **Done when** it converges at the
  same rates as at `v = 0.05`, and `CODE.md` describes the lab slice.
  `HISTORY.md` records that the **(open)** re-expansion is settled.

#### Gates of the finished milestones

- **M1: the derivation disagrees with `INSPIRAL.md`'s near-field
  results.** Find which is wrong with a third method (the
  hyperbolic-motion expansion), record it in `HISTORY.md`, and use the
  verified forms.
- **M2: the projection is too slow** (much more than ten times the wave
  equation's work on the window's elements). First drop `q` to `P + 1`
  far from the particles. Then consider the exterior and core split of
  "Quadrature".
- **M4: no convergence, or visible face-crossing impulses.** For the
  impulses, the smooth reading kernel of width about `hₑ` from "The
  transfer operator" (**(open)** there). For no convergence, report the
  rates per order before going on. Fired on 2026-11-02, both ways: the
  radial force on the circle does not fall with `hₑ`, which the `t′²`
  term of `Φ_C`'s lab-slice re-expansion explains, and every face
  crossing gives an impulse far above radiation reaction. Taken to the
  owner, who chose to carry that term ahead of M5 (2026-11-02). With it
  the forces converge; the impulses fall with resolution but remain
  above radiation reaction at `d/12`, and go to the owner again
  (`CODE.md`, "Prescribed paths" under "Measured results"). At the
  owner's suggestion the force was then read from the field made
  continuous by averaging shared nodes, a cheaper alternative to the
  kernel, which took the impulses away; the owner made it the force,
  with the source written through it too, and the gate passes
  (2026-11-03).
- **M5: the lab-slice forms are too large to generate usefully.**
  Evaluate the re-expansion at runtime with nested `ForwardDiff` in `t′`
  instead of closed forms, and measure the cost.
- **M3: the clean time is too short** for M4's circle to settle in a
  box that fits. A smaller circle radius, or `P = 4` at a coarser `hₑ`.
  Fired; on 2026-11-01 the owner chose nested boxes instead (M3b).
- **M3b: the coarse-fine faces send back more than the window's
  discretization error** into the window before M4's measuring time.
  First a larger finest cube, so that the first coarse-fine face is
  farther out, and more elements across each level. Then the
  degree-preserving coarse ghost through a finer mortar, which the
  decision of 2026-10-31 to keep the energy-compatible ghost left to
  TreeAMR (`CODE.md`, "Open questions").

#### Settled later

- M1: the polynomial degree of the window's step.
- M2: host-side pairing against `mesh_fold`; `r_near`; the near rule's
  point count.
- M3: whether the particle part of the state is an `SVector` or a
  `Vector`.
- M3b: `w₀`, the number of levels, the elements across a level, and the
  block size `E`.
- M4: the circle's radius and measuring time, and the hyperbolic run's
  `g` and duration. Chosen in M4 (`CODE.md`, "Prescribed paths").

## Dependencies

- 2026-10-26: based on TreeWave's `main` plus its open pull request #8
  ("Fold the diagnostics with TreeAMR 0.2.6's mesh_fold"), so TreeAMR
  0.2.6 is required for `mesh_fold`. Julia 1.13 is the floor, as for the
  other Tree* packages.
- 2026-10-26: CI added, TreeWave's test job without its viewer job (there
  is no `bin/`), with TreeWave's Dependabot and Codecov settings.
- 2026-10-30: IMEXRungeKutta 1.4 replaced OrdinaryDiffEqLowOrderRK and
  SciMLBase. It is not registered, so `Project.toml` and
  `test/Project.toml` name its repository under `[sources]`, as
  TreeHydro's do.
- 2026-10-30: ForwardDiff 1, StaticArrays 1.9 and LinearAlgebra added
  for the particle (M1): ForwardDiff and StaticArrays as
  `INSPIRAL-PLAN.md` decided, LinearAlgebra for `dot`. The tests use
  ForwardDiff and StaticArrays directly, so `test/Project.toml` names
  them too.
- 2026-10-31: TreeAMR 0.2.7 required, for the element centering. It is
  not registered yet, so `Project.toml`, `test/Project.toml` and
  `bin/Project.toml` take it from the branch of its release pull request
  (eschnett/TreeAMR.jl#32) through `[sources]`, until it is; the
  viewer's environment needs its own entry, since its direct TreeAMR
  dependency would otherwise resolve from the registry.
- 2026-11-05: CI in two tiers, as in every Tree* repository (TreeAMR's
  `HISTORY.md`, "Development environment and CI", has the measurements):
  a pull request runs the Linux cells only, uninstrumented; `main`, a
  weekly schedule and a manual dispatch add macOS and its coverage;
  Markdown and `docs/` changes and tags run nothing. The serial Linux
  cell is new: the threaded `include` entry, against a matrix with no
  `threads` key, had merged into it, so CI had run macOS at one thread
  and Linux at four only.
- 2026-11-06: CI builds precompiled code for `x86-64-v3` on Linux and
  `apple-m1` on macOS rather than `native`, puts the target into the
  cache key and prints each runner's CPU, carried over from TreeWave
  (whose `HISTORY.md`, "Precompiled code and mixed runners", has the
  runs). GitHub's Linux runners have several CPUs, and TreeWave's and
  TreeHydro's viewer jobs had restored a cache and rebuilt nearly every
  package anyway. Here the viewer's cache hits had held so far, with 4
  to 33 of 323 packages rebuilt, most likely because consecutive runs
  happened to land on the same CPU model.
- 2026-11-06: confirmed on the pull request's run 42281957473. Its
  first attempt found no cache under the new key and rebuilt everything,
  the viewer 284 packages in 299 s on an AMD EPYC 9V74. The re-run
  restored that cache on an Intel Xeon 6973P-C and rebuilt nothing: the
  viewer's instantiate step took 8 s, and the viewer job 1 m 53 in all.
  In TreeWave the macOS cache held across an Apple M1 and an M2 Pro,
  although GitHub documents every arm64 macOS runner as an M1.
- 2026-11-07: TreeAMR taken from a checkout of its unpushed branch
  `eschnett-claude/multi-patch` (M17) beside this one, through
  `[sources]` path entries in `Project.toml`, `test/Project.toml` and
  `bin/Project.toml`, in place of its tag `v0.2.7`, for the annulus.
  CI cannot resolve a path, so it fails until the entries name the
  pushed branch.
- 2026-11-07: TreeAMR 0.2.8 required, which has M17's forests of
  patches, and registered, so all three `[sources]` entries for TreeAMR
  are gone, the path entries above and the `v0.2.7` tag's before them;
  the Symmetry script's `setup` no longer develops a TreeAMR checkout.
  The suite and the annulus pass against it unchanged.
- 2026-11-10: MultiFloats 3.3.2 added, for building the full geometry's
  truncation tables in `Float64x2` ("Multi-patch forests").

# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

M0 to M8, with M3b, M6a, M6b, M8a, M8b and M8c, are the milestones of
`INSPIRAL-PLAN.md`: M0 the documents, M1 the analytic particle, M2 the
transfer operator, M3 a particle at rest and in uniform motion, M3b
nested boxes, M4 prescribed paths, M5 the lab slice, M6a one particle
solved, M6b the binary's parts, M7 the binary's warm-up and orbits, M8
the particles in the seven-patch ball. M16
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
- 2026-11-10: the thirteen-patch ball, at Erik Schnetter's request: the
  seven-patch ball out to a middle sphere and six shells beyond it whose
  radial coordinate is exactly radial, which the 2026-11-08 entry above
  had named and `PLAN.md` had proposed as "Purely radial outer shells".
  `PLAN.md`'s item read: with `r = r(ξ_k)` alone `√g = r² r′ ω(ξ_j, ξ_l)`
  factorizes into a radial and an angular part, so the truncation is a
  one-dimensional rank-one pair along the radius times a
  two-dimensional projection on the face, about 1600 multiply-adds per
  element at `P = 3` against 4200, a `P⁴ = 81`-entry Gram factor against
  `P⁶ = 729`, and one face's angular tables shared by every radial
  element and shell; it would also remove the shear near the cube's
  corners from all but the layer next to the cube. Two choices were his:
  - **The factored truncation now,** not the per-element tables on all
    twelve shells with the factorization left for later. A full geometry
    may declare that a patch's `√g` factorizes (`factored_volume`,
    `radial_volume`, `angular_volume`, `angular_class`), and
    `ElementTruncation` holds a `FactoredTruncation` for those blocks.
  - **Equidistant angles,** the seven-patch ball's `c_j = 2ξ_j/n − 1`,
    over equiangular ones: the inner shells stay the seven-patch ball's
    bit for bit, and the inner–outer glue is the identity. Equiangular
    angles would even the outer layer's tangential spacing, but the inner
    layer would have to blend to an equiangular sphere.

  The seven-patch ball's shell map, glue and cube refinement were
  factored into helpers (`blended_shell`, `radial_glue!`,
  `quarter_turns!`, `refine_cube!`, `check_ball`) that both balls call,
  with the seven-patch ball's arithmetic and glue order unchanged.
  `sevenpatch_errors` and `radiative_errors` take `Rm`, and
  `bin/sevenpatch.jl` and `bin/sphere_waves.jl` take `--Rm`.

  Measured the same day on the laptop: on the outer shells the factored
  truncation costs 3.7 ns per node against the per-element tables'
  9.7, as `PLAN.md`'s operation count had it, with tables 80 times
  smaller. The plane wave converges at order `P` with errors within
  10 % of the seven-patch ball's on the same nodes. The driver's first
  middle sphere, `Rm = 2Lc`, took 2.1 times the seven-patch ball's steps:
  the inner shells between `√3 Lc` and `2Lc` are thin beside the cube's
  corners, a limit the seven-patch ball has at `R1 = 2Lc` too. So
  `Rm ≈ 3Lc` is where the outer layer costs no step, and `PLAN.md`'s
  hope that the outer layer would remove the corners' shear holds for
  the outer layer only.
- 2026-11-11: M8b's runs at M3b's scale (`bin/symmetry/nested_ball.sh`,
  Symmetry jobs 594441, 594442 and 594995), the pulse at `λ = 16` and 4
  through three nested cubes in the ball to `t = 96`. The first two used
  `hshell = 2`; at `λ = 4` they showed the shells, at two radial elements
  per wavelength, keeping a fifth of the energy, as M8a had found on the
  uniform ball, and the owner asked for the third at `hshell = 1`, which
  keeps `1.3e-3`. Observations every 0.5 had taken about half the wall
  time at `λ = 4`, so the third observed every 1. See "The seven-patch
  ball" under "Measured results" in `CODE.md`.
- 2026-11-11: what the cube–shell face sends back, at Erik Schnetter's
  request, after M8b's runs showed it to be the largest error in the
  finest ball at `λ = 16`: one uniform cube with shells of radial
  spacing 8:1 to 1:1 against the cube's, beside nested boxes (Symmetry
  jobs 595817–595821, with `--radii` added to both pulse scripts). At
  2:1 the face sends back what the boxes' flat 2:1 face does, and at 1:1
  below `1e-6`, so the cause is the jump in radial resolution, not the
  bend of the shells' radial lines or the glue. Graded radial spacing in
  the shells was proposed in `CODE.md` ("Possible extensions").
- 2026-11-11: graded shells, at Erik Schnetter's request after the face
  study: `sevenpatch_forest(…; grading)` takes the shells' radial
  coordinate through `s(η) = (e^{αη} − 1)/(e^α − 1)`, each element a fixed
  ratio wider than the one inside it, and `bin/nested_ball.jl --hface`
  grades from the cube's width to `--hshell` at the sphere. The proposal
  had expected 1:1 at the face for about a third more shell nodes. The
  same study on four graded layouts (Symmetry jobs 595921–595924) showed
  that grading does remove the face's jump, and the widening itself
  sends back little, but that what stays behind afterwards is set by the
  outermost elements, which need about 16 per wavelength; so at equal
  nodes uniform shells at 2:1 do as well or better, and grading to 1:1
  saves about a fifth of the nodes of uniform 1:1 shells. The map stays,
  uniform by default; the proposal was replaced by the measurement in
  "What the cube–shell face sends back".

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
- 2026-11-10: Erik Schnetter asked what would most improve performance on
  Symmetry for a thirteen-patch ball, with multi-thread benchmarks and a
  GPU in view. A first estimate sized the ball for a cube of `[−10, 10]³`
  and `R1 = 100` with 8 elements per wavelength at `λ = 16` on the sphere,
  which takes 96 elements across each patch, since every layer shares
  the cube's angular grid: `2.4·10⁸` nodes and about 5 s a step on a
  Symmetry node. It ranked the inner shells' radial width first (wider
  elements lift the step from 0.15 to 0.44 of the cube's, five times on
  the run), then a GPU, then the shells' kernels. He then said that the
  physical runs keep a constant angular resolution everywhere but near
  the centre, for example 16 elements per patch, and that the cube
  follows from `hₑ = 1/2`, with `Rm/Rb = 3`. That ball, the benchmark
  ball, has `1.91·10⁷` nodes and dropped the first estimate: there the
  inner shells are 8 % of the nodes, and a GPU and the outer shells'
  truncation and Laplacian lead (`PLAN.md`, "The benchmark ball").
  `bin/ball_benchmark.jl` and `bin/symmetry/ball_benchmark.sh` measure
  it; job 595219 on `cn084` gave 0.31 s a step at 64 threads, 3.8 times
  the flat box per node (`CODE.md`, "Threads on Symmetry").
- 2026-11-10: Erik Schnetter asked for the package to run on GPUs, and
  what that means for the particles. A survey found the flat path nearly
  ready (TreeAMR runs element sets, glues and folds on a device), a
  geometry refused on one because its truncation tables were arrays in
  structs no kernel could take, and the particles' projection mostly
  host code: of its three rules only the far one is a kernel. At `d/8`
  the projection is about 85 % of an inspiral step on 64 cores, and an
  H200 job gets 12 to 16 host cores, so leaving it on the host would make
  it about four times slower than on a CPU node. He chose the particles
  fully on the device, and, for now, the field ported with the far rule
  measured as a gate, the particles' port written into `PLAN.md`
  ("Particles on a device") rather than done.
- 2026-11-10: the field runs on a device. The tables and the coefficients
  are built on the host and copied, the coefficients' fill so that
  ForwardDiff's frame is never compiled for a device; the truncation
  structs have `Adapt` rules; a full geometry's folds read `√g` from the
  coefficient set, after the fold that evaluated the frame per point
  crashed Metal's compiler; the shells' faces are arithmetic rather than
  an index into a global tuple; a `Float32` geometry's tables are
  rounded once from `Float64x2` through `BigFloat`, which before failed
  on the CPU too, `Float64x2` converting to `Float64` only. The ball
  drivers take a `backend`; `InspiralProblem` refuses a device.
  `test/device_tests.jl` (after TreeHydro's) holds a device to the host,
  passing on the laptop's GPU in `Float32` and on an H200 in `Float64`
  and `Float32`; `bin/backend.jl` gives both benchmarks `--backend` and
  `--type`, and `bin/symmetry/gpu.sh` runs the lot on one H200.
- 2026-11-10: measured on one H200 (job 595237, `cn113`): the benchmark
  ball steps in 54 ms, 5.7 times faster than 64 cores; the flat box only
  2.6 times, its Laplacian pass no faster than on the CPU. The far rule
  runs 2.4 times faster than on 64 cores, short of the ten times the
  gate asked for, so the per-point evaluation comes before the rest of
  the particles' port. Tables in `CODE.md`, "Threads on Symmetry".
- 2026-11-10: Erik Schnetter asked to work on the particles' source
  kernel first. A probe of the source on an H200 (`bin/source_probe.jl`)
  found every variant at 255 registers and 35 to 52 KB of local memory
  per thread, at order 0 too: the 12.9 KB of each particle's coefficient
  tables, a kernel argument, copied into every work item. Inlining the
  call changed nothing, and the source in a one-element device array was
  worse. The tables now go into a device array of their own
  (`device_source`, `TableRef`), read in place, which made the source 2.7
  to 5 times faster there; register caps and block sizes moved it by at
  most 7 %. Timing the far rule's two kernels apart showed the
  contraction negligible and the launch over the ball's blocks costly, so
  a device now runs it over a list of the far elements. The far rule went
  from 2.4 to 7.0 times 64 cores, short of the gate's ten; the choice of
  how to go on is with the owner (`PLAN.md`, B0).
- 2026-11-11: Erik Schnetter chose to reshape the generated order-3
  code. Julia had inlined neither `evalpoly` nor products of four or more
  factors on the nested dual numbers: the order-3 term made 78 calls,
  each passing a tuple through memory, on the CPU as on the device, where
  they were most of 24 KB of local memory per work item. `generate.wl`
  now writes each group's polynomial as a `muladd` chain and nests its
  products, the same operations, so the host's results are the same to
  the bit in `Float64` and `BigFloat` (checked over 120 000 points at
  orders 1–3, and by the suite). Doing the same to the near field's
  radial functions moved results by an ulp, so they keep `evalpoly`. On
  an H200 the source at order 3 runs twice as fast and the far rule in
  0.0546 s; on 64 cores the projection costs 31 % less, and the far rule
  0.506 s, so the H200 is 9.3 times 64 cores of the same code. A
  register cap of 128 now saves a fifth on the device, where before it
  saved at most 7 %.
- 2026-11-11: Erik Schnetter asked for B1. The particles' tail is copied
  in one piece, the ball's passes and the averaging over shared nodes
  are kernels (the averaging brought forward from B5, since without it
  only `continuous = false` would have run), and, rather than a switch
  that ran the far rule alone, the graded and nested rules keep their
  elements' values on the host for one kernel to add on the device, so
  that the whole right-hand side runs there and `InspiralProblem`
  accepts a device. The host's results are the same as before to the
  bit; on an H200 a `d/8` right-hand side takes 2.04 s against 5.19 s
  on eight host threads, nearly all of it the host's two rules, and the
  plan per call 1.4 ms, which leaves B2 little to do.
- 2026-11-11: Erik Schnetter asked for B3. On a device the graded rule
  now runs as the far rule does, per band over its list of elements, the
  source at every point and then the contraction; since the host's
  recursive `tensor_rule` takes the same points, products and order, the
  kernels give its result to the bit on the CPU, which keeps
  `tensor_rule`. On an H200 at `d/8` the graded rule takes 20 ms instead
  of 1.11 s on eight host threads, and a right-hand side 0.93 s instead
  of 2.04, nearly all of it the nested rule on the host.
- 2026-11-11: Erik Schnetter asked for B4. On a device the host lists the
  nested rule's points with their weights and the device evaluates and
  sums them. Listing them first into a vector per point allocated 200 MB
  at `d/8` and took 117 ms on an H200 node's eight threads, mostly
  garbage collection, so it counts and then writes into preallocated
  arrays, 18 ms; a first version that concatenated the vectors with
  `reduce(vcat; init)` took seconds, and one that named the result like
  the threaded loop's locals raced. A `d/8` right-hand side takes 0.102 s
  on the H200, about 7.4 times 64 cores, equal to the host's.
- 2026-11-11: Erik Schnetter asked for B5. The pairing stays on the host
  and reads copies of the blocks around each particle from a device, so
  `particle_forces` takes a device state and the inspiral runs there
  entire. A `d/8` RK4 step takes 0.41 s on an H200, 50 times the node's
  eight host threads; reading the forces adds 35 ms.
- 2026-11-11: Erik Schnetter asked to try a register cap on the source
  passes. A prototype of the far rule's source pass as a plain `@cuda`
  kernel showed half the gain without a cap: KernelAbstractions'
  multi-dimensional index cost a tenth of the pass. So the three source
  passes now launch over one linear index, 7 % off a right-hand side
  and a step. The cap itself went into a package extension on CUDA,
  `@cuda maxregs=128`, another tenth, as its own commit so that it can
  be dropped: whether its gain is worth the extension is the owner's
  call.
- 2026-11-15: Erik Schnetter asked how to make M8d's runs faster, and
  for a profile first (job 598940): a step was the wave operator, 41 %,
  the projection, 34 %, and the truncation, 23 %, and the host's fill of
  the reference cost about as much as the steps. Half the projection
  was its plan's pass over the forest, which B2 had left as a host loop
  at `d/8`'s 1.4 ms. He asked for both fixes proposed. The pass is now
  built once per problem (`SourceBall`), and a step takes 0.83 s
  instead of about 1.03 (job 599012). The reference's fix, to fill
  `Φ_ret` alone where the window vanishes, was written and dropped: the
  data there are `(1 − W)Ψ̂` of the extended particle plus
  `w(Φ_ret − Ψ̂)` of the point model, and the two particle fields differ
  far out by about `5·10⁻¹³`, so it moved the data and the reference by
  about `10⁻⁹` of the field. The exact ways out are in `PLAN.md`, B6.
- 2026-11-15: of B6's two exact ways, Erik Schnetter chose the device
  fill as the simpler, keeping the overlap of the fill with the steps as
  a possible future improvement. `inspiral_initial_state` now fills in
  the problem's own field set on every backend, where it had filled a
  host set and copied it on a device; what kept the callback off the
  device were two error messages built by interpolation, the order's
  check, now made once when the callback is made, and the retarded
  time's, now a `NaN` the fill looks for afterwards. On an H200 the
  reference takes 2.6 s instead of 124 s and equals the host's to
  `1.2·10⁻¹⁵` (job 599075).
- 2026-11-15: once both fixes had merged (pull request #17), Erik
  Schnetter asked for M8d's retarded start again to time it: 5 h 37 min
  instead of 11 h 11 min, 0.80 s a step, the force the first run's to
  `2.5·10⁻¹¹ F_RR` (job 599116). The estimate had been about 5.5 hours.
- 2026-11-16: Erik Schnetter asked that the code, the particles
  included, work multi-threaded, on GPUs and over MPI. Threads and
  devices were in place; nothing had run over MPI. An audit against
  TreeAMR's "What an application must make global itself" found what a
  distributed forest breaks here: the time step's speeds and the
  radiative sphere's boundary energy were host reductions over a rank's
  blocks, the transfer read `locate_point`'s global leaf index as a
  local block (the supports, their neighbours, the averaging's faces,
  `reconstruct`), the source's averaging paired nodes across faces two
  ranks share, and the ball's check, the two-supports refusal and the
  retarded data's `NaN` check were decided from a rank's own blocks.
  Following TreeHydro's `distributed.jl`, `src/distributed.jl` adds
  `rank_reduce`, `agree_refusal` and `gather_blocks`; the pairing reads
  gathered copies of the support's blocks on every rank, as it already
  did from a device; and `average_shared!` exchanges the faces of remote
  pairs point to point, writing the serial means. The forest builders
  and the drivers that build forests take `comm`. `test/mpi_tests.jl`,
  after TreeHydro's, runs a workload on three ranks of three, three and
  one threads and finds every distributed line the serial one-thread
  line bit for bit, the particles' forces included, and the sums to
  roundoff. On the laptop the device tests pass on Metal, and the field
  on Metal over two ranks gave one process's state bit for bit. He asked
  for the suite on Symmetry, and whether a multi-GPU job ran too; one
  was added (`bin/distributed_gpu.jl`): on two H200s the field and a
  particle on nested boxes gave one H200's state and forces bit for bit
  (job 599625), and the CUDA device tests passed there. The suite, the
  MPI test included, passed three ways: plain on Symmetry (job 599630,
  52 minutes), with `--threads=4` there (599631, 24 minutes) and with
  `--check-bounds=yes` on Nibi (27520764, 46 minutes), 9772 passing and
  the one known broken test.

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
- 2026-11-11: Erik Schnetter asked, before M8c, whether M8a's and M8b's
  accuracy should be improved first. The review, from the recorded
  runs and nothing rerun: inside the cube the ball is the nested boxes;
  the cube–shell faces send back about `4·10⁻⁵` of the pulse at
  `λ = 16` with shells at 4:1, the boxes' level at 2:1; the sphere sends
  back `2·10⁻⁴` of the monopole and `10⁻³` of the centred dipole at
  `R₁ = 50`, from `t ≈ 2R₁` on; with radial elements at four per
  wavelength the shells let `λ = 4` out but for `1.3·10⁻³` of its
  energy. Estimated, not measured: an incoming wave of fractional
  amplitude `ε` moves the force by about `ε F_RR`, against M4's force
  error at `d/8` and `d/16` of 0.7% and 0.2% in the mean. So the
  face's return is far below the force's own error, and the sphere's
  dipole return is about `d/16`'s, but comes after `2R₁ − R₂`, beyond
  M7's first five orbits at `R₁ = 50`. M8a's gate fires as written, the
  sphere sending back about eight times what the boxes' face at 32 does,
  a third later; the owner passed it, with the layout as the answer:
  the shells' radial elements at most twice the cube's root width, and
  `R₁` such that the sphere's return comes after the run. Not measured
  and left to M8c's runs: the off-centre dipole, a dipole through the
  refined ball, `λ = 16` with `hshell = 1` at M3b's scale, the
  thirteen-patch ball's dipole and larger `R₁`, and the near zone's
  static multipoles against Sommerfeld's condition.
- 2026-11-11: M8c started on branch `inspiral-m8c-ball`, at Erik
  Schnetter's request. TreeAMR 0.2.9 had the location in a patch that
  the plan had asked for upstream, `locate_point(forest,
  PatchPoint(ξ, p))`. The plan's "transfer in physical coordinates" is a
  `ParticleFrame`, the patch and its affine map `x = origin + scale ⋅ ξ`
  (`affine_map(geo, p)`, the balls' cube), through which the transfer
  reads every block origin, spacing and extent and locates every point;
  on a brick it is the identity and changes no bit. Where the plan had
  the pairing's integrals times `a³` and the gradients times `1/a`, the
  pairing integrates over the mapped, physical element boxes and needs
  neither. Added beyond the plan: `ball_mean`, the signed mean, for the
  drift of `Π_r`'s mean; a refusal of a ball or a support that leaves
  the cube, ahead of the finest-level check; and `--ball` in
  `bin/uniform_motion.jl` and `bin/prescribed.jl` through `bin/ball.jl`.
  The suite's ball first had its sphere at `2Lc`, where the shells beside
  the cube's corners set the step, 0.6 of the boxes'; at `3Lc` the
  finest cube sets it and the ball gives the boxes' error to `10⁻¹¹`
  and force to `10⁻¹⁴` (`CODE.md`, "Particles in the ball").
- 2026-11-12: M8c's first job on Symmetry (596438, `cn107`), after
  Erik Schnetter gave permission to run there, failed at the end of its
  first run: `uniform_motion.jl --cost` summed the gradient sets, empty
  on a full geometry, whose gradient is `cartgrad`; the three jobs
  behind it were held and cancelled. Its one run, `d/4` to `t = 0.5`,
  was set against M3b's (587559) and differed by 10% in the error and
  60% in the force; the remainder's norm differed at `t = 0` already.
  Two causes. M3b's runs predate M4's averaging and M5's lab slice,
  which change the remainder, so they are no reference, and
  `ball_motion.sh` now runs the boxes of the cube's three levels beside
  the ball with the current code. And the cube's map, computed as
  `Lc(2ξ/n − 1)`, rounds `2ξ/n` for `n = 12`, which moved the nodes lying
  exactly on the window's spheres, `R₁ = 1.25` and `R₂ = 2.5`, in or out
  of the norms' balls: `err_R1` by 0.6%, the mean of `Π_r` by 2%. Both
  balls now compute the cube as the frame does, `(2Lc/n)ξ − Lc`
  (`cube_position`), and on the laptop the ball gives the three-level
  boxes' numbers at `d/4` to twelve digits and more, the remainder at
  `t = 0` bit for bit. The 2% difference in the mean recorded the day
  before had been this rounding, not the field. Resubmitted as 596453
  (the series, ball and boxes), 596454 (at rest, `R₁ = 36` and the
  boxes), 596455 (at rest, `R₁ = 50`) and 596456 (the circle at
  `v = 0.2`, `d/8`, order 3, `R₁ = 50`).
- 2026-11-12: M8c's runs finished (596453 to 596456; `CODE.md`,
  "Particles in the ball" under "Measured results"). Inside the boxes'
  clean time the ball gives their numbers to the digits printed: the
  series at `d/4` and `d/8` to eleven digits and more, and M5's circle
  at `d/8`, order 3, in the plan's ball at `R₁ = 50`, every step's force
  to `1.3·10⁻⁸ F_RR`. At `d/12` the norms agree to four digits only,
  since the step `2/3` is not dyadic and the boxes' node positions and the
  cube's round differently on the norms' spheres; the force agrees to
  eleven. The particle at rest to `t = 60` stays clean in the ball at
  `R₁ = 36` and 50 alike, its error at `2`–`3·10⁻⁷` and the mean of
  `Π_r` below `10⁻⁹`, while the boxes' error grows 30 000-fold and their
  mean drifts to `−3·10⁻⁴`. The sphere's return shows at about
  `(2R₁ − R₂)/2`, `t ≈ 35` and 49, half the light-crossing time: the
  remainder's under-resolved error crosses the shells at about twice the
  speed of light. It raised the error by 40% at `R₁ = 36` and by 15% at
  50. A first reading had put the return at `2R₁ − R₂`, after both runs,
  and taken the bumps for scatter; Erik Schnetter then asked for the
  results to be read as what the transitions and the sphere send back
  before the light-crossing time, and the bumps' times matched.
  Not run: uniform motion past the clean time, since at `v = 0.2` the
  particle leaves the finest cube of `w0 = 3` at `t ≈ 12`.
- 2026-11-13: Erik Schnetter asked for the runs to be read by what the
  transitions send back before the light-crossing time, and the
  resolutions that control it, then for those scans to run on GPUs.
  `bin/nested_ball.jl`, `bin/prescribed.jl` and `bin/uniform_motion.jl`
  took `--backend`, a device test the particles in the ball, and
  `bin/symmetry/gpu_scans.sh` the scans: the pulse against the radial
  jump at the cube's face and the wavelength, the circle against the
  shells' width, and the particle at rest at a third sphere. The pulse
  scan fills in what M8a and M8b measured at single points (`10⁻³` at a
  2:1 face resolved at 8 elements per wavelength, below `10⁻⁶` at 1:1).
  Submitted the same day: the device tests 597907 on `h200debugq`, and
  behind them on `h200q` the pulses 597908, the circle at shells of 0.5,
  1 and 2 597909 to 597911, and the particle at rest at `R₁ = 72`
  597912.
- 2026-11-13: the scans finished on H200s of `cn111` and `cn113`, after
  waiting some hours for the group's GPUs (`CODE.md`, "What the
  transitions send back" under "Particles in the ball" in "Measured
  results"). The pulse separates what M8a and M8b had varied together:
  at equal radial elements per wavelength in the shells, a 1:1, 2:1 or
  4:1 jump at the cube's face sends back within a factor of two of the
  same, and going from 4 to 8 per wavelength lowers it about 300 times;
  so M8b's "the reflection is the jump" became "the coarse side's
  resolution", and the layout's rule for the shells, at most twice the
  cube's root width, became about 8 radial elements per wavelength of
  the shortest wave carried out, at every radius. The circle at
  `v = 0.2` agrees between shells of width 1 and 0.5 to `10⁻⁵ F_RR` up to
  `t = 28`; width 1.83 moves the force by `4·10⁻⁴ F_RR`. From `t ≈ 25`
  every circle was swamped by what its initial data beyond the retarded
  window sent in, the particle field of order 3, whose near-field terms
  grow like `r²` and are a hundred times the field at `r = 48`. The
  script had placed the window's edge at 48 so that its signal would
  arrive after `t = 40`; it arrived at about twice the speed of light
  through the shells, as runs on the laptop that moved the blend and
  timed the front at six radii confirmed. Initial data that are the
  field far from an accelerated particle became an open item. The
  particle at rest at `R₁ = 72` shows no sphere's return above `10⁻⁸`.
  On the H200 the field takes 3.9 ns per node and step, and the
  particles' runs about a sixth of a 64-core EPYC node's time per node.
- 2026-11-14: Erik Schnetter closed M8c and asked for a milestone M8d,
  designed with him, for what it left open. The plan's estimate of the
  shells' cost had kept their tangential spacing at most 2 out to `R₁`,
  so that their nodes grew like `R₁³`; M8c's shells keep the cube's
  angular resolution instead, which is what an outgoing wave needs, and
  grow like `R₁`, and the plan now says so. He set `R₁ = 100` for M8d,
  the sphere causally disconnected from what is measured: what it sends
  back after the run does not matter, so nothing is scanned in `R₁`, and
  the scans of the transitions' resolution are mostly settled by M8c.
  The data: the retarded field out to the sphere for the circle, and,
  at his suggestion, a simpler start that switches radiation reaction on
  at once, for comparison. The first form proposed, a zero remainder,
  is the particle field inside the window and nothing beyond `R₂`; he
  pointed out that the charge is then seen near the particle at
  `t = 0`, which showed that it is not seen beyond `R₂`, and that what
  should start at `t = 0` is only what the trajectory's acceleration
  changes. That is uniform motion's field outside the window, the
  boosted static field `Ψ̂₀` with its rate along the tangent line, which
  he chose. Dropped: the zero remainder, whose front would carry the
  whole Coulomb monopole out; `Ψ̂₀` with `particle_rate`'s `𝒟Ψ̂₀`,
  which carries `a` through `v̇` and is not an exact solution; retarded
  far data for hyperbolic motion, whose retarded field ends `1/g`
  behind the particle and whose runs are short. The dispersion relation
  that would explain the error faster than light went into M8d, and a
  filter damping it, should it be the cause, is the owner's decision.
- 2026-11-14: M8d started once M8c's pull request had merged. Uniform
  motion's far data became `inspiral_initial_state(p; far = :uniform)`,
  which at `a = 0` is the default's data bit for bit, and
  `bin/prescribed.jl --data=uniform`. On a coarse ball the uniform
  start's remainder beyond the window is a sixth of the order-3 particle
  field's, and its `Π_r` a ninetieth. The retarded reference costs
  2.7 µs per node on one laptop core, about 8 core-minutes per output at
  `R₁ = 100`. `bin/dispersion.jl` took the dispersion relation from the
  operator itself: with truncation the fastest group velocity is 1.58,
  on the physical branch's unfolded, under-resolved part, in every
  direction since the operator is the sum of the one-dimensional ones;
  without it a spurious branch carries long waves at 3. Ahead of the
  group velocity a dispersive tail reaches thresholds of `10⁻⁶` to
  `10⁻¹⁰` at 1.7 to 2.5 times the speed of light, which accounts for
  M3's twice it and M8c's 1.7 to 1.9 in the shells; the mechanism is no
  longer open.
- 2026-11-15: M8d's two circles ran on H200s overnight (jobs 598636
  and 598637, 11.2 and 12.4 hours). From the retarded field everywhere
  the circle runs clean to `t = 90`, its force over each orbit
  `1.04163 F_RR` to `6·10⁻⁵`, and it is M8c's circle until M8c's
  particle field beyond 48 reaches the particle, at `10⁻⁸ F_RR` by
  `t ≈ 10`, the dispersive tail seen directly. From uniform motion's
  field far out the force comes to the retarded start's after three
  time units and agrees with it over each orbit to `6·10⁻⁴ F_RR`, but
  from `t ≈ 10` on carries an excess of RMS `1.5·10⁻² F_RR`, near
  `ω ≈ 7.5`, that does not decay; the data's front leaves each ball at
  the speed of light and leaves a residual of about `10⁻⁶` behind. Its
  cause is open: the dispersion relation's group velocity at that
  frequency is above 1 on the cube's root and in the shells, so it is
  not a trapped mode. Filling the retarded reference on the host cost
  about as much as the steps. Prescribed paths take the retarded field
  to the sphere; the inspiral's choice, and whether to filter the fast
  modes, which had been M8d's, are M7's.
- 2026-11-15: Erik Schnetter asked for M8d's convergence test, which
  had been proposed when he asked whether its runs had one: both starts
  at `d/4` and `d/6` beside `d/8`, the shells refined with the cube
  (jobs 599121 to 599124, one H200 each, `bin/symmetry/gpu_scans.sh`'s
  new `HDIV`). The retarded start converges at order 3.6 to 4.2 in its
  errors and about 3 in its force's scatter. The uniform start's
  residual in the balls of 32 and 64 does not move with the resolution:
  it is the two starts' solutions' difference, not the scheme's error.
  Its excess in the force, whose cause
  M8d had left open, falls in every band of frequency from `d/6` to
  `d/8`, at about second order against the retarded start's third, and
  its peak moves from about 5.6 at `d/4` to 7.5 at `d/8`: an error of the
  scheme, which the uniform data excite twenty to thirty times more
  than the retarded data; what in them does is still open. He asked for
  `d/12` if it fit one H200: at 401 bytes per node on the device it
  needs about 232 GiB, and `d/10` 137 of the 140, so neither ran.
- 2026-11-16: looking at M8d's results, Erik Schnetter asked whether its
  resolutions hold several orbits, whether its waves are clean of what
  the grid's transitions, the sphere and error ahead of light send, and
  whether its two starts differ; the balls' norms answered in part, and
  he asked for waveforms, designed first as a milestone, M8e, with the
  extraction tested on its own before any run. The modes are read from
  the elements' own polynomials at points located once, TreeAMR's
  `ElementPolynomial`, rather than from an interpolation through the
  nodes of several elements, which would have read the ghosts and their
  coarse-fine error; the seven-patch ball's map is inverted in closed
  form, since a shell keeps each ray through its face, rather than by
  Newton's method or TreeAMR's atlas, which would run the inverse in
  every launch. The reference's modes are the same rule at the same
  points, so the rule's aliasing does not enter the error.
- 2026-11-16: on CUDA (job 599405) TreeAMR 0.2.9's kernel for located
  `PatchPoint`s read from a state vector did not compile, a `typejoin`
  that allocates; the host and the CPU stand-in had passed. Rather than
  wait for TreeAMR, the extraction takes each point's element and
  weights from TreeAMR on the host and reads the state on the device by
  a gather and a weighted sum, which the CPU holds to TreeAMR's
  `interpolate` to `10⁻¹³`. The kernel is TreeAMR's to fix.
- 2026-11-16: Erik Schnetter asked about that bug, and for its fix:
  `element_series` captured a reassigned variable in a closure, which
  boxed it and made the weights `Any`. TreeAMR's pull request 42 fixes
  it, with tests that the weights are inferred and that the element
  polynomial agrees with the host on a device; until a release has it
  the extraction keeps its gather.
- 2026-11-16: M8e's six runs (jobs 599424 to 599429, one H200 each)
  answered the three questions, and M8e was closed. At M8d's
  resolutions the retarded start's waves converge at order 4 to 6 at
  every radius, clean of anything from the transitions and of anything
  ahead of light, until the sphere's reflection of the dipole wave,
  `4·10⁻⁴` of it, arrives at `t = 100 − r`; their phase error grows with
  the distance travelled, not with the orbits. The uniform start's
  reference was taken anew, the retarded field of a charge on the
  tangent line until `t = 0` and on the circle since, which is the
  uniform data's solution in all space: against it the uniform start
  sends a pulse of 17 % of the dipole wave out of the window, leaves a
  wake near the centre that falls at about second order, and receives
  from the sphere, from `t = 100 − r` on, a static dipole that grows to
  22 % of the dipole wave at `r = 90` by `t = 90`, the same at every
  resolution. That is M8d's far residual, which *Its convergence* had
  read as the two starts' solutions' difference; that difference
  vanishes where `t > r`. M8d's guess, that the sphere sends it back,
  holds.
- 2026-11-16: Erik Schnetter asked whether a uniform motion can go
  smoothly into a circular one, how many derivatives of `a` must then
  vanish, and whether radiation reaction could be switched on smoothly
  so that a retarded start stays nearly right; he asked for the test
  designed as a milestone first, M8f. Its assumed past is the circle
  spun up from rest, not the moving tangent line, because the line's
  displaced charge is what the sphere sent back in M8e and because a
  curve of constant speed and ramped curvature has no closed form; its
  ramps are polynomial smoothsteps of order `n`, not a `C^∞` step,
  because they keep every jet closed for the device's root-finding, and
  `n` is what the runs vary. Switching radiation reaction on is posed as
  the force taking over from whatever held the assumed orbit, and tested
  as a prescribed departure from the circle at `t = 0`.
- 2026-11-16: M8f built on branch `inspiral-m8f-smooth-past`
  (`CODE.md`, "Assumed pasts"). The smoothstep is evaluated above
  `x = 1/2` from its mirror image: from the coefficients directly, its
  first derivative at `1 − 10⁻⁹` was `2·10⁻¹¹` at `n = 7`, the
  roundoff of coefficients up to 10⁴, where it should be a power of the
  distance. Both steps are continuous from the right, so that an abrupt
  departure's `a(0)` is the one the evolution sees from the first step,
  and the initial remainder holds the jump as the run does. The
  ramped circle's jets are written by Leibniz's rule on `r e^{iφ}` for
  every piece rather than per piece by hand; on `[−T, 0)` it calls the
  circle itself, so that M8d's path is kept bit for bit. M8e's analysis
  scripts became `bin/waveforms.jl`, which gives M8e's clean-window
  tables digit for digit, its uniform start's reference, recomputed,
  equal to the one M8e used. The window's step in `src/particle.jl`,
  also called `smoothstep` until then, became `window_step`: the
  exported `smoothstep(n::Integer, x::Real)` was the more specific
  method and replaced it, clamped and allocating, which the window's
  tests caught.
- 2026-11-17: the past series ran overnight on the H200s after the
  branch was rebased on `main` with MPI (jobs 600171 to 600182, in an
  environment of its own, `env-m8f`, since `SCAN=tests` re-`develop`s
  the one it is given and `env-perf` served other checkouts). `n = 0`
  was not clean, and `n = 4` was as clean as the circle at `d/8`
  (`CODE.md`, "Assumed pasts" under "Measured results"). `n = 1` and 2
  were as good as 4 at `d/6` but were not run at `d/8`, so the
  departure series was submitted from the past of `n = 4`, as the plan
  said, from the past that the series found clean (jobs 601825 to
  601832). Contrary to the plan's expectation, the sphere's return was
  not the circle's own reflection but six times it: the static dipole of
  the charge at rest, which the sphere sees until `t = 60`.
- 2026-11-18: the departure series (jobs 601825 to 601832) closed M8f
  (`CODE.md`, "Assumed pasts" under "Measured results").
  - **The method.** It was analysed without the plan's `--reading`
    runs. From the ramp's end on, every `n₂` follows the same path, so
    the differences between runs at one resolution are error alone.
    Inside the ramp the force's difference between resolutions is
    dominated by face crossings at different times, which readings
    would have had to match step by step at thousands of times.
  - **Where the error appears.** The plan expected an abrupt start to
    show as a pulse at `t = 0`. There, the data take the jump, and
    only the field's own kink is left, converging at order 1. What
    showed was the ramp's end at `t = Δ₂`: a pulse from the window's
    shell that has the same size at every resolution, at `n₂ = 0` and
    at 2. The account that fits is an impulse in time in `□(WΨ̂)`,
    which the stages miss whenever `ä` or `a⃛` or below jumps during
    the run.
  - **What it means for the binary.** `n₂ = 4` over one orbit was
    clean, and it is what the binary's take-over takes. `n₂ = 0`
    also left a force error from `t ≈ 30` on at every resolution, half
    the circle's scatter at `d/8`, which was not traced further.
- 2026-11-18: Erik Schnetter asked for a movie of the system in the
  style of a binary black hole visualisation: the equatorial plane as a
  heatmap of a smooth field scaled by `r`, the particles on top, and a
  few `Y_lm` modes on a sphere. The start-up could show, and the outer
  boundary should not. He chose:
  - **two equal charges half an orbit apart** over one charge: their
    `ℓ = m = 2` spiral is the binary's picture, where one charge's is a
    one-armed `ℓ = m = 1` spiral;
  - **a past spun up from rest just before `t = 0`** over the circle
    forever or uniform motion's data: the start-up is then a physical
    front, behind which the spiral fills in, and which reaches the
    sphere only at the run's end.

  The run writes frames and a separate viewer turns them into a movie,
  since what to show needed trying out. A new driver,
  `bin/orbit_movie.jl`, took the place of a `--charges=2` for
  `bin/prescribed.jl`, whose one-particle diagnostics hold measured
  results (`CODE.md`, "The movie"). The exact frames
  (`bin/orbit_exact.jl`) came from the retarded field being known for a
  prescribed path. They let the viewer be tried before the run, and
  they show the run's error. The run on an H200 (jobs 605550 and
  605551) gave the exact field to 3 % of the picture's scale at worst,
  at the front (`CODE.md`, "The movie" under "Measured results"). On
  seeing the movie he asked:
  - **for the `plasma` colour scheme** in place of the diverging
    `balance`, which became the default;
  - **for a second version with the elements in grey**, which the
    viewer draws with `--grid` from the run's forest built again from
    its layout, the elements' faces only, not their nodes.

  The movies were kept out of the repository.
- 2026-11-18: Erik Schnetter asked for a review of M8 by a sub-agent.
  It found no error in the evolution, the radiative sphere, the cube's
  frame or the paths, and two in M8e's extraction:
  - **A sphere beyond a periodic brick read the field of its images.**
    TreeAMR's fold code marks mirrors and quarter turns, not periodic
    wraps, so `bin/prescribed.jl --waves` without `--ball` would have
    read wrapped field on its spheres at 70 and 90 at the default box
    of half-width 64. The extraction now refuses such a point, as the
    plane already left it out. Every run recorded used `--ball`.
  - **Over MPI the extraction took a forest's leaf for a rank's
    block.** No driver extracts on a distributed forest, so it now
    refuses one rather than gathering the leaves.

  It also found `CODE.md`'s table of positions and weights given as
  128 bytes per point at `P = 3`, where it is 1 KB, and stale comments
  in `test/device_tests.jl` and `src/evolution.jl`, now fixed.
- 2026-11-19: M6a started, a particle that moves under its forces
  (`CODE.md`, "The equations of motion"). Erik Schnetter chose two
  departures from the plan:
  - **The own near field's average by quadrature**, over the moving
    particle's support, rather than M1's at-rest coefficients, which miss
    the lab slice's `O(v²)` self-force. The quadrature then showed the
    order-1 force with its lab slice to be `−δm γ a` in every direction,
    the covariant renormalization of the mass.
  - **`a`, `ȧ` and `ä` solved together** by Newton's method, rather than
    in `INSPIRAL.md`'s order with each one's higher jets from the
    predictor. `E` reads `ȧ` with a coefficient of about `vδm`, so the cut
    would have put the predictor's error into the tangential force.

  Asked how the start compared with M8f's ways of starting a path
  smoothly, he had the force take over from the past as M8f's departure
  took over from the circle, by a blend of residuals with `n₂ = 4`, so
  that the data are exactly the past's and nothing jumps. The abrupt
  start, in which the data take the jump, stays as `onset = 0` for
  comparison.

  The first solve took 47 s on the laptop's coarse box. It was cut to
  about 1.3 s by three changes:
  - projecting the source near the particle once per residual, in
    parallel, and pairing every kernel with it;
  - holding that projection while Newton converges, since it moves the
    residual `10⁻⁴` as much as the jets;
  - starting from the last solve's jets.

  The projection near the particle had to copy the projection's choice
  of rule: a graded rule one element out differs from the nested one by
  `2·10⁻⁷`.

  The first series job (606216, `d/4`) ran out of its hour at
  `t = 1.28` of 8: a step took 12 to 24 s on 64 threads, 97% of it in
  the solve's source projections. Its four dependent jobs never started
  and were cancelled. A profile (job 606310) found three causes, and the
  step at `d/4` fell from 16.9 s to 2.34 s (job 606323):
  - the projection gave each thread one of 27 elements, while the
    support's own element cost more than the rest together, so 2.04 s on
    64 threads against 1.24 s on the laptop's four; threaded over the
    points of the outermost line, as `project_near_host!` is, 0.064 s;
  - `Ṡ` was projected as a dual number and `S` again by itself, though
    the dual's values are `S`'s;
  - `source_tol = 10⁻⁶` found the source twice per solve, since the
    first guess is up to 9% from the result while the force takes over;
    never finding it again moves the force by `3·10⁻⁸ F_RR`, so
    `source_tol = 0.1`.

  `bin/dynamic.jl` also stopped solving again to report, which the
  radial direction toward the circle's centre makes unnecessary.

  The resubmitted series (jobs 606324–606328) recorded `cond_aa = 1`
  and `δm_h/M = 1.252·10⁻³` at every step and resolution with
  `onset = 2`: that is `γ - 1`, the Jacobian of the first step's blend,
  where the past still held the jets. The chord iteration converges with
  it fast enough that the solve never found it again; the jets were
  right, at a median of four iterations against three with `onset = 0`,
  whose Jacobian, found after the take-over, recorded `δm_h/M ≈ 10⁻⁸`
  (below, 2026-11-20). The
  solve now finds the Jacobian again when the blend has moved by a
  quarter and when the force has taken over. The runs already in the
  queue kept the old rule.
- 2026-11-20: M6a closed (`CODE.md`, "The equations of motion" under
  "Measured results"). The series finished but for the spring at
  `d/12`, which ran out of its eight hours at `t = 6.9`; Erik Schnetter
  let it, rather than run it again for sixteen. The spring reads the
  prescribed circle's force at every resolution, the two parting by
  `0.068 F_RR` rms at `d/4` and `0.002` at `d/12`. The gap of several
  `F_RR` seen at `d/4` before the series came from comparing against
  M4's records of the circle, which predate M5, so the series ran the
  circle again at every resolution on the same code.

  The conditioning was measured again on the code that finds the
  Jacobian after the hand-over, with `onset = 0` to `t = 0.25` (jobs
  606769 and 606770): `δm_h/M = 7·10⁻¹⁰` at every resolution from `d/4`
  to `d/12`, and a condition number of 1, so the plan's gate for M6a, an
  ill-conditioned `ä` system, did not fire. The `δm_h/M ≈ 10⁻⁸` the
  series' `onset = 0` run recorded later in time is probably the kept
  Jacobian measured against a `γ` that has moved since, not the
  self-inertia; not checked.

  The cost showed what the profile of 2026-11-19 at `d/4` did not: the
  solve near the particle costs the same at every resolution, but `LΠ`,
  a whole `wave_rhs!` per stage, doubles the circle's step at `d/12`. A
  Laplacian of `Π_r` on the support's elements alone is proposed in
  `CODE.md`.
- 2026-11-20: M6b started, the binary's parts (`CODE.md`, "Initial
  data" and "Diagnostics"). Erik Schnetter settled the plan's questions:
  - **The ball and M8f's past**, not M3b's boxes released at once: the
    boxes' kinked far data and short clean time are what M8 removed.
  - **`v = 0.2` fixed and `G` solved** (0.08204), rather than
    `INSPIRAL.md`'s `G = 0.0804` and `Ω` solved, so that M8's grids and
    measured pasts carry over.
  - **The flux check on M8e's `v = 0.2` circle in the ball**, not on M4's
    `v = 0.05` boxes, the plan's text: the ball's data are the retarded
    field, clean from the start, and the flux's `O(v²)` departure from
    the dipole power is then large enough to see.
  - **`LΠ` on the support's elements alone deferred to M7.** Asked what
    it would take, he was told: gathering `Π_r` on the `5³` elements
    about the support and applying the element's Laplacian tables there,
    tested against `wave_rhs!`; a second copy of the Laplacian's
    arithmetic for about 2 s of a step estimated at 8–9 s.
  - **The quarter ball** (`x ≥ 0`, `z ≥ 0`, with a reflection and a
    half-turn), his suggestion, to be recommended if the runs proved very
    expensive; not for M6b, since the transfer would have to fold every
    read across both faces.

  Through M6a's session he asked that the runs watch for the
  instabilities in the higher derivatives that other self-force schemes
  report: every step the jets, the solved `ä`'s rate against the
  predicted `a⃛`, Newton's residual and a fresh Jacobian's `δm_h`, at
  two resolutions and half the step. He also had the large runs,
  test files included, go to a cluster rather than the laptop.

  Writing the tests found that opposite charges repel in the scalar
  theory and like ones attract; a comment in the predictor's tests said
  otherwise and was fixed. The binary's symmetry test first asked for
  `10⁻⁸` in every jet; `ȧ` and `ä` keep it to `10⁻¹⁰` and `10⁻⁸`
  because each time derivative reads the unsymmetric field through
  another of the grid's second differences, so the test's tolerances are
  per jet. The first flux test, of a field like `1/r`, had a singularity
  at a node; it takes `g(r²)` and `h(r²)`.

  The flux runs found that a sphere whose centre is not the circle's
  sees a flux that changes over the orbit, the energy inside it moving
  as the charge goes round, so `bin/flux.jl` compares means over whole
  orbits. On the centred sphere the retarded field's flux is
  `γ⁴Q²a²/12π` exactly.

  The first binary step at `d/4` took 9.2 s, the own and other fields'
  quadrature over the support serial on the host; threaded over chunks
  of the rule summed in order, the same numbers, it took 4.6 s. The runs
  at half the step and at `d/6`, which would have taken about 18 hours,
  were cancelled after their first hour and restarted on the threaded
  code at his request; the `d/4` run and the circle ran on.
- 2026-11-21: M6b's runs finished (`CODE.md`, "The flux" and "The
  binary" under "Measured results"). The flux converges to the
  retarded field's at about second order in the orbit's mean, so the
  plan's done-when for the flux holds. The binary's orbit is the same at
  `d/4`, `d/6` and half the step, its centre of mass drifting as the
  grid's error. Nothing grew that depends on the resolution or the step;
  the solved `ä` jumps by up to three times itself where the support
  straddles an element face, halving from `d/4` to `d/6`, and the
  binary's `δm_h/M` is `3.9·10⁻⁴`, about `q²/4π`, at every resolution,
  both recorded as open. `d/8` was not run: about 28 hours, past
  `amdq`'s 24, without a checkpoint.
- 2026-11-21: two experiments on the solved `ä`, at the owner's
  question whether there were two problems in it, a 1 % that does not
  converge and noise at the faces (`CODE.md`, "The binary" under
  "Measured results"). Particles of twice the size cut the jump at a
  face about as `ℓ⁻²` and left the departure away from the faces as it
  was, to the digits shown; so that departure is the grid's, falling
  from `d/4` to `d/6`, not the tail of the crossings, as it had seemed
  in the morning's reading of the runs. The source written with the predictor's
  `ä` left the solved `ä` and the orbit as they were, so the noise does
  not feed back. `Dynamic`'s `source_addot` and `bin/binary.jl`'s
  `--ell` and `--source-addot` stay as experiments. The full suite
  passed on Symmetry three ways on the same commit.
- 2026-11-21: the binary's `δm_h/M ≈ 3.9·10⁻⁴`, left open in the
  morning, traced (`CODE.md`, "The binary" under "Measured results"):
  it is the other charge's potential in the equation's mass
  `m = M₀ + QΦ̄ₑ`, `γQΦ̄_b = q²/(4πdγ)` at `d = 1`, which the residual's
  analytic parts alone give to five digits, and the cross block is
  `Ψ̂₂`'s order-3 near field. What is left for the grid is
  `1.5·10⁻⁷ M` typically. The recorded `δm_h` measures the grid's
  self-inertia only for a lone particle; subtracting `QΦ̄ₑ` for the
  binary was proposed, not done.
- 2026-11-21: the 1–2 % in the binary's solved `ä` away from the faces
  traced, from the runs' records (`CODE.md`, "The binary" under
  "Measured results"). It is a ringing of the grid's field at
  `ω hₑ ≈ 8` to 10, the same at half the step, which the force carries
  at about 1 % of `F_RR`, the prescribed circle's equally. `ȧ` reads it
  times `ω` and `ä` times `ω²`, so `ä`'s part falls little with `hₑ`
  while the force's falls by 2.4 from `d/4` to `d/6`. What rings the
  field stays open.
- 2026-11-21: the prescribed circle at `v = 0.1` and 0.05
  (`jobs/ringing.jl`, at the owner's request) put the ringing's band at
  `ω hₑ ≈ 8` to 9 at every speed, with the same amplitude, which rises at
  `t ≈ 4` and falls fourfold by `t = 24`. So it is a transient of the
  grid's own modes, not the moving source's projection, which would
  have moved the band as `v/hₑ`, and not driven by the motion. `ä`'s
  departure away from the faces follows it, 1.3 % to 0.55 % after the
  hand-over. That the transient is the initial data's error from the
  coarser levels is likely and was not checked.
- 2026-11-21: the owner asked for the two left over. The likely cause
  of the ringing was wrong: the circle with its coarse-fine faces at 8
  instead of 4 (`jobs/ringing.jl`'s `circle-w8`) rings as before, to a
  few percent window by window. With the window twice as wide,
  `R₁ = 2.5`, `R₂ = 5` (`--window`, `circle-w8-win`), the ringing is 30
  to 50 times less from `t = 0` on and the mean force the same to
  `3·10⁻⁴`: the grid answers the source in the window's shell at its own
  high frequencies. The wider window costs twice the step; whether the
  binary should take it is the owner's, and why the ringing grows from
  `t ≈ 4` and then falls stays open. `bin/binary.jl` now records the
  grid's part of `δm_h` alone (`dmhgrid<k>`), less the other charge's
  potential through the hand-over's blend; a `d/4` run to `t = 20` gives
  `1.7·10⁻⁷` after the hand-over, as the analysis of the old rows had.
- 2026-11-22: the movie's dynamic binary at `q² = 0.2`, `ℓ = 0.04`
  (job 607865) went unstable: its separation oscillated at `ω = 7.7`,
  growing 1.3 per unit time from `t ≈ 9`, until the solve failed at
  `t ≈ 16`. The movie's session asked for the cause before any fix. Its
  guess, a grid feedback `∝ q²/(M hₑ)`, did not hold: the grid's part
  of `δm_h` is `10⁻⁵` at `d/4`. `jobs/instability.jl` scanned the
  charge, `ℓ`, `hₑ`, the step, the window and the near field's order
  (`CODE.md`, "The breathing instability" under "Measured results").
  `ω ≈ 8` throughout, and the growth depends only on the charge: 1.4,
  0.5 and none at `q² = 0.2`, 0.05 and 0.0125. Orders 1 and 2 are
  stable. The window twice as wide slows the growth to 0.24 without
  removing it. A model without a grid, `bin/breathing_loop.jl`,
  identified the loop and gives these numbers to 10 to 20 %. The solved
  `ä` writes into the source an oscillation the predictor's `a⃛` and
  `a⃜` do not follow; the order-3 near field's `ä` term leaves an error
  source over the window's ball; and its field pushes both particles
  in phase along their separation. The run with the predictor's `ä` in
  the source, which the model predicts stable, is stable to `t = 20` at
  `q² = 0.2`. The cure is left to the owner.
- 2026-11-23: the owner preferred the source with the predictor's `ä`
  and asked what error it introduces. `Dynamic` now keeps the blended
  predictor's `ä` in its solve record, and `bin/binary.jl` writes it per
  row with the positions and velocities. `jobs/addot_error.jl` ran the
  binary with the solved and the predicted `ä` side by side at
  `q² = 0.005` and 0.01, at `d/2` and `d/4`. They agree in the
  tangential force and the energy loss to `7·10⁻⁴` or better, with no
  difference that persists at `d/4`. At `q² = 0.2` the predicted `ä`
  is stable at both resolutions, and its force against the dipole's
  is 0.3 % above the small charges'. A second way, `bin/addot_error.jl`,
  estimates the error from one run's rows through the retarded field of
  the source's mismatch. It failed the check below the threshold,
  overestimating the measured effect by orders of magnitude: it
  differences the solved jets' step-to-step jumps across rows, and the
  continuum's retarded field counts the result in full (`CODE.md`,
  "The breathing instability"). It stays as a starting point, with that
  caveat.
- 2026-11-24: the owner asked to try two cures at once: the wider window
  with the predicted `ä`, and the near field to order 2. `jobs/cures.jl`
  ran both at `q² = 0.2` and 0.01, at `d/2` and `d/4`. Both are stable at
  0.2.
  - The wide window brings the solved `ä` to within 4 % of the
    predictor's, against 60 to 240 % with the standard window.
  - With the wide window, `d/2` and `d/4` agree to `10⁻⁴`, and at 0.01
    it agrees with order 3 and the standard window.
  - At 0.2 it inspirals 1.3 % further than the standard window and
    order 2 do, at every resolution. That is likely the error source of
    the ringing jets, which only the wide window removes.
  - Order 2 gets the radial force's sign wrong, and its value does not
    converge.

  `CODE.md`, "Two cures side by side" under "The breathing instability".
  The choice is left to the owner.
- 2026-11-25: the owner asked for the check at `q² = 0.05`. It reversed
  the reading of 2026-11-24: there the wide window holds the binary
  with either `ä`, and with the solved `ä` the two windows agree to
  0.1–0.24 %. The wide window with the predicted `ä` loses its energy
  0.45 % too fast, at both resolutions. So the 1.3 % at 0.2 was the wide
  window's error, not the standard window's.

  The cause is a bias of the predictor's `ä`, low by about `0.07q²` of
  itself whichever `ä` the source takes, since the predictor has
  neither the self force nor retardation. The wide window weighs a wrong
  `ä` ten to thirty times more than the standard window. `CODE.md`,
  "Which window is right" under "The breathing instability". A source
  `ä` corrected by a slow running mean of the bias is proposed, not
  tried.
- 2026-11-25: the owner asked to try that source. `Dynamic`'s
  `source_addot = :corrected` adds to the predictor's `ä` a running mean
  over `τ` of a fit of the solved `ä`'s departure along the predictor's
  `ä` and `ȧ`. The first runs (study `twoldg-2a5c44f`) let the onset's
  fits into the mean, and the smoke run's solve went to NaN; they were
  cancelled, and the mean now waits for the force to take over
  (`twoldg-c9b9501`). The smoke run still fails, at `t ≈ 0.03` with the
  mean zero throughout, so with the predictor's `ä` alone: the smoke's
  order 3 at `q² = 0.2` with an onset of `1/16` on one level, not the
  correction.

  With `τ = 2` every run is stable. With the standard window at `d/4` it
  agrees at `q² = 0.2` with order 2 to `5·10⁻⁵` in `d(20)`, and at 0.05
  with the wide window's solved `ä`. With the wide window it changes
  nothing, although its mean is the bias: so the bias was not the cause
  of the wide window's energy loss, and the reading of the morning is
  withdrawn. At `d/2` the standard window's mean takes up the face
  crossings' noise. `CODE.md`, "The corrected `ä`" under "The breathing
  instability". The wide window's runs at `d/4` were still running.

### What the plan said of its finished milestones

`INSPIRAL-PLAN.md` keeps one line for each finished milestone. Its
earlier text for M0 to M5 is here, as it stood when M5 was done, and
for M8a and M8b as it stood when M8c started, and for M8c as it stood
when it was closed and M8d planned, and for M8d, M8e, M8f and M6a as
they stood when each was closed, with
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

#### M6a: One particle, solved

**Status:** done (2026-11-20), merged with pull request #21; its runs
are jobs 606324 to 606328, 606429, 606769 and 606770, and their
results are in `CODE.md` ("The equations of motion" under "Measured
results"). As planned before the work, which departed from it in the
own near field's averages, taken by quadrature, and in solving `a`,
`ȧ` and `ä` together (above, 2026-11-19). `INSPIRAL.md`: "Equations of
motion", "The acceleration and its derivatives"; validation step 6,
first half.

- The averages `Φ̄ₑ`, `∇̄Φₑ`, `Π̄ₑ`: the remainder's part from M2's
  pairing, the other particle's `Ψ_b` and the particle's own near field
  over `ball_rule`, the own near field's self-force from M1's
  coefficients.
- **The `a` solve**: the equation of motion with `m = M₀ + QΦ̄ₑ`, `F_N`
  added, a 3×3 solve.
- **`ȧ`**: the time derivative of the forces, with `∂ₜΦ_r = Π_r`, and
  the one `ä` in the relativistic `vΠ̄` term taken from the predictor.
- **The `ä` solve**: steps 3–4 of "The right-hand side": `wave_rhs!`
  first, then `∂ₜΠ_r` near the particle with the part of `P S` linear in
  `ä`, projected on the support elements, giving a 3×3 system.
- **The predictor** `ä_pred(X)`: the two-body instantaneous
  acceleration (Newtonian force and each particle's `Φ_C` on the other,
  with physical masses), differentiated twice along its own flow with
  nested `ForwardDiff`.
- An `ExternalForce` motion: a single particle with an analytic force,
  for example a central spring that holds it on a circle.

Tests: the solves reduce to `M a = F` at rest with no field; the
predictor equals finite differences of a numerically integrated two-body
orbit; the spring-held particle at `v = 0.05` reproduces M4's circle
forces.

Measured and recorded: the condition number of the `ä` system against
`hₑ`, and `δm_h/M` (the **(open)** stability question, in part).

**Done when** the external-force particle matches M4's circle at each
resolution, and the `ä` system's conditioning is recorded.

#### M8a: What the sphere sends back

**Status:** measured (2026-11-10), on branch
`eschnett-claude/elastic-colden-7ba544`; the results are in `CODE.md`
("The radiative sphere" under "Measured results"), and the gate below is
with the owner. The sphere sends back about eight times what the nested
boxes' outer face does at `λ = 16`, and at `λ = 4` the shells keep a
fifth of the energy. The dipole was centred, a pure `ℓ = 1` wave about
the sphere, not off-centre, and `n_e` was varied only with `R₁`, at
fixed spacings. The field alone, on the uniform ball, with no code
beyond a script.

- `bin/sevenpatch.jl`'s radiative run with `bin/nested_waves.jl`'s pulse,
  `[g(r − t) − g(r + t)]/r` with `g(s) = exp(−(s/σ)²) cos(2πs/λ)`, at
  `λ = 16` (the first run's radiation) and `λ = 4` (the start-up junk),
  and an off-centre dipole at `λ = 16`, since the binary radiates at
  `ℓ = 1`.
- Against `R₁` and `n_e`: what comes back to the window's ball, its
  amplitude and when, and the energy left in the ball.
- Beside it, the same pulses through M3b's five-level layout, from
  `nested_waves.jl`'s existing runs where they cover it.

**Done when** the reflection at the window is recorded against `R₁` and
`n_e` beside the nested boxes', and the owner has decided whether M8b
and M8c go ahead (a gate, below).

#### M8b: Refinement inside the cube

**Status:** measured (2026-11-11), before M8a's gate, at the owner's
request; its pull request is open. The code and tests are in, and
`bin/symmetry/nested_ball.sh`'s runs at M3b's scale are recorded
(`CODE.md`, "The seven-patch ball" under "Measured results"): inside the
cube the ball gives the nested boxes' coarse-fine reflections, the
energy holds as it does on the boxes, and the step time and memory are
recorded. The cube–shell faces send back about forty times what the
boxes' face at 16 does at `λ = 16`, and at `λ = 4` the shells need
radial elements of four per wavelength not to trap the wave. The field
side, with no particles.

- **The refusal.** `check_metric_forest` refuses every refined forest on
  a geometry, because the coarse ghost is the adjoint in the flat
  weights, not the `√g`-weighted ones. Where `√g` is constant the two
  differ by a constant factor, so a refined forest is accepted when
  every coarse-fine face lies in a patch whose map is affine (a trait of
  the geometry, `affine_patch(geo, p)`), and refused otherwise with that
  reason.
- **The Cartesian gradient across a coarse-fine face.** `cartgrad` is
  one set with ghosts on every face, which no refined forest has had.
  In the cube `S` is diagonal and constant, so the Laplacian reads
  `F^d = S[d][d] V_d` as the flat path reads its `d`-th set; whether
  TreeAMR's coupling treats each component as it treats that set is
  checked by the energy test.
- **The truncation by block.** On an affine block the `√g`-orthogonal
  projector is the element's own, so those blocks take the element's
  tables and `ElementTruncation` holds only the shells' elements. On
  the nested cubes that saves about 100 bytes per node.
- **The forest.** `sevenpatch_forest(…; refined)`, with `nested_cubes`'
  regions in patch 1's logical coordinates.
- **The time step** stays `oldg_timestep`'s, the finest logical `hₑ`
  over the largest logical wave speed anywhere, which is conservative.

Tests: the energy, with the Neumann sphere (`κ = 0` without `u/R`), an
invariant to rounding on a refined ball; the plane wave converging on a
refined ball with Dirichlet data; refinement in a shell refused, naming
the patch.

Measured and recorded: the right-hand side's time per node on the
cube's blocks against the flat path's on M3b's layout. A flat fast path
for affine blocks is written only if the difference matters against the
source's cost (below, "Left to settle later").

**Done when** M8a's pulses through the refined ball give the nested
boxes' coarse-fine reflections, the energy holds as it does on nested
boxes, and the memory and step time are recorded.

#### M8c: Particles in the cube

**Status:** under way (2026-11-11), on branch `inspiral-m8c-ball`. The
code and the tests are in (`CODE.md`, "Particles in the ball"): TreeAMR
0.2.9 has the location in a patch, the transfer works through the
cube's `ParticleFrame`, and the drivers take `--ball`. The runs were
made on Symmetry on 2026-11-12 (jobs 596453 to 596456, `HISTORY.md`;
results in `CODE.md`, "Particles in the ball" under "Measured results"),
with the boxes of the cube's three levels run beside the ball by the
current code, since M3b's runs predate M4's and M5's changes to the
remainder. Their layout, from M8a's and M8b's measurements and the
owner's answer to M8a's gate: the shells' radial elements at most twice
the cube's root element width; the reflection scans then found what
comes back set by the shells' radial elements per wavelength, about 8
for the shortest wave carried out, at every radius, more than by the
jump at the cube's face; `R₁` at least `3Lc`, where the finest cube
sets the step, and large enough that what the sphere sends back, from
about `2R₁ − R₂` on, arrives after the run. The plan's ball at
`R₁ = 50`, `Lc = 16` with `hshell = 1` is `bin/ball.jl`'s default for
`--ball --levels=3 --w0=4 --hdiv=8`, and is clean to about `t = 97` for
a resolved wave; M8c's runs at rest show the under-resolved error back
from about `t = 49`.
The runs' scripts: `bin/symmetry/ball_motion.sh`, M3b's series and the
particle at rest to `t = 60` at `R₁ = 36` and 50, and `prescribed.sh`
with `TREEWAVEOLDG_RUNS="circle-ret:8:3"` and
`TREEWAVEOLDG_EXTRA="--v=0.2 --ball --levels=3 --R1=50"` for M5's circle.

- **Point location, upstream.** TreeAMR refuses `locate_point` on a
  forest of patches, and its refusal names what would lift it: a
  location given the patch of each point. That is TreeAMR's,
  `locate_point(forest, p, ξ)` ("Scope and non-goals"), followed by a
  `[compat]` bump in both `Project.toml`s and both manifests.
- **`InspiralProblem`** accepts a `FullGeometry` with a particle patch
  whose map is affine, in place of its refusal of every geometry and
  forest of patches, and refuses a window's ball or a support outside
  that patch's finest level, as it refuses one outside the finest level
  now.
- **The transfer in physical coordinates** through the patch's affine
  map, origin `o` and scale `a`: block origins and spacings mapped into
  space, so that `hₑ`, `rnear`, `rfar` and the graded bands are physical;
  `P S` unchanged, since the Jacobian cancels in `(1/wₙ)∫Sφₙ`; the
  pairing's integrals times `a³` and the gradients times `1/a`. This
  covers `project_source!`, `pairing`, `support_elements`,
  `SharedNodes`, the finest-level checks and `reconstruct`. (Done, with
  the pairing over the mapped element boxes, which needs neither factor:
  `CODE.md`, "Particles in the ball".)
- **Initial data and norms.** `inspiral_initial_state` fills `(1 − W)Ψ̂`
  through `WithPatch` and `physical_coordinates` on all seven patches;
  `ball_norm` takes physical positions and `√g`-weighted volumes.
- **The averaging** stays on patch 1's finest level, where it is the
  nested boxes' unchanged.

Tests: on the cube, the projection, the pairing and the initial
remainder equal a nested-box forest's over the same finest cube element
for element, to an ulp, as M3b's test against a uniform box; a support
outside the particle patch is refused.

Runs, beside M3b's and M4's on nested boxes:

- a particle at rest and in uniform motion past the nested boxes' clean
  time, with the error in the `R₂` ball and the force recorded to then;
- a single charge's `Π_r` mean, which drifts in the periodic box and
  should not here;
- M4's circle at `v = 0.2`, `d/8`, order 3, its forces against M5's.

**Done when** inside the nested boxes' clean time the ball gives their
numbers to the digits printed, and beyond it the error and the forces
are recorded against `R₁`. M7 then chooses its outer region (below,
"Left to settle later").

#### M8d: Far data that are the field

**Status:** done (2026-11-15) on branch `inspiral-m8d-far-data`. The
uniform start, its tests, `--data=uniform`, `SCAN=far` and
`bin/dispersion.jl` were in on 2026-11-14, and the two runs ran on
H200s that night (jobs 598636 and 598637; results in `CODE.md`,
"Particles in the ball" under "Measured results"). As planned before
the runs: it settles what M8c left open (`CODE.md`, "Particles in the ball" and "What the
transitions send back"):

- **Initial data that are not the field far out.** M8c's circle took the
  retarded field inside `rx1 = 48` and the particle field `Ψ̂` of order
  3 beyond 54. Far from an accelerated particle `Ψ̂`'s near-field terms
  grow like `r²`, a hundred times the field at `r = 48`, and what they
  sent in swamped the remainder from `t ≈ 25`. `INSPIRAL.md`'s initial
  data, `(1 − W)Ψ̂`, would do the same to the inspiral.
- **Error that outruns light.** That signal crossed the shells at 1.7 to
  1.9 times the speed of light, nearly whatever their radial width, and
  the cube's fine levels at about the speed of light. The periodic box's
  faces send their error in at about twice it (M3), the nested boxes'
  levels at about 1.5 times (M3b), and the sphere's under-resolved
  return comes at `(2R₁ − R₂)/2`, not `2R₁ − R₂`. `CODE.md` puts it down
  to high modes whose group velocity OLDG does not bound by one, which
  nothing has checked.

**The layout** is M8c's and stays fixed (the owner, 2026-11-14): the
plan's cube (three levels, `w0 = 4`, `Lc = 16`, `d/8`, `P = 3`), shells
of radial width 1, 16 elements per wavelength of the circle's
`λ = 15.7`, and `R₁ = 100`, so that the sphere is causally disconnected
from what is measured. Nothing is scanned in `R₁`, and the shells'
resolution is M8c's rule. The runs end at `t = 90`, before the
under-resolved return at about `(2R₁ − R₂)/2 ≈ 99`. That is about
`1.8·10⁸` nodes and, from M8c's circles, about a second a step on an
H200: about 23 000 steps at `dt = 1/256`, seven hours.

**The data.** The circle at `v = 0.2` from two starts, beside M8c's:

- **The retarded field everywhere.** The eternal circle's retarded field
  at every node: `--data=retarded` with `rx1` beyond `R₁`, so that the
  retarded window is 1 in the whole ball and no `Ψ̂` is left far out.
  Inside 48 these are M8c's data, so the run repeats M8c's circle in
  shells of width 1 (job 597910) to `t ≈ 25`, a check. It needs the
  path at every retarded time, back to `t ≈ −R₁`, which the circle has
  in closed form.
- **Uniform motion's field outside the window.** `Φ_r = (1 − W)Ψ̂₀` and
  `Π_r = (1 − W)∂ₜΨ̂₀`, with `Ψ̂₀` the boosted static field
  (`particle_field(…; order = 0)`) of the particle's position and
  velocity at `t = 0`, and `∂ₜΨ̂₀` its rate along the tangent line, with
  `a` and its derivatives zero: not `particle_rate`'s `𝒟Ψ̂₀`, which
  carries `a` through `v̇`. Beyond `R₂` the data are then the field of a
  charge that has always moved on the circle's tangent, an exact
  solution: the charge is seen everywhere, and what starts at `t = 0` is
  only what the acceleration changes, radiation reaction switched on at
  once. It needs no past path and no retarded time, so it would also do
  for the dynamic binary.
- **M8c's circle** from `Ψ̂` of order 3 beyond 54 (job 597910, to
  `t = 40`) is the third point, not rerun.

In the continuum both starts give the eternal circle's retarded field at
the particle from `t ≈ 3` on. The solution is the source's part from
`t = 0` on plus the data's free evolution. The source's part is the
eternal field wherever the past light cone meets the path after `t = 0`,
at the particle from `t ≈ 0.5`. The data's part vanishes there by
Huygens' principle: the window's part once `t > |x − c| + R₂`, since it
lies within `R₂` of the centre, and the tangent line's part since the
past light cone meets that line after `t = 0` too. What can differ is
numerical: the front that leaves the window, across which `Φ` is
continuous to leading order and `Π` changes by the radiation's
`Q(R·a)/(4πγρ²)`, about `a` times the Coulomb scale, over the window's
transition `R₂ − R₁ = 1.25`. It is partly under-resolved on the cube's
root and in the shells, and what the transitions send back from it is
the measurement. A zero remainder, the particle field inside the window
and nothing beyond, was dropped: it leaves the Coulomb field out beyond
`R₂`, and its front would carry the whole monopole (`HISTORY.md`).

Hyperbolic motion keeps its data: its retarded field does not reach
beyond `1/g` behind the particle, and its runs end long before anything
from `rx2` arrives. Rest and uniform motion need nothing: at `a = 0`,
`Ψ̂` is the boosted Coulomb field, their field everywhere.

**The code.**

- `inspiral_initial_state(p; far)`: `far = :particle`, today's
  `(1 − W)Ψ̂` of `p.order`, or `far = :uniform`, `(1 − W)Ψ̂₀` and its
  rate along the tangent line, through `RemainderData` with order 0 and
  the trajectory's uniform part.
- `bin/prescribed.jl --data=uniform`, and `--rx1` beyond `R₁` for the
  retarded field everywhere; the errors stay against the eternal
  retarded field, now in any ball, the shells' included. `--radii` up to
  64.
- `bin/symmetry/gpu_scans.sh`: the two circles as scans of their own.
- **The reference's cost.** It is filled on the host at every output, a
  retarded-time solve per node, `1.8·10⁸` of them at `R₁ = 100`. Its
  cost is measured first; if it matters, it is filled less often, only
  on the balls measured, or on the device.

Tests: at `a = 0`, `far = :uniform` gives today's data bit for bit; on
the circle its `Π_r` beyond `R₂` is the boosted field's derivative along
the tangent, a finite difference in `t`, and differs from `𝒟Ψ̂₀`; a
coarse ball takes a few steps from it.

**Why the error outruns light.** The dispersion relation of OLDG at
`P = 3`, with and without truncation: the Bloch eigenproblem of the
semi-discrete operator on a periodic uniform line, then along a diagonal
of a periodic brick, and the group velocity of each branch against `kh`,
in a `bin/` script on the laptop. Recorded: whether the largest group
velocity and the wavenumbers that carry it explain the measured speeds,
about 2 on the box's faces, 1.5 across nested levels, 1.7 to 1.9 in the
shells and about 1 on the cube's fine levels; and whether the speed in
the shells follows their tangential spacing, which was the same in M8c's
three circles while their radial width changed.

**Runs**, on an H200, the circle at `v = 0.2`, `d/8`, order 3, from both
starts, to `t = 90`: the force every step, and the error against the
eternal retarded field in the balls of radius 4, 8, 16, 32 and 64 about
the centre every half time unit.

Measured and recorded: `F_t/F_RR` and `F_r/F_RR` over windows to
`t = 90`; when the uniform start's force comes to the retarded start's
and how close it stays; when the front leaves each ball and what comes
back into it; the reference's cost.

**Done when** the retarded start repeats M8c's circle to `t ≈ 25` and
its force is recorded to `t = 90`, clean or with what arrives and from
where; the uniform start's force and errors are recorded against it;
the dispersion relation's speeds are recorded against the measured
ones; and `CODE.md`'s "Particles in the ball" names the far data for
prescribed paths and records the uniform start. M7 then chooses its
initial data (below, "Left to settle later").

#### M8e: Waveforms

**Status:** done (2026-11-16) on branch `inspiral-m8e-waveforms`;
its runs are jobs 599424 to 599429, and their results are in `CODE.md`
("Waveforms" under "Measured results"). As planned before the runs:

M8d measured the field only as the L2 norm of its error in balls about
the centre, and the force at the particle. Three questions need the
waves themselves:

- **Several orbits.** Does the resolution of M8d hold the waves over the
  five orbits a run to `t = 90` covers, and how does their error grow
  with the orbits, in phase and in amplitude?
- **Clean waves.** Are the waves that leave free of what the cube's
  coarse-fine faces (at 4 and 8) and the cube–shell faces (16 to 27.7)
  send back, of the sphere's return, and of error ahead of light? In
  the balls' norms the retarded start's error converges at about fourth
  order at every time; a mode at a fixed radius says when and from
  where anything arrives.
- **The two starts.** In the continuum the uniform start is the eternal
  circle's field wherever `t > |x|` (M8d), so its residual of about 1 %
  of the field in the ball of radius 64, which does not move with the
  resolution and grows again after `t ≈ 70`, is not the data's free
  evolution. That the sphere sends it back, answering the uniform data's
  large multipoles near `R₁` (the field of a charge displaced by `vR₁`
  along the tangent), is a guess; an incoming wave in the modes, at the
  outer radii first, would show it.

**Extraction.** Spheres of radius `r` about the centre; on each,
Gauss–Legendre in `cos θ` at `n_θ` points times `n_φ = 2n_θ` equally
spaced angles, which integrates `Y_ℓm Ȳ_ℓ′m′` exactly for
`ℓ + ℓ′ < 2n_θ`. The modes `Φ_ℓm`, `ℓ ≤ L`, `0 ≤ m ≤ ℓ` (the field is
real), are that quadrature of `Φ Ȳ_ℓm`. Each point goes to its patch and
logical position by the seven-patch ball's inverse map: a shell's point
lies on the ray through its point on the cube's face, so its face is
the largest coordinate's and its sign, its tangential position is the
face point's, and its radial one inverts the blend and the grading in
closed form; the cube's map is affine. The points are located once
(TreeAMR's `locate` of `PatchPoint`s) and read at every sample with
TreeAMR's `ElementPolynomial`: the element's own degree-`P` polynomial,
which is the solution itself, from the state vector, with no ghosts, on
the problem's backend. Beyond `R₂` the remainder is the field, so the
modes are `Φ`'s. The reference's modes are the same quadrature of
`retarded_field` at the same points, on the host; aliasing from
`ℓ′ ≥ 2n_θ − ℓ` is the same in both and cancels in the error.

**Its test, on its own** (`test/extraction_tests.jl`), before any run:

- the quadrature recovers the modes of a sum of `r^ℓ Y_ℓm` given at the
  points, to roundoff, and the table is orthonormal under it;
- the inverse map: for points throughout the ball, in the cube and in
  every shell, graded and not, on faces and edges, the forward map of
  the `(ξ, p)` it finds is the point to roundoff, and `ξ` lies in the
  patch's box;
- the extraction on the grid: a smooth field filled at the nodes of a
  small ball with refinement in its cube, on spheres inside the cube,
  across its corners (`Lc < r < √3 Lc`) and in the shells: its modes
  against the function's own at the points converge at order `P + 1`;
- on a device the modes are the host's to roundoff
  (`test/device_tests.jl`).

**The runs.** M8d's circle from both starts at `d/4`, `d/6` and `d/8`,
on M8d's layout to `t = 90`, the shells refined with the cube as in
*Its convergence*, with spheres at `r` = 10 (the cube's root level), 20
(across the cube–shell faces), 30, 50, 70 and 90 (the shells), `L = 6`,
`n_θ = 24`, every `1/16`: six runs on one H200 each, about 17 GPU-hours,
the extraction's cost negligible beside a step. Each writes its modes
and the reference's to a file of its own beside the rows.

**What tells the answers.** For each mode and radius, the error against
the reference's, `r(Φ_ℓm − Φ_ℓm^ret)`, over time:

- the waves are clean if it converges at the scheme's order at every
  time and nothing in it reaches a radius sooner than light from the
  transitions or from the sphere (`t ≈ 100 − r`) could bring it;
  something the sphere sends back reaches 90 first and 10 last;
- the orbits: its amplitude and phase parts, against
  `|Φ_ℓm^ret|` and `Ω = 0.4`, `mΩ` for `(ℓ, m)`, orbit by orbit;
- the starts: the same for the uniform start, and whether its residual
  is incoming.

Done when the test passes on the CPU and on a device, `gpu_scans.sh`
runs the six, and `CODE.md` records them with the three answers.

#### M8f: A smooth past and a smooth departure

**Status:** done (2026-11-18) on branch `inspiral-m8f-smooth-past`;
its runs are jobs 600172 to 600182 and 601825 to 601832, and their
results are in `CODE.md` ("Assumed pasts" under "Measured results").
As planned before the runs:

M8e found the circle's retarded field clean and the uniform start not.
The difference lies in no fault of the uniform data far out, which are
exact: it lies in the path those data assume. They are the retarded
field of a charge on the tangent line until `t = 0` and on the circle
since, a path whose acceleration jumps at `t = 0`, at the particle; and
inside the window they are the circle's near field, not that path's. A
past need not be the true one: the retarded field of any path is an
exact solution with that path's source. What makes data clean is that
the assumed past joins the motion smoothly, and far enough back that
what its joints send out never meets the particle. The binary has no
known past; it can have an assumed one, its orbit for a while, reached
smoothly from rest, and from `t = 0` the force can take over smoothly
from whatever held that orbit. M8f tests both halves on prescribed
paths, where the reference is exact.

**Questions.**

- **The past.** Do data that are the retarded field of a circle spun up
  from rest, on the circle for a time `T` before `t = 0`, give waves as
  clean as the circle forever, with no pulse out of the window, no wake
  at second order, and from the sphere no more than the circle's own
  reflection? How smooth must the spin-up be: with how many of `a`,
  `ȧ`, `ä`, … continuous does its front, a shell of radius `T` to
  `T + Δ` at `t = 0`, leave at the scheme's order, and how much of it
  runs ahead of light or back in toward the particle?
- **The departure.** On the binary, the force takes over at `t = 0`
  from what held the assumed orbit; on a prescribed path that is a
  departure from the circle at `t = 0`. Does a departure whose
  acceleration's derivatives are continuous through order `n₂ − 1` send
  nothing measurable, and an abrupt one, a jump in `a` like switching
  radiation reaction on at once, something measurable? Of what size,
  and for which `n₂` is it gone?

**The steps.** Both ramps use the smoothstep of order `n`, the
polynomial `S_n` of degree `2n + 1` that rises from 0 at `x = 0` to 1 at
`x = 1` with its first `n` derivatives zero at both ends; `S_0(x) = x`.
Its integral and all its derivatives are closed forms, so every path
below has exact jets and runs in a device kernel. A smooth step of class
`C^∞` (from `e^{−1/x}`) would need a quadrature in the path, on every
root-finding step on the device; `n` up to 6 is expected to be enough,
and the runs say whether it is.

**The spun-up past.** The circle of M8d, `z = c + r_c(cos φ, sin φ)`,
with `φ̇ = Ω` for `s ≥ −T` and `φ̇ = Ω S_n((s + T + Δ)/Δ)` before, so at
rest at its point of the circle before `s = −T − Δ`. Then `a` and its
first `n − 1` derivatives are continuous at both joints and its `n`-th
jumps (`n = 0`: `a` itself jumps). The speed stays below `v`. It is a
uniform motion, at rest, joined smoothly to the circle; the moving
tangent line M8e's uniform start assumed is not taken, because its
charge, displaced by `v(R₁ − t)` on the sphere, is what the sphere sent
back (22 % of the dipole wave by `t = 90`), while a charge at rest on
the circle has the circle's own static dipole; and because a curve of
constant speed and ramped curvature has no closed form, so its position
would need a quadrature. For `t ≥ 0` the path is the circle's, with its
exact jets, so the particle field, the window and the force are M8d's;
the past enters only through `retarded_field`, which needs `z` and `v`
and differentiates them.

**The departure.** From `t = 0` the radius shrinks:
`r(t) = r_c − εΔ₂ I_n(t/Δ₂)`, with `I_n` the integral of `S_n`, so the
radial speed rises to `−ε` over `Δ₂` and stays there, at the angular
velocity `Ω`; the jets through `a⃜` follow from Leibniz's rule in closed
form. `a` and its first `n₂ − 1` derivatives are continuous at `t = 0`
and its `n₂`-th jumps, by `ε/Δ₂^{n₂+1}` times a number of order one;
the same happens again at `t = Δ₂`, where the ramp ends.
`ε = 10⁻³` and `Δ₂ = 2π/Ω`, one orbit: the radius falls from 0.5 to
about 0.41 by `t = 90`, inside M8d's window, and the jump in `a` at
`n₂ = 0`, `6·10⁻⁵`, is fifteen times the radiation reaction's
acceleration, `F_RR/M = 4·10⁻⁶`, large enough to see and small enough to
be a realistic switch-on's scale. The data are the spun-up past's, with
`n = 6`, or the circle forever if the past's runs have shown them equal.

**The code.**

- `src/prescribed.jl`: `smoothstep(n, x)` and its integral; a path
  `RampedCircle` (the spun-up past, `T`, `Δ`, `n`; the departure, `ε`,
  `Δ₂`, `n₂`), with exact jets for `s ≥ 0` and `z`, `v` before, and a
  constructor beside `circular_motion`. `ε = 0` is the circle for
  `s ≥ 0`, `T = ∞` the circle before.
- `bin/prescribed.jl`: `--past=spinup --T --ramp --n` and
  `--depart=ε --onset --n2` on `--path=circle --data=retarded`. The
  reference, the waves' reference and `--reading` all take
  `p.motion.path`, so they follow the path with no change.
- `bin/waveforms.jl`: M8e's analysis made a script, the errors by
  radius and time, their orders, the clean windows, the arrivals, and
  the orbits' amplitude and phase from the `--waves` files, with the
  two-piece reference of M8e's uniform start as an option, so that
  M8e's numbers can be made again.
- `bin/symmetry/gpu_scans.sh`: `SCAN=past` and `SCAN=depart`, with
  `N`, `T`, `RAMP` and `N2`.

**Its test, on its own** (`test/prescribed_tests.jl`), before any run:

- `smoothstep`: the values, the vanishing derivatives at both ends to
  roundoff, and the integral against a quadrature;
- the path's jets are the derivatives of its `z`, by `ForwardDiff` in
  `BigFloat`, on each piece and at the joints; at a joint the
  derivatives below `n` agree on both sides and the `n`-th does not; the
  speed stays below 1; with `ε = 0` and `T = ∞` it is `CirclePath` bit
  for bit;
- `retarded_field` of the spun-up past equals the circle forever's to
  roundoff wherever the retarded time is after `−T`, and is the static
  charge's where it is before `−T − Δ`;
- on a device the reference filled from it is the host's
  (`test/device_tests.jl`).

**The runs.** On M8d's ball (`R₁ = 100`, three levels, `w0 = 4`) to
`t = 90`, with M8e's spheres and modes and `--every=0.5` for the norms:

- the past: `T = 30`, `Δ = 10`, so that the front starts in the shells
  beyond the cube's corners and crosses the spheres of 50, 70 and 90,
  while those of 10 and 20 see what it sends back in; `n` = 0, 1, 2
  and 4 at `d/4` and `d/6`, and `n` = 0 and 4 at `d/8`, ten runs;
- the departure: from the past that the first series found clean,
  `n₂` = 0, 2 and 4 at `d/4` and `d/6`, and 0 and 4 at `d/8`, eight
  runs.

Eighteen runs, about 45 GPU-hours on H200s at M8e's 47 min, 2 h 10 min
and 5 h 45 min: two nights on six GPUs, since the departure starts from
the past's answer; the force every step as in M8d.

**What tells the answers.**

- The past is as clean as the circle forever if, against its own
  retarded field, the errors on the spheres of 10 and 20 and in the
  balls are the retarded start's of M8e to their scatter, the force's
  orbit means and scatter M8d's, and the front's error converges at the
  scheme's order on the spheres it crosses. The smallest `n` for which
  it does is the answer to "how many derivatives"; below it, the order
  on the front falls, and what arrives ahead of it at up to 1.58 times
  the speed of light, or behind it on the inner spheres, says what the
  joint costs.
- From the sphere: the static `ℓ = 1` field that arrives at
  `t = 100 − r` against M8e's two starts, expected at the circle's own
  reflection, not growing.
- The departure: its error against the path's retarded field on the
  spheres, which an abrupt start should show as an outgoing pulse from
  `t = 0` at `r = t`, and the force against `--reading`'s, the exact
  remainder's, step by step over the first orbits. Clean if neither
  shows anything at `n₂` beyond what `n₂ = ∞`, the circle, shows.

Done when the tests pass on the CPU and on a device, the runs are in,
and `CODE.md` records them with what the binary's initial data need: the
`n`, `T` and `Δ` of its assumed past, and the `n₂` and `Δ₂` of the
force's take-over, for M6b to take.

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

- **M8a: the sphere sends back more than the nested boxes do.** If, at
  an `R₁` whose shells cost no more than two more nested levels, the
  radiation the sphere sends back to the window exceeds what the nested
  boxes' faces send back over the same run, M8b and M8c wait. The
  fallbacks are a second-order absorbing condition (Bayliss–Turkel,
  `CODE.md`'s **(open)** item under "The seven-patch ball") or the
  13-patch ball, which now exists (`CODE.md`, "The thirteen-patch
  ball"; `bin/sphere_waves.jl --Rm --m2`). Passed on 2026-11-11: the gate fired as written, the sphere
  at `R₁ = 50` sending back about eight times what the boxes' face at
  32 does at `λ = 16`, a third later; the owner passed it with M8c's
  layout as the answer, the sphere's return after the run (above, the
  entry of 2026-11-11 under "Particles").
- **M6a: the `ä` system is ill-conditioned** (`δm_h` not small against
  `M`). Coarser `hₑ`, or the "free amplitudes" alternative from
  `INSPIRAL.md`'s "Open questions". Did not fire (2026-11-20):
  `δm_h/M = 7·10⁻¹⁰` and a condition number of 1 at every resolution from
  `d/4` to `d/12` (`CODE.md`, "The equations of motion" under "Measured
  results").

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
- M8a: `R₁`, `Lc`, `n_e` and the shells' radial elements. Settled
  2026-11-11 by M8a's and M8b's runs: the plan's `R₁ = 50`, `Lc = 16`,
  `n_e = 64`, with radial elements at most twice the cube's root width
  (M8c's status in `INSPIRAL-PLAN.md`).
- M8b: whether affine blocks get a flat fast path in the right-hand
  side. Settled 2026-11-09: the owner had it written, since the full
  geometry's right-hand side cost 7.7 times the flat one's per node on
  the same blocks; with it the cube's blocks cost 1.9 times (`CODE.md`,
  "The seven-patch ball", and `PLAN.md` for what is left).
- M8c: `R₁` for each run, from the time it has to stay clean. M8c's
  runs at rest showed the sphere's return at about `(2R₁ − R₂)/2`, not
  `2R₁ − R₂`, the two spheres agreeing before it, and at `R₁ = 72` below
  `10⁻⁸`. Settled 2026-11-14 by the owner: `R₁` large enough that the
  sphere is causally disconnected from the run, 100 for M8d, and no
  scans in `R₁`.
- M6b: whether diagnostics get their own file. Settled 2026-11-20:
  `src/diagnostics.jl`, since they read the particles' state and nothing
  of the grid (`CODE.md`, "Diagnostics").
- M8c and the inspiral: initial data that are the field far from an
  accelerated particle. `(1 − W)Ψ̂` of order 3 grows like `r²` and is a
  hundred times the field at `r = 48`, and on the circle what it sent in
  crossed the shells at about twice the speed of light. Became M8d
  (2026-11-14).

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
- 2026-11-10: `test/runtests.jl` logs each test file's name as it
  starts, at Erik Schnetter's request, after CI's one-thread run went
  silent for over an hour: Test prints nothing until the outermost
  testset ends, so a slow run and a stuck one looked the same.
- 2026-11-11: TreeAMR 0.2.9 required, for `locate_point` in a patch
  (`PatchPoint`), which M8c's particles in the ball need. The suite
  passes against it unchanged elsewhere.
- 2026-11-20: the Symmetry batch scripts became SimLaunch job specs in
  `jobs/`, one per script, at Erik Schnetter's request, following
  SimLaunch's `CONVERTING.md`. SimLaunch takes over what every script
  did by hand: the rsynced checkout and its `setup` (a snapshot and
  `prepare`, which precompiles on the node type that runs it), the
  pinning, the output directory and SimWatch (`bin/symmetry/simwatch.sh`).
  A script whose runs were given by a variable (`TREEWAVEOLDG_RUNS`,
  `SCAN`, `MODE`) became a spec with its default runs written out; M6a's
  finer series, which ran with the variable on `amdq`, has a spec of its
  own (`dynamic_series.jl`), and the GPU scans one spec whose runs are
  submitted one by one (`gpu_scans.jl`), with `gpu_pulses.jl` for the
  pulses that ran in one job. The suite's three ways now run side by
  side in one job of about an hour, where they took three. The GPU scans'
  specs keep the old 128 GB of host memory by passing `--mem` through,
  since a GPU job's memory comes from the site's profile (reported to
  SimLaunch). Checked on Symmetry against the old scripts' runs (study
  `twoldg-sl-check`): the suite passed all three ways (job 606982,
  `cn107`, 10 410 tests and the known broken one each); the flat box's
  step at 64 threads was 71.0 and 63.1 ms against job 586013's 72.1 and
  63.5 (606983, `cn091`); the H200's flat box and ball 1.60 and 2.76 ns
  per node against 1.61 and 2.84, the device tests 93 of 93 (606985,
  `cn111`); the benchmark ball the old script's numbers up to 32 threads
  and, on a second run, at 64 (606984, 607029, against the old script on
  the same node, 607027; "Threads on Symmetry" has the 64-thread jump of
  the first run). `suite.sh`, `benchmark.sh`, `ball_benchmark.sh` and
  `gpu.sh` were then removed; the others stay until a run of their spec
  is due and gives their numbers.

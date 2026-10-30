# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [The right-hand side](#the-right-hand-side)
- [Truncation](#truncation)
- [The blast wave and the viewer](#the-blast-wave-and-the-viewer)
- [Threads and the integrator](#threads-and-the-integrator)
- [Dependencies](#dependencies)

## Scope and non-goals

- 2026-10-26: package created by Erik Schnetter, to implement the scalar
  wave equation like TreeWave but with the OLDG stencils of
  arXiv:1604.00075. The first version was decided to be the unrefined
  prototype that TreeAMR's `CODE.md` proposes ("Element-local DG
  stencils"), since a refined forest needs a Gauss–Lobatto centering,
  element-local transfer operators and a schedule-builder change in
  TreeAMR first.
- 2026-10-26: TreeWave's refinement criterion, regridding, pulse and
  blast-wave cases, viewers, benchmark and device tests were not carried
  over; all but the viewers and device tests exist to exercise
  refinement.
- 2026-10-27: reflecting (solid-wall, Neumann `∂ₙu = 0`) boundaries
  were added by way of TreeAMR's reflection symmetries: a `reflecting`
  keyword on `wave_forest` marks every face reflecting, and `WaveProblem`
  derives the per-variable parities the wall imposes (`u`, `v` even;
  `∂_d u` odd in `d`) for the field sets' ghosts, so the wall needs no
  boundary hook. A standing `cos` mode, `reflect_exact`, exercises it and
  converges at order `P` with energy conserved to the integrator's own loss.
  Global (rotating-seam, partly reflecting or outflow) boundaries remain a
  non-goal.
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
  corner of all five variables in 3D. They took half a right-hand side
  on the laptop (33 of 65 ms; 3D, `P = 3`, 512 blocks of `16³`). Now
  `v` is read from the state and never exchanged, `u` has a set of its
  own, and each `∂_d u` a set with ghosts along `d` only, `1 + D`
  exchanges of about 2.7 times fewer points. The state set keeps
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
  serial broadcast, and cost more than half a right-hand side (41 ms on
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
- 2026-10-30: two changes to IMEXRungeKutta were considered and kept as
  ideas: partitioning an `ArrayPartition` state, which `INSPIRAL-PLAN.md`
  plans for the inspiral with OrdinaryDiffEq, and a low-storage method.
  `INSPIRAL-PLAN.md` was left as it is at Erik Schnetter's request, so
  its `ArrayPartition`/OrdinaryDiffEq state in M3 is now at odds with
  the package's integrator; "Possible extensions" in `CODE.md` has the
  two ways out.

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

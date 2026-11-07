# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [The right-hand side](#the-right-hand-side)
- [Refined forests](#refined-forests)
- [Adaptive refinement](#adaptive-refinement)
- [Truncation](#truncation)
- [The blast wave and the viewer](#the-blast-wave-and-the-viewer)
- [Threads and the integrator](#threads-and-the-integrator)
- [Particles](#particles)
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
  left as decided, and the item is **(open)** in `CODE.md`: a
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
    vanishing derivatives at both ends, `m = P + 1` (`INSPIRAL-PLAN.md`,
    "Left to settle later").
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
  This is a gate of `INSPIRAL-PLAN.md` ("Gates and fallbacks") and goes
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
- 2026-10-30: the cost gate of `INSPIRAL-PLAN.md` ("Gates and
  fallbacks") fired: one projection costs far more than ten times the
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
- 2026-10-30: the pairing reads the state on the host, over the support's
  elements found from `z` (`INSPIRAL-PLAN.md`, "Left to settle later"),
  measured against the `mesh_fold` it replaces (`CODE.md`, "The
  transfer's cost").
- 2026-10-30: M3 of `INSPIRAL-PLAN.md`, a particle at rest and in uniform
  motion, on branch `inspiral-m3-uniform`, started at Erik Schnetter's
  request: `src/inspiral.jl`, `test/inspiral_tests.jl`,
  `bin/uniform_motion.jl` and `bin/symmetry/uniform_motion.sh`. It
  departs from the plan in two places:
  - **The state is a plain vector, not an `ArrayPartition`.** The
    decision of 2026-10-30 ("Decisions" in `INSPIRAL-PLAN.md`) was made
    for OrdinaryDiffEq, before the integrator became IMEXRungeKutta, whose
    stage arithmetic by block owner takes a plain `Array` only and would
    fall back to its broadcast for an `ArrayPartition`. The particles'
    six numbers each go after the field's state vector, owned by thread 1
    in the partition, the first of the two options `CODE.md` listed
    under "Possible extensions"; IMEXRungeKutta needs no change. This
    also settles the plan's "`SVector` or `Vector`" for the particles'
    part (`INSPIRAL-PLAN.md`, "Left to settle later").
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
  `INSPIRAL-PLAN.md` ("Gates and fallbacks") for M4's circle, measured
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
  open ("Left to settle later"), for the owner to review:
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
  Schnetter (`INSPIRAL-PLAN.md`, "Gates and fallbacks"; `CODE.md`,
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
  term. It is also what M1 found an extended particle's bracket to keep
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
- 2026-11-06: CI failed on main from M4 on, on every job, on one
  assertion: the retarded field's `∂ₜΦ` at `t = 0` of hyperbolic motion,
  bounded by `10⁻¹⁵|Φ|`, is `1.2·10⁻¹⁵|Φ|` at `r = 3`. It vanishes
  there, so the value is roundoff, mostly of the retarded time, which
  lies about `r` back; it is under `3·10⁻¹⁷|Φ|` at `r ≤ 1`. The bound
  is now `10⁻¹⁵ max(r, 1)|Φ|`.

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

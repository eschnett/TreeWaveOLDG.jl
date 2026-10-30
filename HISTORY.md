# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [The right-hand side](#the-right-hand-side)
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
- 2026-10-30: found in M2: elements cut by the window's spheres `R₁`
  and `R₂` carry a quadrature error of `10⁻⁸`–`3·10⁻⁷` relative at
  `hₑ = d/8`, since the window is only `C^{P+1}` there. It is static,
  and `INSPIRAL.md`'s tolerance of `10⁻¹²` does not hold on those
  elements.
- 2026-10-30: the pairing reads the state on the host, over the support's
  elements found from `z` (`INSPIRAL-PLAN.md`, "Left to settle later"),
  measured against the `mesh_fold` it replaces (`CODE.md`, "The
  transfer's cost").

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

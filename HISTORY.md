# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [Truncation](#truncation)
- [The blast wave and the viewer](#the-blast-wave-and-the-viewer)
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

## The method

- 2026-10-26: the element tables were checked against the paper's Figs.
  5–7 (`P = 2, 3, 4`), and a 1D implementation using dense global
  matrices was written first as a reference. The package reproduces it
  to the printed digits for `P = 1 … 6`, and the dense operators became
  `test/reference.jl`.

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

## The blast wave and the viewer

- 2026-10-26: Erik Schnetter asked for a 2D visualization like
  TreeWave's radial blast wave, leaving out what this package does not
  support (refinement, backend, centering, type). The blast wave, its
  exact solution and TreeWave's 2D viewer were ported to the nodes; the
  refinement panel (block count and indicator over time) became the
  error and energy over time, and the refinement levels in the radial
  panel became face nodes against interior nodes. SpecialFunctions
  became a dependency, for `besselj0`.

## Dependencies

- 2026-10-26: based on TreeWave's `main` plus its open pull request #8
  ("Fold the diagnostics with TreeAMR 0.2.6's mesh_fold"), so TreeAMR
  0.2.6 is required for `mesh_fold`. Julia 1.13 is the floor, as for the
  other Tree* packages.
- 2026-10-26: CI added, TreeWave's test job without its viewer job (there
  is no `bin/`), with TreeWave's Dependabot and Codecov settings.

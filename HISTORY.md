# TreeWaveOLDG history

How the package got where it is, by topic in `CODE.md`'s order.

**Contents**

- [Scope and non-goals](#scope-and-non-goals)
- [The method](#the-method)
- [Truncation](#truncation)
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

- 2026-10-26: truncation is applied to `∂ₜv` rather than to the state at
  each substep, as the paper describes it, so that the right-hand side
  stays a pure function of `u` and works with OrdinaryDiffEq's
  integrators. The two are the same scheme to roundoff.
- 2026-10-26: the default is the paper's truncated scheme. Measured
  against the plain one, truncation gives order `P` for `u` and `v`
  alike, while the plain scheme gives a smaller and faster-converging
  `u` and an irregular `v` (tables in `CODE.md`, "Convergence").

## Dependencies

- 2026-10-26: based on TreeWave's `main` plus its open pull request #8
  ("Fold the diagnostics with TreeAMR 0.2.6's mesh_fold"), so TreeAMR
  0.2.6 is required for `mesh_fold`. Julia 1.13 is the floor, as for the
  other Tree* packages.
- 2026-10-26: CI added, TreeWave's test job without its viewer job (there
  is no `bin/`), with TreeWave's Dependabot and Codecov settings.

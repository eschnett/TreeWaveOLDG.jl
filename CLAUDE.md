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
| `INSPIRAL-PLAN.md` | anyone building the inspiral | concrete planned work, milestone by milestone, with how one knows each is done; a status line per milestone |
| `PLAN.md` | — | does not exist; other planned work would go there |

To amend the documents, fix `CODE.md` to describe the current state and
record in `HISTORY.md` what changed, when and why. Where the inspiral
code departs from `INSPIRAL.md` or settles one of its open items, that
goes into `CODE.md` and `HISTORY.md` too, never into `INSPIRAL.md`. Keep
each file's table of contents current (`##` and `###` in `CODE.md`,
`INSPIRAL-PLAN.md` and `HISTORY.md`, `##` in `README.md` and here), and
update every reference to a heading you rename, in code comments and
docstrings too.

## What this package is

TreeWave with the finite-difference stencil replaced by the OLDG one
(arXiv:1604.00075), as the unrefined prototype TreeAMR's `CODE.md`
proposes. TreeAMR is the mesh; anything about trees, ghosts or
interpolation belongs upstream ("Scope and non-goals").

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

Each takes under two minutes. The kernels are `@inbounds`, so run the
second one too after touching a kernel or the layout.

The viewer, in its own environment (CairoMakie is not a package
dependency), about 20 seconds; it writes `bin/output/blast_2d.png`:

```bash
julia --project=bin bin/visualize2d.jl
```

The benchmark, in the package environment; on Symmetry,
`sbatch bin/symmetry/benchmark.sh` from a rsynced checkout runs it at
1–64 threads ("Threads on Symmetry" in `CODE.md`):

```bash
julia --project=. -t 4,0 bin/benchmark.jl --scan=4:8
```

The transfer's cost on the first run's source ("The transfer's cost"),
a few minutes on four threads:

```bash
julia --project=. -t 4,0 bin/projection.jl --hdiv=8
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

- **Uniform forests only.** Do not relax `check_layout` to let a refined
  forest through: TreeAMR's coarse-fine ghosts would be wrong ("What a
  refined forest needs").
- **Positions, initial data and norms come from this package, not
  TreeAMR.** Use `node_coordinates`, `fill_at_nodes!` and
  `quadrature_norm`; `coordinates`, `fill_by_coordinates!` and
  `volume_weighted_norm` assume equally spaced cell centres ("Positions
  and quadrature").
- **`N = E(P+1)`, and `N` even** ("The layout").
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
  right-hand side").
- **Pass `d` to the gradient and Laplacian kernels as a `Val`, through
  an `@inline` closure**; a runtime `d` costs 2.5× ("The right-hand
  side").
- **The tables are built in `BigFloat` and rounded once.** Do not build
  them in `T` or convert from the `Float64` ones ("Precision").
- **The time step is `cfl · hₑ/(P+1)²`**, not TreeWave's `cfl · h`
  ("Time step").
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
- **A particle's support is a breakpoint.** Integrate `ρ`, `∇ρ` or the
  source near a particle with the nested rule, never a Gauss or sinh rule
  across the support's edge, where the profile is only `C³`
  ("Quadrature").
- **The transfer looks for no periodic images.** The source's ball and
  every particle's support lie inside the box ("The transfer").
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
- Not registered. TreeAMR comes from the General registry, not from a
  local checkout; IMEXRungeKutta, which is not registered, from its
  GitHub repository through `[sources]` in `Project.toml` and
  `test/Project.toml`.
- CI (`.github/workflows/CI.yml`) runs the suite with
  `--check-bounds=yes` on Linux and macOS, and on four threads on Linux.
  Do not drop its `check_bounds: 'yes'` line ("The right-hand side").
- `Manifest.toml` (root and `bin/`), `bin/output/` and `TODO.md` are
  gitignored. `bin/` has its own Manifest, so after a dependency change
  update both, or the viewer fails on an API the tests already use. `TODO.md` is the owner's
  own list: do not edit it, and move nothing into or out of it.

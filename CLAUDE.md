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
| `PLAN.md` | — | does not exist; concrete planned work, with how one knows each step is done, would go there |

To amend the documents, fix `CODE.md` to describe the current state and
record in `HISTORY.md` what changed, when and why. Keep each file's
table of contents current (`##` and `###` in `CODE.md`, `PLAN.md` and
`HISTORY.md`, `##` in `README.md` and here), and update every reference
to a heading you rename, in code comments and docstrings too.

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

Each takes about 50 seconds. The kernels are `@inbounds`, so run the
second one too after touching a kernel or the layout.

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
- **Truncation projects `∂ₜv`, never the state.** Writing into `u` inside
  the right-hand side would break RK4 silently; `initial_state` projects
  the initial `v` ("Truncation").
- **The tables are built in `BigFloat` and rounded once.** Do not build
  them in `T` or convert from the `Float64` ones ("Precision").
- **The time step is `cfl · hₑ/(P+1)²`**, not TreeWave's `cfl · h`
  ("Time step").
- **The `u` and `v` of a state vector interleave block by block**; use
  `statearray` and `selectdim`, not halves of the vector.
- **The asserted rates are measurements.** If one moves, re-measure and
  update "Measured results" rather than widen a tolerance.

## Conventions

As in TreeWave and TreeAMR: 4-space indent, wrap at about 80 columns,
`return` on the last line of a non-trivial function, Unicode in
mathematical contexts, prose-first docstrings, testset names that are
claims, each testset opening with a comment naming the failure it
guards.

## Repository facts

- Not yet a git repository, and not registered. TreeAMR comes from the
  General registry, not from a local checkout.
- `Manifest.toml` and `TODO.md` are gitignored. `TODO.md` is the owner's
  own list: do not edit it, and move nothing into or out of it.

# Planned work outside the inspiral

Work that is proposed but not part of `INSPIRAL-PLAN.md`'s milestones,
each item with what it would save, measured where it says so, and how
one knows it is done. Settled parts go into `CODE.md`, and what changed
into `HISTORY.md`, as for the rest of the documents (`CLAUDE.md`, "Where
things go").

**Status.** Every item is **(proposed)**. The cube's flat path, the
first of them, is written (2026-11-09; `CODE.md`, "The seven-patch
ball"), and what it left is the first item below.

**Contents**

- [The full geometry's cost](#the-full-geometrys-cost)
- [The shells' truncation tables](#the-shells-truncation-tables)
- [Memory](#memory)
- [Not worth it now](#not-worth-it-now)

The measurements below are from 2026-11-09, after M8b, on the laptop
(Apple silicon, four threads unless it says one), `P = 3`, blocks of
`16³`. "The local ball" is `bin/nested_ball.jl --levels=2 --w0=2
--hdiv=4 --lambda=4`: a cube of `Lc = 4` holding two nested levels and
shells to `R1 = 12`, 2.06 M nodes, 76 % of them in the shells.

## The full geometry's cost

On the same nested-box forest, the full geometry with the identity map
cost 7.7 times the flat path per right-hand side, nearly all of it in the
two kernels. The cube's flat path (`CODE.md`, "The seven-patch ball",
written 2026-11-09) brought it to 1.9 times. By pass, in ms:

| | `u` exchange | gradient kernel | gradient exchange | Laplacian kernel | whole |
|---|---|---|---|---|---|
| flat | 0.22 | 0.41 | 0.26 | 0.42 | 1.51 |
| full, before the cube's flat path | 0.22 | 2.59 | 0.59 | 6.62 | 11.42 |
| full, with it | 0.22 | 0.57 | 0.66 | 1.13 | 2.92 |

On the local ball the right-hand side went from 23.7 to 13.6 ns per node;
its shells, 76 % of the nodes, take the general path and are now most of
it. At M3b's scale (0.06 s per right-hand side on `10⁸` nodes flat,
against 0.19 s for the source; `INSPIRAL-PLAN.md`, M7's "The box") the
cube's part is now about twice the flat cost.

- **The rest of the cube's cost.** The flat path was to bring the cube
  to within about 20 % of the flat cost, and stops at 1.9 times: the
  Cartesian gradient's exchange, with ghosts on every face, costs 2.5
  times the three per-dimension ones, and the Laplacian kernel 2.7
  times the flat one, though it does the same arithmetic. It reads one
  component of a set of three with ghosts on every face, where the flat
  kernel reads a set of one with ghosts along `d` only; why that costs
  2.7 times is not measured. Done when the cause is measured, and either
  removed or recorded as the price of the shells' glue.
- **A cheaper general Laplacian on a full geometry.** The general
  Laplacian kernel costs 16 times the flat one for about three times the
  flops: it forms `F^d = S[d] ⋅ V` at every point of every window, 18 per
  node, each reading three coefficients and three components. Forming
  `F^d` once per owned node, and at the ghosts only on the face planes,
  would read each coefficient once. The shells take this path, and are
  now most of the ball's right-hand side. Done when the shells' right-hand
  side per node falls by a measured factor and the energy tests still
  hold to `1e-13`.

## The shells' truncation tables

The build, per element of a shell, on one thread, in ms:

| number type | volume `√g` | Gram matrix | Cholesky | total | rounded tables |
|---|---|---|---|---|---|
| `BigFloat`, 512 bits (now) | 1.84 | 1.26 | 0.68 | 3.78 | — |
| `Float64x2` (MultiFloats) | 0.19 | 0.37 | 0.018 | 0.58 | identical to `BigFloat`'s, all ~200 000 entries |
| `Float64` | 0.18 | 0.35 | 0.003 | 0.54 | wrong by up to `10¹¹` ulp |

`Float64x2` costs about what `Float64` does, so its build is dominated
by the build loops' allocations (arrays and generators per element in
`gram_sumfact`), not by arithmetic. Start-up, in CPU time (divide by the
thread count; each block is its own task):

| | elements | `BigFloat` (now) | transposed | `Float64x2` | both |
|---|---|---|---|---|---|
| the local ball, `n = 4`, `m = 2` | 12 288 | 46 s | 15 s | 7 s | 1.4 s |
| M3b's scale, `nested_ball.sh`, `n = 16`, `m = 4` | 393 216 | 25 min | 8 min | 3.8 min | 43 s |

On 64 threads the M3b-scale build is about 23 s now, against runs of
hours; the saving matters most on the laptop and in the suite, whose
seven-patch tests take about 100 s, part of it these builds.

- **Non-allocating build loops** first: with the arithmetic in
  `Float64x2` close to `Float64`'s, this is where the time is.
- **Build in `Float64x2` when `T` is at most `Float64`,** `BigFloat`
  otherwise. The shells' tables need only `+ − × /` and `sqrt`, through
  ForwardDiff's Jacobian, `det`, `inv` and Cholesky, which MultiFloats
  has to about `1e-32`; it has no `sin`, `cos`, `tan`, `atan` or `cospi`,
  which no table build needs (the Gauss–Lobatto nodes' `cospi` start
  stays in `BigFloat`). It adds MultiFloats as a dependency and amends
  the rule of `CODE.md`'s "Precision" and `CLAUDE.md`'s sharp edge
  ("The tables are built in `BigFloat` and rounded once").
- **Transpose one shell's tables to the other five.** Each shell's `√g`
  is a reference shell's with its axes permuted and its radial axis
  reversed for `s = −1`, so the weights and the Gram matrix carry over by
  a signed permutation, exactly. The Cholesky factor depends on the
  basis's order and is recomputed. Each shell is also symmetric under
  reflecting either tangential axis and swapping the two, a group of 8,
  which would cut the volume and Gram steps by up to 8 more.

Done when the build's rounded tables equal today's bit for bit (a test),
and the start-up is measured as above.

## Memory

The local ball's problem by part, in bytes per node over all its nodes:

| part | bytes per node | × state vector |
|---|---|---|
| state vector `u`, `v` | 16.0 | 1.0 |
| state set's own array | 16.0 | 1.0 |
| `u` with ghosts | 11.4 | 0.71 |
| Cartesian gradient with ghosts | 34.2 | 2.14 |
| coefficients `√g`, `S` (10 per node) | 80.0 | 5.0 |
| truncation: Cholesky factors | 69.4 | 4.34 |
| truncation: weights | 6.1 | 0.38 |
| truncation: scratch | 2.6 | 0.16 |
| whole problem | 219.9 | 13.7 |

Per shell node the tables are 102 bytes, 6.4 times the state vector; the
integrator adds the state and five scratch arrays, 96 bytes per node. At
M3b's scale (about `7·10⁷` nodes, `2.5·10⁷` in the shells) the tables
are about 2.6 GB of a 256 GB node, so memory is not what binds there.

- **Pack the Cholesky factors' upper triangles.** Each is stored as a
  full `27 × 27` matrix, 351 of whose 729 entries are zero: 37 bytes
  per shell node, half the largest table, for some index arithmetic in
  the kernel.
- **Keep the cube's coefficients per block.** The coefficient set holds
  10 numbers at every node, the cube's 0.49 M included, where `√g` and
  `S` are constants: 80 bytes per cube node, 39 MB on the local ball.
  The cube's flat path reads only the block's first node's
  coefficients already.
- **Share the tables at run time** across the six shells and the
  in-shell symmetries: up to about 48 times fewer, but the kernel would
  read each element's nodes through its block's axis map (a permutation
  and reversals) and write them back the same way.
- **Purely radial outer shells** (the 13-patch ball, `HISTORY.md`,
  "Multi-patch forests"): with `r = r(ξ_k)` alone, `√g = r² r′ ω(ξ_j, ξ_l)`
  factorizes into a radial and an angular part, so the truncation is a
  one-dimensional rank-one pair along the radius times a two-dimensional
  projection on the face: about 1600 multiply-adds per element at
  `P = 3` against 4200, a `P⁴ = 81`-entry Gram factor against
  `P⁶ = 729`, and one face's angular tables shared by every radial
  element and shell. `ω` does not factorize on a cubed sphere, so it is
  not a full tensor product. It would also remove the shear near the
  cube's corners from all but the layer next to the cube.

## Not worth it now

- **A barrel-distorted layer of 26 patches** around a Cartesian cube,
  ending on a bulged cube, with six shells beyond it to the sphere, 33
  patches in all. The 26 patches cannot reach the sphere themselves: an
  edge or corner patch with two or three faces on a smooth surface
  degenerates there. The shells' fastest nodes are at the cube's corners
  on their inner face, 3.1 times the cube's logical wave speed at
  `R1 = 3Lc` (2.6 near the edges, 1.5 at the faces' middles), from the
  shear of their radial lines against the flat face; the layer would
  soften that shear, by an unmeasured factor. But with three or more
  levels in the cube, M8's layout, the cube's finest level sets the step,
  so the shells' speed costs nothing. Worth revisiting with local time
  stepping, with a cube of one or two levels, or if M8a's runs show the
  skewed elements near the cube's edges cost accuracy.

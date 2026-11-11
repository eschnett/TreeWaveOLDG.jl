# Planned work outside the inspiral

Work that is proposed but not part of `INSPIRAL-PLAN.md`'s milestones,
each item with what it would save, measured where it says so, and how
one knows it is done. Settled parts go into `CODE.md`, and what changed
into `HISTORY.md`, as for the rest of the documents (`CLAUDE.md`, "Where
things go").

**Status.** Every item is **(proposed)**. The cube's flat path, the
first of them, is written (2026-11-09; `CODE.md`, "The seven-patch
ball"), and what it left is under "The full geometry's cost". The shells' tables are
built in `Float64x2` (2026-11-10; `CODE.md`, "Precision"), and what that
left is in "The shells' truncation tables". The purely radial outer
shells, the thirteen-patch ball with its factored truncation, are
written (2026-11-10; `CODE.md`, "The thirteen-patch ball"), and what
they left is under "Memory". The field on a GPU is written (2026-11-10;
`CODE.md`, "On a device"), and what it left is under "The benchmark
ball" and in "Particles on a device". The benchmark ball
(2026-11-10; `CODE.md`, "Threads on Symmetry") ranks what is left by
what it would save on the runs to come; that ranking is the first
section below.

**Contents**

- [The benchmark ball](#the-benchmark-ball)
- [Particles on a device](#particles-on-a-device)
- [The full geometry's cost](#the-full-geometrys-cost)
- [The shells' truncation tables](#the-shells-truncation-tables)
- [Memory](#memory)
- [Not worth it now](#not-worth-it-now)

The measurements in the sections after the first are from 2026-11-09,
after M8b, on the laptop (Apple silicon, four threads unless it says
one), `P = 3`, blocks of `16³`. "The local ball" is `bin/nested_ball.jl --levels=2 --w0=2
--hdiv=4 --lambda=4`: a cube of `Lc = 4` holding two nested levels and
shells to `R1 = 12`, 2.06 M nodes, 76 % of them in the shells.

## The benchmark ball

The physical runs keep a constant angular resolution everywhere but near
the centre. The benchmark ball is sized so: 16 elements across each
patch angularly, `hₑ = 1/2` in the cube and radially in the shells, so
`Lc = 4`, `Rm = 3Lc`, `R1 = 100`, `1.91·10⁷` nodes, nine tenths of them
in the outer shells. On a Symmetry node at 64 threads a step takes
0.31 s, 3.8 times the flat box per node, and a run to `t ≈ 224` about
3.8 hours (`CODE.md`, "Threads on Symmetry"). Of a step, the
right-hand sides are 63 %, the truncations 30 %, the stage arithmetic
6 %. By what each would save, largest first:

- **A GPU. Written** (2026-11-10; `CODE.md`, "On a device"): the field
  runs on CUDA and Metal and holds to the host's results, and on one
  H200 the benchmark ball steps 5.7 times faster than on 64 cores, 54 ms
  a step, 40 minutes to `t ≈ 224` (`CODE.md`, "Threads on Symmetry").
  What it left:
  - **The flat kernels on a device.** The Laplacian pass costs on the
    H200 what it costs on 64 cores, 0.23 ns per node, about a tenth of
    the device's bandwidth, and with the gradient pass it is nearly 90 %
    of the flat box's right-hand side, which gains only 2.6 times. Both
    read the stencil through a run-time tuple index and form their
    position with `@index(Global, NTuple)`; TreeAMR's flat launch with
    precomputed inverses made its own copies and fills 1.3 to 2.4
    times faster. It matters for the cube, and for nested boxes, which
    are all flat path. Done when the flat Laplacian and gradient passes
    per node fall by a measured factor on the H200 and the CPU's stay
    where they are.
  - **Several GPUs** through MPI, which TreeAMR has, for what does not
    fit one H200 (nested boxes at `d/16`).
- **The outer shells' truncation kernel.** 1.19 ns per node at 64
  threads, 30 times the flat box's, for about 25 multiply-adds: it is
  bound by its memory accesses. Each element's numbers in registers
  rather than the scratch array, and compile-time indices, should bring
  it near the right-hand side's arithmetic intensity. It would save up to
  a quarter of a step on the CPU, and matters more on a GPU. Done when the
  benchmark ball's truncation per node falls by a measured factor, with
  the factored truncation still equal to the per-element one to `1e-14`.
- **The outer shells' Laplacian.** Their metric is block-diagonal and
  their frame separable, `S[d] = R_d(r) A_d(ξ_⊥)`: inside a block the
  radial flux is `|S[k]|²/√g D_k u` and needs no Cartesian gradient, and
  the coefficients could come from the angular tables the truncation
  already shares, instead of 80 bytes per node. This is "A cheaper
  general Laplacian" below, specialized to the shells that are now nine
  tenths of the nodes. With the truncation, up to about 2.4 times on a
  step. Done when the outer shells' right-hand side per node falls by a
  measured factor and the energy tests hold to `1e-13`.
- **A graded radial map in the outer layer.** `hₑ = 1/2` radially out to
  100 is four times what `λ = 16` needs. A spacing growing linearly from
  1/2 at `Rm` to 2 at `R1` takes 81 radial elements instead of 176, 2.2
  times fewer nodes, at no cost in the step; the factored truncation
  already takes any radial factor. Whether the physics allows it is the
  owner's call. Done when `r(ξ)` is not linear, a test holds the frame
  and the factors, and the plane wave still converges at order `P`.
- **The 64-thread anomaly** of the variant with wider inner shells
  (`CODE.md`, "Threads on Symmetry"): rerun, and find the cause if it
  persists.

Not worth it on this ball: the set-up (half a minute), the stage
arithmetic (6 %), load balance (the blocks are nearly all alike), and
local time stepping, since nested levels down to `hₑ = 1/8` for the
binary would leave the shells' own step only 1.3 to 2.2 times the finest
level's.

## Particles on a device

The field runs on a GPU (`CODE.md`, "On a device"); the particles do
not, and `InspiralProblem` refuses a device. They have to follow, decided
with the owner on 2026-11-10. At `d/8` on M4's five-level forest,
75.5 M nodes, `project_source!` takes 0.585 s per right-hand side at
order 3 on 64 cores against 0.073 s for the wave equation, about 85 % of
a 2.71 s step; the binary's projection off the grid takes 1.24 s
(`CODE.md`, "Prescribed paths" under "Measured results"). An H200 job
gets 12 to 16 host cores, on which the host passes would take about
four times as long as on a CPU node. So all
per-node and per-point work goes on the device, and only `O(particles)`
work stays on the host.

Today, per right-hand side at `d/8` with two particles, the far rule
evaluates the source at about 6.7 M points in two KernelAbstractions
kernels, the graded rule at 2.6 M on host threads, the nested rule at
1.6 to 2.5 M on host threads after a serial, allocating plan; the
averaging over shared nodes is a host loop, and the pairing, which reads
the forces, a serial host call over 1 to 8 elements. One source
evaluation costs about 5 µs on one CPU thread at order 3 for both
particles, so the projection is bound by arithmetic, which a GPU has
plenty of in `Float64`.

- **B0, the gate. Measured** (2026-11-10; `CODE.md`, "On a device" and
  "Threads on Symmetry"). As first ported the far rule ran 2.4 times
  faster on one H200 than on 64 cores. Two changes took it to 7.0 times,
  0.122 s against 0.856 s, equal to the host's to `1.7·10⁻¹¹`: the
  source's coefficient tables in a device array (`device_source`), which
  every work item had copied into its local memory, and the far rule
  launched over a list of its elements. Still short of ten times. What
  is left is the source itself, 87 % of whose far-rule points are in the
  window's transition at 15 ns each, and in it the lab slice's order-3
  term, a generated function whose frame takes 255 registers and 24 KB
  of local memory per thread. The ways on, for the owner to choose:
  - **Accept seven times** and go on to B1. The projection would then
    cost about what the field does on the device, and an inspiral step
    at `d/8` perhaps a fifth of the 64-core node's.
  - **Reshape the generated order-3 term** for a device: Horner chains
    on the dual numbers instead of `evalpoly` over tuples of lifted
    coefficients, or the term split into launches by group. This is a
    change to `derivations/generate.wl`, and the host's results must stay
    bit for bit.
  - **Truncate the order-3 polynomials far from the particles**, which
    `CODE.md` leaves open for want of a bound; the transition is far from
    them.
  Done when one of these is chosen, and, for the last two, the far rule
  on the H200 is timed again.
- **B1, the right-hand side's plumbing.** The `6K` particle numbers at
  the end of the state are copied as one tail rather than read one by
  one; `inspiral_rhs!`'s threaded passes over the ball's blocks (zero
  `u̇` there, add the source to `v̇`, restore `u̇ = v`) become broadcasts
  or kernels. Done when an inspiral right-hand side with only the far
  rule runs on CUDA and matches the host to rounding.
- **B2, the plan per call.** The elements' classification into far,
  graded and nested runs in a kernel over the ball's blocks instead of a
  host loop; the jets are uploaded to a device buffer each call. Done
  when the plan's cost per call is measured and small beside the far
  rule's.
- **B3, the graded rule.** The far rule's two kernels with the number
  of points per dimension a `Val`, one launch per band. Done when the
  graded rule matches the host's to rounding on CUDA.
- **B4, the nested rule.** The host plans the segments of each nested
  element's lines (`line_fold`, `support_breaks`) into preallocated flat
  arrays, without the allocations of today's plan; one kernel evaluates
  all nested points; each element's sums are taken in a fixed order and
  written to its nodes. Done when the nested rule matches the host's to
  rounding on and off the grid, and the projection's three rules are
  timed on an H200.
- **B5, the averaging and the forces.** `average_shared!` becomes a
  kernel over `SharedNodes`, uploaded once. The pairing stays on the
  host: it reads the density, not the source, over at most a few dozen
  elements, whose nodal values (the support's elements and their
  sharers') are copied down each call. `InspiralProblem` then accepts a
  device. Done when the inspiral tests pass on CUDA in `Float64`,
  matching the host to rounding, and a `d/8` right-hand side is timed on
  an H200. The particles need `Float64`, so Metal cannot run them.

Not planned here: nested boxes at `d/16`, 151 GB, exceed one H200's
141 GB; they would need several GPUs through MPI, which TreeAMR has, or
the ball's smaller layout.

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
| `BigFloat`, 512 bits (before 2026-11-10) | 1.84 | 1.26 | 0.68 | 3.78 | — |
| `Float64x2` (MultiFloats; now) | 0.19 | 0.37 | 0.018 | 0.58 | identical to `BigFloat`'s, all ~200 000 entries |
| `Float64` | 0.18 | 0.35 | 0.003 | 0.54 | wrong by up to `10¹¹` ulp |

`Float64x2` costs about what `Float64` does, so its build is dominated
by the build loops' allocations (arrays and generators per element in
`gram_sumfact`), not by arithmetic. Start-up, in CPU time (divide by the
thread count; each block is its own task), estimated from the table
above:

| | elements | `BigFloat` (before) | transposed | `Float64x2` (now) | both |
|---|---|---|---|---|---|
| the local ball, `n = 4`, `m = 4` | 24 576 | 93 s | 31 s | 14 s | 2.9 s |
| M3b's scale, `nested_ball.sh`, `n = 16`, `m = 4` | 393 216 | 25 min | 8 min | 3.8 min | 43 s |

On 64 threads the build is bound by the garbage collector instead
(Symmetry, EPYC 7543, 2026-11-10): at M3b's scale the `BigFloat` build
took 1051 s, 884 s of it collecting 2.3 TB of allocations, against the
25 min / 64 = 23 s its arithmetic would take; in `Float64x2` it takes
93 s, 74 s of it collecting 204 GB.

**Written:** the build in `Float64x2` when `T` is at most `Float64`
(`table_type`), whose rounded tables a test holds to `BigFloat`'s bit for
bit. On the local ball's 24 576 shell elements it takes 11.9 s on one
thread against 88.6 s, 0.48 ms per element, close to the estimate above.

- **Non-allocating build loops:** with the arithmetic in `Float64x2`
  close to `Float64`'s, the allocations are where the build's time is
  now, and on many threads the collector's: 74 of the 93 s at M3b's
  scale. The volume, the weights and the Gram matrix in preallocated
  per-task arrays, and `gram_sumfact`'s generators as loops, would take
  most of it; the rest of the problem's build (fill, speeds) allocates
  another 117 GB and collects for about 40 s. Done when the M3b-scale
  build's collection time is a small part of it, measured as above.
- **Transpose one shell's tables to the other five.** Each shell's `√g`
  is a reference shell's with its axes permuted and its radial axis
  reversed for `s = −1`, so the weights and the Gram matrix carry over by
  a signed permutation, exactly. The Cholesky factor depends on the
  basis's order and is recomputed. Each shell is also symmetric under
  reflecting either tangential axis and swapping the two, a group of 8,
  which would cut the volume and Gram steps by up to 8 more.

Done when the transposed build's rounded tables equal the direct
build's bit for bit (a test), and the start-up is measured as above.

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
- **What the purely radial outer shells left** (the thirteen-patch
  ball, written 2026-11-10; `CODE.md`, "The thirteen-patch ball"). Its
  outer shells take the factored truncation, a radial pair times an
  angular projection whose tables all six shells share, but its inner
  shells keep the per-element tables, and so does the seven-patch ball.
  Most of the shells' nodes can be in the outer layer, where the tables
  are nearly free, but the inner layer cannot be thin: its step is 0.15
  of the cube's at `Rm = 2Lc` and 0.32 at `3Lc`, the seven-patch ball's,
  so `Rm ≈ 3Lc` (measured 2026-11-10; `CODE.md`, "The thirteen-patch
  ball" under "Measured results"). The radiative pulse leaves 1.4 times
  more energy behind than on the seven-patch ball at `R1 = 3`; the
  dipole and a larger `R1` are not measured. Equiangular angles would even the outer layer's tangential
  spacing; they need the inner layer to blend to an equiangular sphere.

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

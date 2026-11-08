# TreeWaveOLDG.jl

[![CI](https://github.com/eschnett/TreeWaveOLDG.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeWaveOLDG.jl/actions/workflows/CI.yml)

`TreeWaveOLDG` solves the scalar wave equation in 2nd-order form with
Miller and Schnetter's operator-based local discontinuous Galerkin method
(OLDG, [arXiv:1604.00075](https://arxiv.org/abs/1604.00075)), on a
[TreeAMR](https://github.com/eschnett/TreeAMR.jl) mesh. It is
[TreeWave](https://github.com/eschnett/TreeWave.jl) with the
finite-difference stencil replaced by the OLDG one.

**Contents**

- [What it solves](#what-it-solves)
- [Installation](#installation)
- [Example](#example)
- [Status](#status)
- [Design](#design)

## What it solves

That is `∂ₜu = v`, `∂ₜv = ∇²u` on a periodic box — or a reflecting
(solid-wall) box — in any number of dimensions, or on an annulus of four
patches in polar coordinates. Space is cut into elements
of `P + 1` Gauss–Lobatto nodes, with both face nodes of every element
stored. The derivative at a node reads its own element and, at the two
face nodes only, the neighbour's face node across the face. The Laplacian
is that derivative applied twice. The method is a DG finite element method
written as a finite-difference stencil with non-constant coefficients.

## Installation

The package is not registered. From a checkout:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Example

```julia
using TreeWaveOLDG
r = wave_errors(Val(2); P=4, E=2, roots=8)
r.l2, r.linf          # error of the standing sine mode after a quarter period
```

`P` is the polynomial order, `E` the number of elements per block per
dimension (so a block holds `N = E(P+1)` points), and `roots` the number
of blocks per dimension.

The same on a reflecting box (solid walls, Neumann `∂ₙu = 0`) uses the
`reflecting` keyword and the standing `cos` mode:

```julia
r = reflect_errors(Val(1); P=3, E=2, roots=4)
r.l2, r.linf, r.energy / r.energy0   # error, and energy is conserved
```

A static two-level forest refines the middle half of the box once
(`refined=true`), or any region given as one interval per dimension:

```julia
r = wave_errors(Val(2); P=3, E=2, roots=8, refined=true)
r.l2, r.energy / r.energy0           # the error, and energy is conserved
```

`E` must be even, so that a fine block covers whole elements of its
parent. A vector of regions refines level by level, one region per
level; `nested_cubes` gives the nested cubes about the centre that the
inspiral runs on, here three levels, the finest of half-width 1/8:

```julia
f = wave_forest(Float64, Val(3), 3, 2; roots=4, L=1.0,
                refined=nested_cubes(Val(3), 1/8, 3))
```

The annulus `1 ≤ r ≤ 2` is TreeAMR's ring of four patches glued at
their faces (its M17), with its polar metric; the rotating Neumann
Bessel mode turns a quarter of the way round, across every glued face:

```julia
r = annulus_errors(; P=3, E=2, roots=(4, 2), m=2)
r.l2, r.energy / r.energy0           # error, and energy is conserved
```

An adaptive forest follows the 2D blast wave's ring, refining where a
modal indicator says the elements do not resolve the data and
coarsening where nothing is left. The thresholds, the level cap and the
time between regrids have no defaults; these are the recommended ones
for `P = 3` up to two levels:

```julia
r = adaptive_blast(; P=3, E=2, roots=4, maxlevel_cap=2,
                   refine_tol=1e-2, coarsen_tol=1e-5, chunk=0.05, t_end=0.2)
r.err, r.nregrids, r.node_steps   # error, regrids, work
```

## Status

A prototype. It runs on a periodic or reflecting forest, uniform, with
a static refined region, or adaptive, in any number of dimensions, at
any order `P ≥ 1`, and converges at order `P`. Two
initial conditions are provided, a standing sine mode and a 2D radial blast
wave, and a viewer draws the second, on a uniform or an adaptive forest:

```bash
julia --project=bin bin/visualize2d.jl
julia --project=bin bin/visualize2d.jl --adaptive
```

On a reflecting box the wall is a Neumann (solid, `∂ₙu = 0`) boundary,
an exact discrete symmetry whose energy is conserved to the integrator's own
loss.

On TreeAMR's forests of patches (its M17, in TreeAMR 0.2.8) the wave
equation runs in a diagonal metric. On the four-patch annulus it
converges at order `P`, and the discrete energy is an exact invariant
across the glued faces and the walls, the truncation being orthogonal in
the metric's volume element. A second viewer draws a pulse crossing the
glued faces and the rotating Bessel mode's error:

```bash
julia --project=bin -t 4,0 bin/visualize_annulus.jl
```

The mesh is TreeAMR's, with its element centering (TreeAMR 0.2.8), and
across a coarse-fine face the coupling keeps the discrete energy an
exact invariant. The L2 error converges at order `P` on a refined
forest; in two and more dimensions the largest pointwise error loses
part of an order at the coarse-fine faces, an open question recorded in
[CODE.md](CODE.md). On the adaptive blast wave the error is the uniform
run's at the finest spacing, to 0.3%, at 61–90% of its work, with
thresholds three orders apart. Partly reflecting or outflow boundaries,
and a device run, are not there yet.

The first parts of [two inspiralling scalar particles](INSPIRAL.md) exist
([INSPIRAL-PLAN.md](INSPIRAL-PLAN.md) has the milestones): a scalar
particle of physical size much smaller than an element, its analytic
field, the transfer of its source to the grid and of the grid's field
back to it, and a particle on a prescribed path stepped together with the
remainder field. A particle at rest or in uniform motion keeps its exact
remainder to the discretization error, which converges at order `P`, in
the part of the periodic box the error from its faces has not reached;
the particle in uniform motion is

```bash
julia --project=. -t 4,0 bin/uniform_motion.jl --hdiv=4 --tend=0.125
```

The particles do not yet move under their forces. They run on uniform
forests and on static nested boxes whose finest level holds their
source, which moves the box's faces out: a particle at rest on five
levels stays clean to the end of a run of `t = 20`, and the coarse-fine
faces send back nothing above the discretization error.

A particle on a prescribed path with known acceleration, a slow circle
or hyperbolic motion, runs against the retarded field of a point
charge. On the circle the remainder's force converges to the
radiation-reaction force along the motion and to zero across it, now
that the particle field carries the second-order term of the boosted
static field's lab-slice re-expansion. The source is written and the
force read through the field made continuous by averaging the nodes
that elements share, so that the force stays smooth when the particle
crosses an element face:

```bash
julia --project=. -t 4,0 bin/prescribed.jl --path=hyperbolic --hdiv=2 --levels=3 --tend=0.25
```

## Design

See [CODE.md](CODE.md) for the design and the measured results, and
[HISTORY.md](HISTORY.md) for how it got there.

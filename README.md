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
(solid-wall) box — in any number of dimensions. Space is cut into elements
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

## Status

A prototype. It runs on a uniform periodic or reflecting forest, in any
number of dimensions, at any order `P ≥ 1`, and converges at order `P`. Two
initial conditions are provided, a standing sine mode and a 2D radial blast
wave, and a viewer draws the second:

```bash
julia --project=bin bin/visualize2d.jl
```

On a reflecting box the wall is a Neumann (solid, `∂ₙu = 0`) boundary,
an exact discrete symmetry whose energy is conserved to the integrator's own
loss. Mesh refinement, partly reflecting or outflow boundaries, and a device
run are not there yet.

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

The particles do not yet move under their forces.

## Design

See [CODE.md](CODE.md) for the design and the measured results, and
[HISTORY.md](HISTORY.md) for how it got there.

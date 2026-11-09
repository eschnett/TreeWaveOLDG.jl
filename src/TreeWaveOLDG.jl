"""
    TreeWaveOLDG

The scalar wave equation in 2nd-order form,

    ∂ₜu = v
    ∂ₜv = ∇²u                    (wave speed c = 1)

discretized with Miller and Schnetter's operator-based local discontinuous
Galerkin method (OLDG, arXiv:1604.00075) on a
[TreeAMR](https://github.com/eschnett/TreeAMR.jl) mesh. It is TreeWave with
the finite-difference stencil replaced by the OLDG one: elements of `P + 1`
Gauss–Lobatto nodes, both face nodes stored, a wide derivative that reads
one point past its element, and the Laplacian as that derivative applied
twice.

It runs on TreeAMR's element-centered field sets, whose points are the
elements' Gauss–Lobatto nodes, with TreeAMR's element-local operators
across coarse-fine faces, so the forest may be refined, statically or
following the solution. The particle transfer of the inspiral needs the
finest level under the source's ball.

The reference element is in `element.jl`, the node layout and quadrature
in `layout.jl`, the right-hand side in `evolution.jl`, the time integrator
in `stepping.jl`, the map of a forest of patches to physical space in
`geometry.jl`, the refinement criterion in `refinement.jl`, and the initial
conditions, a standing sine mode, a radial blast wave and the rotating
Bessel mode of the four-patch annulus, in `sinewave.jl`, `blast.jl` and
`annulus.jl`. The analytic particle of the inspiral (`INSPIRAL.md`) is
in `particle.jl`, on top of the closed forms that `derivations/` generates
into `particle_forms.jl`, and its transfer to and from the grid in
`transfer.jl`; the coupled problem, the remainder stepped with the
particles, is in `inspiral.jl`, and the prescribed paths of its
validation, with the retarded field of a point charge, in
`prescribed.jl`. See `CODE.md` in the
package root for the design and the measurements.
"""
module TreeWaveOLDG

using TreeAMR

import ForwardDiff
using IMEXRungeKutta: IMEXProblem, RK4, init, solve!, step!
using KernelAbstractions: @kernel, @index, @Const, Backend, CPU, get_backend,
                          allocate, synchronize
using LinearAlgebra: Symmetric, cholesky, det, dot, eigvals
using SpecialFunctions: besselj0
using StaticArrays: SMatrix, SVector, push, setindex

# The reference element
export Element, gll_nodes_weights

# Layout and quadrature
export node_coordinates, fill_at_nodes!, quadrature_norm

# Evolution system
export WaveProblem, wave_rhs!, truncate_state!, truncate_stage!, initial_state,
       wave_energy, wave_norm, oldg_timestep, convergence_rate

# Geometry: multi-patch forests in a metric
export Geometry, PolarAnnulus, annulus_forest, geometry_terms, physical_coordinates,
       annulus_wavenumber, annulus_exact, annulus_errors

# Geometry: the seven-patch ball, in a full metric
export FullGeometry, SevenPatchBall, sevenpatch_forest, geometry_frame, PlaneWave,
       sevenpatch_exact, sevenpatch_errors, RadiativeBoundary, OutgoingPulse,
       radiative_errors

# Time integration
export state_partition, wave_integrator

# Initial condition: standing sine mode
export wave_omega, wave_exact, wave_forest, nested_cubes, wave_errors,
       reflect_omega, reflect_exact, reflect_errors

# Adaptive refinement
export modal_indicator, mode_amplitude, refine_mark, refine_flags, refinement_buffer

# Initial condition: radial blast wave
export blast_initial, blast_reference, blast_radial_table, blast_exact,
       uniform_blast, adaptive_blast

# The analytic particle
export ParticleModel, Trajectory, trajectory_at, self_inertia, particle_density,
       boosted_static, near_field, lab_slice, particle_field, particle_rate,
       source_bracket, Window, window_terms, ParticleSource, particle_source

# The transfer between particles and grid
export TransferTables, gauss_nodes_weights, sinh_rule, ParticleSupport, project_source!,
       projection_scratch, support_elements, pairing, reconstruct, reconstruct_gradient,
       ball_rule, SharedNodes, average_shared!

# The coupled problem
export InspiralProblem, Prescribed, uniform_motion, field_state, particle_state,
       inspiral_initial_state, inspiral_rhs!, inspiral_stage!, inspiral_partition,
       inspiral_integrator, particle_forces, ball_norm

# Prescribed paths
export circular_motion, hyperbolic_motion, retarded_field, point_model

include("precision.jl")
include("device.jl")
include("element.jl")
include("layout.jl")
include("geometry.jl")
include("evolution.jl")
include("stepping.jl")
include("refinement.jl")
include("sinewave.jl")
include("blast.jl")
include("annulus.jl")
include("sevenpatch.jl")
include("particle_forms.jl")
include("particle.jl")
include("transfer.jl")
include("inspiral.jl")
include("prescribed.jl")

end

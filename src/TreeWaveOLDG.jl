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

This is the *unrefined prototype* TreeAMR's `CODE.md` proposes ("Element-
local DG stencils"): it runs on a uniform forest with the mesh as it is,
to judge the method on its equations before TreeAMR grows a Gauss–Lobatto
centering and element-local transfer operators.

The reference element is in `element.jl`, the node layout and quadrature
in `layout.jl`, the right-hand side in `evolution.jl`, the time integrator
in `stepping.jl`, and the two initial
conditions, a standing sine mode and a radial blast wave, in `sinewave.jl`
and `blast.jl`. The analytic particle of the inspiral (`INSPIRAL.md`) is
in `particle.jl`, on top of the closed forms that `derivations/` generates
into `particle_forms.jl`, and its transfer to and from the grid in
`transfer.jl`; the coupled problem, the remainder stepped with the
particles, is in `inspiral.jl`. See `CODE.md` in the
package root for the design and the measurements.
"""
module TreeWaveOLDG

using TreeAMR

import ForwardDiff
using IMEXRungeKutta: IMEXProblem, RK4, init, solve!, step!
using KernelAbstractions: @kernel, @index, @Const, Backend, CPU, get_backend,
                          allocate, synchronize
using LinearAlgebra: dot
using SpecialFunctions: besselj0
using StaticArrays: SMatrix, SVector, push, setindex

# The reference element
export Element, gll_nodes_weights

# Layout and quadrature
export node_coordinates, fill_at_nodes!, quadrature_norm

# Evolution system
export WaveProblem, wave_rhs!, truncate_state!, truncate_stage!, initial_state,
       wave_energy, oldg_timestep, convergence_rate

# Time integration
export state_partition, wave_integrator

# Initial condition: standing sine mode
export wave_omega, wave_exact, wave_forest, wave_errors, reflect_omega,
       reflect_exact, reflect_errors

# Initial condition: radial blast wave
export blast_initial, blast_reference, blast_radial_table, blast_exact,
       uniform_blast

# The analytic particle
export ParticleModel, Trajectory, trajectory_at, self_inertia, particle_density,
       boosted_static, near_field, particle_field, particle_rate, source_bracket,
       Window, window_terms, ParticleSource, particle_source

# The transfer between particles and grid
export TransferTables, gauss_nodes_weights, sinh_rule, ParticleSupport, project_source!,
       projection_scratch, support_elements, pairing, reconstruct, reconstruct_gradient,
       ball_rule

# The coupled problem
export InspiralProblem, Prescribed, uniform_motion, field_state, particle_state,
       inspiral_initial_state, inspiral_rhs!, inspiral_stage!, inspiral_partition,
       inspiral_integrator, particle_forces, ball_norm

include("precision.jl")
include("device.jl")
include("element.jl")
include("layout.jl")
include("evolution.jl")
include("stepping.jl")
include("sinewave.jl")
include("blast.jl")
include("particle_forms.jl")
include("particle.jl")
include("transfer.jl")
include("inspiral.jl")

end

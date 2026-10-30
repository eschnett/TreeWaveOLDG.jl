# The time integrator: IMEXRungeKutta's explicit `RK4`, its stage
# arithmetic run by block owner, with truncation in its two limiter hooks.
#
# Why IMEXRungeKutta and not OrdinaryDiffEq: OrdinaryDiffEq forms its stage
# vectors with a serial broadcast on one core, which is the Amdahl term
# TreeWave measured capping a threaded step at 3.6×, and allocates and
# first-touches its cache serially, on one NUMA domain. With a step limiter
# its RK4 also re-evaluates the right-hand side after the limiter, five
# evaluations and five truncations per step instead of four. IMEXRungeKutta
# forms each stage combination as one pass over the state, split by the
# partition below so that every block's entries are combined on the thread
# `map_blocks!` runs that block on, and first-touches its scratch through
# the same partition. The result is bitwise the same on every path and at
# every thread count. See "Threads and devices" in `CODE.md`.

"""
    state_partition(fs::FieldSet, u) -> Vector{UnitRange{Int}} or nothing

The block-ownership partition of the state vector `u` of `fs`, as
IMEXRungeKutta's `partition` keyword takes it: element `c` holds the
entries of the blocks in chunk `c` of TreeAMR's `threadchunks(nblocks)` —
the chunk `launch_by_owner!` runs on default-pool thread `c` — one element
per thread, padded with empty ranges where there are fewer blocks than
threads. A block's `u` and `v` are contiguous in the state vector
(`statearray`'s last index is the block), so each thread owns one range.

`nothing` for a state that is not a CPU `Array`, which is
IMEXRungeKutta's broadcast path: a device array is combined by one fused
broadcast on the device, and a partition would be refused there.

TreeHydro carries the same helper.
"""
function state_partition(fs::FieldSet{T,D}, u) where {T,D}
    u isa Array || return nothing
    L = fs.forest.N^D * fs.nvars
    length(u) == L * nblocks(fs) || throw(DimensionMismatch(
        "the state vector has $(length(u)) entries but the field set's blocks " *
        "hold $(L * nblocks(fs)): a partition of one mesh's state cannot be " *
        "applied to another's."))
    return owner_ranges(fs)
end

# Thread `c`'s range of the state vector of `fs`, one per thread, padded
# with empty ranges.
function owner_ranges(fs::FieldSet{T,D}) where {T,D}
    L = fs.forest.N^D * fs.nvars
    parts = UnitRange{Int}[((first(r) - 1) * L + 1):(last(r) * L)
                           for r in TreeAMR.threadchunks(nblocks(fs))]
    while length(parts) < Threads.nthreads()
        push!(parts, 1:0)
    end
    return parts
end

"""
    wave_integrator(p::WaveProblem, u0, t0, t1; dt, alg=RK4(), alias_u0=false,
                    reuse=nothing)

An IMEXRungeKutta integrator for [`wave_rhs!`](@ref) from `t0` to `t1` in
`⌈(t1 - t0)/dt⌉` equal steps of explicit `alg`, with
[`truncate_stage!`](@ref) installed as both its stage and its step
limiter, so that a run cannot leave truncation out, and its stage
arithmetic split by [`state_partition`](@ref). Advance it with
`step!(integ)` or `solve!(integ)`; the state is `integ.u` and the time
`integ.t`.

`alias_u0 = true` steps `u0` itself. `reuse = integ′`, an earlier
integrator of the same problem, takes over its scratch arrays instead of
allocating and first-touching new ones (IMEXRungeKutta's "Scratch
reuse").
"""
function wave_integrator(p::WaveProblem, u0, t0, t1; dt, alg=RK4(),
                         alias_u0::Bool=false, reuse=nothing)
    prob = IMEXProblem(wave_rhs!, nothing, u0, (t0, t1), p)
    return init(prob, alg; dt=dt, stage_limiter=truncate_stage!,
                step_limiter=truncate_stage!, partition=state_partition(p.fs, u0),
                alias_u0=alias_u0, reuse=reuse)
end

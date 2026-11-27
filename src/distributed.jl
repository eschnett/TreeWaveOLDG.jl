# Running over MPI (added 2026-11-16, on TreeAMR's M7).
#
# TreeAMR distributes a forest's blocks over the ranks of a communicator
# and keeps the forest itself, every leaf, on every rank. The ghost
# exchange, the regrid and its own reductions (`mesh_fold`,
# `mesh_mapreduce`) it makes collective and independent of the rank count.
# What it cannot do is this package's half, which this file is for, as
# TreeHydro's `distributed.jl` is for that package:
#
# - a number combined on the host from per-block values is a rank's own:
#   the time step's speed over the blocks, the radiative sphere's boundary
#   energy, a refusal found in one rank's blocks;
# - a global leaf index is not a local block index: `locate_point` answers
#   with the forest's, and a block across a face, or under a particle's
#   support, may be another rank's.
#
# The verbs are TreeAMR's internal `commrank`, `commsize`, `allgather`,
# `allgatherv`, `isend`, `irecv` and `waitall`, reached through
# `forest.comm`; this package never names MPI. A serial run takes every
# function here through its one-rank branch, which gathers nothing and
# returns its argument, so every serial number stays where it was. See
# "Running distributed" in `CODE.md`.

rank_comm(comm::TreeAMR.Communicator) = comm
rank_comm(forest::Forest) = forest.comm
rank_comm(fs::FieldSet) = fs.forest.comm

"""
    rank_reduce(op, over, value; present = true)

`value` combined over every rank of `over` (a forest, a field set or a
`TreeAMR.Communicator`) with `op`, in rank order, the same bits on every
rank: collective. `present = false` leaves this rank's `value` out, for a
rank that holds nothing to contribute. On one rank it is `value` itself,
and nothing is gathered. `value` must be `isbits`.
"""
function rank_reduce(op, over, value; present::Bool=true)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return value
    parts = TreeAMR.allgather(comm, (present, value))
    acc, found = value, false
    for (has, v) in parts
        has || continue
        acc = found ? op(acc, v) : v
        found = true
    end
    return acc
end

"""
    rank_agree(over, v::AbstractVector)

Rank 0's `v` on every rank of `over`: collective. For a number every
rank computes from the same inputs, but through a library that may pick
its kernels by the CPU it runs on, as LAPACK's solves do; the ranks of a
job on several nodes may get different CPUs, and a rank whose number
differed in its last bit could take another branch and call collectives
the others do not (`solve_jets`' Newton steps). Serially `v` itself.
"""
function rank_agree(over, v::AbstractVector)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return v
    return TreeAMR.bcast(comm, v, 0)
end

"""
    agree_refusal(f, over)

`f()`, run on every rank of `over`; if it throws an `ArgumentError` on
any rank, every rank throws: the rank that refused its own error, the
others one naming the ranks that refused. For a check that reads one
rank's blocks, whose refusal on one rank alone would leave the others
waiting in their next collective. Anything else is rethrown at once.
Serially this is `f()`. Returns what `f` returns.
"""
function agree_refusal(f, over)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return f()
    err, result = nothing, nothing
    try
        result = f()
    catch e
        e isa ArgumentError || rethrow()
        err = e
    end
    refused = TreeAMR.allgather(comm, err !== nothing)
    err === nothing || throw(err)
    any(refused) && throw(ArgumentError(
        "rank(s) $(join(findall(refused) .- 1, ", ")) refused, and this rank did " *
        "not: the check reads each rank's own blocks, and the run stops on every " *
        "rank rather than leave the others waiting. Its error is printed by that " *
        "rank."))
    return result
end

# The local block of global leaf `g`, or `nothing` where another rank owns
# it. Serially `g` itself.
function local_block(forest::Forest, g::Integer)
    r = blockrange(forest)
    return g in r ? Int(g) - first(r) + 1 : nothing
end

# The global leaf of local block `b`.
global_leaf(forest::Forest, b::Integer) = first(blockrange(forest)) + Int(b) - 1

"""
    gather_blocks(fs, ua, leaves, var) -> Dict{Int,Array}

Variable `var` of every block of the global leaf indices `leaves`, as
host arrays keyed by leaf, on every rank: each rank copies the blocks it
owns, and one `allgatherv` concatenates them, in leaf order, since every
rank owns a contiguous run of leaves in rank order. Collective, with the
same `leaves` on every rank. Serially, and on one rank, the blocks are
copied from `ua` and nothing is gathered.
"""
function gather_blocks(fs::FieldSet{T,D}, ua, leaves, var) where {T,D}
    forest = fs.forest
    sorted = sort!(unique(Int.(collect(leaves))))
    N = forest.N
    n = N^D
    copy_block(b) = Array(view(ua, ntuple(_ -> :, D)..., var, b))
    if !TreeAMR.isdistributed(forest)
        return Dict(g => copy_block(g) for g in sorted)
    end
    own = [(g, b) for g in sorted for b in (local_block(forest, g),) if b !== nothing]
    buf = Vector{T}(undef, n * length(own))
    for (i, (_, b)) in enumerate(own)
        copyto!(buf, (i - 1) * n + 1, vec(copy_block(b)), 1, n)
    end
    all = TreeAMR.allgatherv(forest.comm, buf)
    length(all) == n * length(sorted) || throw(DimensionMismatch(
        "gathered $(length(all)) values for $(length(sorted)) blocks of $n: the " *
        "ranks asked for different blocks"))
    return Dict(g => reshape(all[((i - 1) * n + 1):(i * n)], ntuple(_ -> N, D))
                for (i, g) in enumerate(sorted))
end

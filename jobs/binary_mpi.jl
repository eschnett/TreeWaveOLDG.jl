# INSPIRAL-PLAN.md's M6b, the short binary run of `jobs/binary.jl` at
# `d/8`, over MPI (`bin/binary.jl --mpi`): one rank per node, each on the
# node's 64 cores, pinned, as `jobs/binary.jl`'s runs are on one node.
# On one node the run would take about 28 hours, past `amdq`'s 24 ("The
# binary" under "Measured results" in `CODE.md`); the field's right-hand
# side splits over the ranks, and every rank solves the jets.
#
#     simlaunch submit <study> jobs/binary_mpi.jl --site symmetry --run binary-h8
#
# `scan-h8-r<N>` takes the same run for 32 steps at `N` ranks, for the
# cost of a step against the rank count.
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --every=1 --mpi"
pinned = (julia_args = ["--threads=64,0"], vars = (JULIA_EXCLUSIVE = "1",), threads = 64)

scan = [(label = "scan-h8-r$n", cmd = "$bi --hdiv=8 --tend=0.125", ranks = n, nodes = n,
         pinned...) for n in (1, 2, 4)]

(name = "binary_mpi",
 description = "TreeWaveOLDG M6b: the binary at d/8 over MPI, two orbits",
 site = (symmetry = (time = "24:00:00", resources = (nodes = 4,)),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 40 \"$d/output.log\"; done",),
 run = [(label = "binary-h8", cmd = "$bi --hdiv=8 --tend=32 --flux=flux-binary-h8.tsv",
         ranks = 4, nodes = 4, pinned...),
        scan...])

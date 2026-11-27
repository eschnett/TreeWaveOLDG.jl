# INSPIRAL-PLAN.md's M6b, the short binary run of `jobs/binary.jl` at
# `d/8`, over MPI (`bin/binary.jl --mpi`): four ranks, one per node, each
# on its node's 64 cores, pinned, as `jobs/binary.jl`'s runs are on one
# node. On one node a step takes about 9 seconds early on and the run
# would take about a day, past `amdq`'s 24 hours ("The binary" under
# "Measured results" in `CODE.md`); the field's right-hand side and `LΠ`
# split over the ranks, and every rank solves the jets. The run writes a
# checkpoint every two hours and before the time limit, and a later job
# resumes it (`--checkpoint`):
#
#     simlaunch submit <study> jobs/binary_mpi.jl --site symmetry
#     simlaunch resubmit <study> jobs/binary_mpi.jl --site symmetry   # if stopped
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --every=1 --mpi"

(name = "binary_mpi",
 description = "TreeWaveOLDG M6b: the binary at d/8 over MPI on four nodes, two orbits",
 checkpoints = true,
 restart = (to = "--checkpoint", from = "--restart-from", walltime = "--walltime",
            margin = "0:20:00", continue_code = 3),
 site = (symmetry = (time = "24:00:00", resources = (nodes = 4,), queue = "amdq"),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 40 \"$d/output.log\"; done",),
 run = [(label = "binary-h8", cmd = "$bi --hdiv=8 --tend=32 --flux=flux-binary-h8.tsv",
         ranks = 4, nodes = 4, threads = 64, julia_args = ["--threads=64,0"],
         vars = (JULIA_EXCLUSIVE = "1",))])

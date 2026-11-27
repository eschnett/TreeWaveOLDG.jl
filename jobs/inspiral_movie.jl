# The movie of a binary that inspirals under its forces
# (`bin/inspiral_movie.jl`, "The movie" in `CODE.md`): `bin/binary.jl`'s
# binary at `q² = 0.2`, `ℓ = 0.04`, in its ball (three levels at `w0 = 4`,
# `Lc = 16`, shells of radial width 1 to `R1 = 50`), at `d/4`, to
# `t = 72`, over which the separation falls from 1 to about 0.7. Its
# frames go to `frames-q2-h4/`, the retarded field of the recorded path's
# to `ref-q2-h4/`, the modes on spheres to `waves-q2-h4.tsv`. One job per
# run:
#
#     simlaunch submit <study> jobs/inspiral_movie.jl --site symmetry --run smoke
#
# - `smoke`: the smallest ball whose finest cube holds the window, a few
#   steps with frames, before the movie.
# - `movie-q2-h4`: the movie, about 9 200 steps at 4.1 s on 64 threads,
#   11 hours. The source takes the predictor's `ä` (the driver's default):
#   with the solved one the run broke down at `t = 17` ("The breathing
#   instability" in `CODE.md`).
# - `cuda-smoke`, in `inspiral_movie_gpu.toml`: whether `Dynamic` runs on
#   an H200.
im = "bin/inspiral_movie.jl --q2=0.2 --ell=0.04"

(name = "inspiral_movie",
 description = "TreeWaveOLDG: the movie of a binary that inspirals under its forces",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 40 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "$im --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.25 --pevery=0.0625 --every=0.0625 --grids=7:256,3:256 --plane=frames-smoke --ref=ref-smoke --waves=waves-smoke.tsv --wradii=5,7"),
        (label = "movie-q2-h4",
         cmd = "$im --hdiv=4 --levels=3 --w0=4 --R1=50 --hshell=1 --tend=72 --pevery=0.25 --plane=frames-q2-h4 --ref=ref-q2-h4 --waves=waves-q2-h4.tsv --wradii=10,20,30,40")])

# M8's scans on one GPU, each its own job, one run named per submit:
#
#     simlaunch submit <study> jobs/gpu_scans.jl --site symmetry --run far-retarded
#
# What comes back into the window before the sphere's return is what the
# cube's coarse-fine faces and the cube–shell faces reflect ("Particles in
# the ball" under "Measured results" in `CODE.md`); `gpu_pulses.jl` has
# the pulses. Measured (2026-11-13): the tests take 5 minutes, the circle
# 3.1 to 5.3 hours and the particle at rest 4.5.
#
# The old script's 128 GB of host memory, above the 64 GB a GPU gets by
# the site's profile (`--mem` twice; Slurm takes the last).
#
# - `device_tests`: `test/device_tests.jl` on CUDA, with the particles in
#   the ball.
# - `circle-h1`: `bin/prescribed.jl`'s circle at `v = 0.2`, `d/8`, order 3,
#   from the retarded field, in the plan's ball (three levels at `w0 = 4`,
#   `Lc = 16`) with shells of radial width 1, twice the cube's root
#   element width, to `R1 = 60`, to `t = 40`. The retarded data reach
#   `rx1 = 48`, blended into the particle field by 54; what the particle
#   field beyond the blend sends in arrives from `t ≈ 25`, so the run is
#   clean to about `t = 24`.
# - `rest-R72-h8`: `bin/uniform_motion.jl`'s particle at rest at `d/8` in
#   `ball_motion.jl`'s ball at `R1 = 72`, to `t = 80`, past the sphere's
#   under-resolved return at about `(2R1 − R₂)/2 = 70.75`.
# - `far-*`: M8d's circle from data that are the field far out
#   (`--data=retarded`) or uniform motion's (`--data=uniform`), at `d/8`
#   or `d/4`, to `R1 = 100` and `t = 90`, with the waves ("Waveforms"
#   under "Measured results" in `CODE.md`).
# - `past-n4-h6`, `depart-n20-n4-h6`: M8f's circle with an assumed past,
#   spun up with `n = 4` over 10 from `−30`, and its departure with
#   `n₂ = 0` ("Assumed pasts").
# - `movie-h4`: the movie of two charges on a circle at `hₑ = 1/4`, to
#   `t = 90`, its frames in `frames-h4/` ("The movie").
pr = "bin/prescribed.jl --backend=cuda --path=circle"
far = "--v=0.2 --levels=3 --w0=4 --order=3 --ball --R1=100 --rx1=200 --rx2=210 --tend=90 --every=0.5 --radii=4,8,16,32,64 --cost"
past = "--past=spinup --T=30 --ramp=10 --n=4"

(name = "gpu_scans",
 description = "TreeWaveOLDG M8: what the ball sends back, the far data, the past and the movie, on one H200",
 env = "test+cuda",
 site = (symmetry = (time = "24:00:00", resources = (gpus = 1,), queue = "h200q",
                         sbatch_args = ["--mem=128G"]),),
 run = [(label = "device_tests", cmd = "test/device_tests.jl",
         vars = (TREEWAVEOLDG_TEST_BACKEND = "cuda",)),
        (label = "circle-h1",
         cmd = "$pr --data=retarded --v=0.2 --hdiv=8 --levels=3 --w0=4 --order=3 --ball --R1=60 --hshell=1 --rx1=48 --rx2=54 --tend=40 --every=0.5 --radii=4,8 --cost"),
        (label = "rest-R72-h8",
         cmd = "bin/uniform_motion.jl --backend=cuda --ball --R1=72 --levels=3 --w0=3 --hdiv=8 --v=0 --tend=80 --every=0.25 --cost"),
        (label = "far-retarded",
         cmd = "$pr --data=retarded --hdiv=8 $far --waves=waves-retarded.tsv"),
        (label = "far-uniform",
         cmd = "$pr --data=uniform --hdiv=8 $far --waves=waves-uniform.tsv"),
        (label = "far-uniform-h4",
         cmd = "$pr --data=uniform --hdiv=4 $far --waves=waves-uniform-h4.tsv"),
        (label = "past-n4-h6",
         cmd = "$pr --data=retarded --hdiv=6 $far $past --waves=waves-past-n4-h6.tsv"),
        (label = "depart-n20-n4-h6",
         cmd = "$pr --data=retarded --hdiv=6 $far $past --depart=1e-3 --n2=0 --waves=waves-depart-n20-n4-h6.tsv"),
        (label = "movie-h4",
         cmd = "bin/orbit_movie.jl --backend=cuda --hdiv=4 --hshell=1 --tend=90 --pevery=0.25 --plane=frames-h4 --waves=waves-movie-h4.tsv")])

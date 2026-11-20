# The reflection scan of M8c on one GPU ("Particles in the ball" under
# "Measured results" in `CODE.md`): `bin/nested_ball.jl`'s outgoing pulse
# from the origin, in `Float64`, through three nested cubes (`w0 = 4`,
# `hₑ = 1/4`, so the cube's root element is 1 wide at `Lc = 16`) and
# shells to `R1 = 64`, for `λ = 4, 8, 16` (4, 8 and 16 coarse elements per
# wavelength at the face at 8) against shells of radial width 1, 2 and 4
# (1:1, 2:1 and 4:1 at the cube–shell face), and graded from 1 to 4, to
# `t = 60`, before the sphere's under-resolved return at 64; the errors in
# the balls of radius 4, 8 and 16 every half time unit. The twelve took
# 2.4 hours (2026-11-13). The old script's 128 GB of host memory, above
# the 64 GB a GPU gets by the site's profile (`--mem` twice; Slurm takes
# the last).
shells = [("h1", "--hshell=1"), ("h2", "--hshell=2"), ("h4", "--hshell=4"),
          ("h1to4", "--hshell=4 --hface=1")]
runs = [(label = "pulse-l$lambda-$s",
         cmd = "bin/nested_ball.jl --backend=cuda --type=f64 --levels=3 --w0=4 --hdiv=4 --lambda=$lambda --R1=64 $shell --tend=60 --every=0.5 --radii=4,8,16")
        for (s, shell) in shells, lambda in (4, 8, 16)]

(name = "gpu_pulses",
 description = "TreeWaveOLDG M8c: what the ball's faces reflect, outgoing pulses on one H200",
 env = "test+cuda",
 site = (symmetry = (time = "24:00:00", resources = (gpus = 1,), queue = "h200q",
                         sbatch_args = ["--mem=128G"]),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = vec(runs))

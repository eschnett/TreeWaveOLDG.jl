# INSPIRAL-PLAN.md's M8c: `bin/uniform_motion.jl --ball` in the
# seven-patch ball's cube, beside the same runs on nested boxes. The cube
# holds M3b's three finest levels about M3's box (the finest cube of
# half-width `w0 = 3`, so the cube's half-width is 12), with shells of
# radial elements twice the cube's root width to a Sommerfeld sphere at
# `R1` (`bin/ball.jl`). First M3b's convergence series to `t = 0.5` at
# `v = 0.2`, in the ball at `R1 = 36` and on the periodic box of the
# cube's three levels, whose numbers the ball has to give inside the box's
# clean time; then the particle at rest at `hₑ = d/8`, to `t = 60`, far
# past it, for `R1 = 36` and 50 and once on the box: the error in the
# `R₂` ball, the force, and the mean of `Π_r`, which the net charge
# drives at `Q/(γV)` in the periodic box and which should stay flat in
# the ball, until what the sphere sends back arrives, from about
# `2R1 − R₂` ("Particles in the ball" under "Measured results" in
# `CODE.md`).
um = "bin/uniform_motion.jl"
series = [[(label = "series-R36-h$h",
            cmd = "$um --ball --R1=36 --levels=3 --w0=3 --hdiv=$h --v=0.2 --tend=0.5 --every=0.125 --cost"),
           (label = "series-box-h$h",
            cmd = "$um --levels=3 --w0=3 --hdiv=$h --v=0.2 --tend=0.5 --every=0.125 --cost")]
          for h in (4, 8, 12)]
rest = [[(label = "rest-R$R-h8",
          cmd = "$um --ball --R1=$R --levels=3 --w0=3 --hdiv=8 --v=0 --tend=60 --every=0.0625 --cost")
         for R in (36, 50)];
        (label = "rest-box-h8",
         cmd = "$um --levels=3 --w0=3 --hdiv=8 --v=0 --tend=60 --every=0.0625 --cost")]

(name = "ball_motion",
 description = "TreeWaveOLDG M8c: uniform motion in the ball, beside nested boxes",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = vcat(reduce(vcat, series), rest))

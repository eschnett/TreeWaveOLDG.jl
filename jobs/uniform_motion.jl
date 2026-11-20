# INSPIRAL-PLAN.md's M3: `bin/uniform_motion.jl` as a convergence series,
# `hₑ = d/4 … d/12` at `v = 0.2` and `v = 0` to `t = 0.5`, with the cost
# of each resolution; `bin/projection.jl` on the first run's source; and
# the clean time, the same particle at `hₑ = d/8` and `d/12` to `t = 5`,
# past the time the error of the box's faces reaches the centre
# ("Uniform motion" under "Measured results" in `CODE.md`).
series = [(label = "series-v$v-h$h",
           cmd = "bin/uniform_motion.jl --hdiv=$h --v=$v --tend=0.5 --every=0.125 --cost")
          for h in (4, 6, 8, 10, 12), v in ("0.2", "0.0")]
projection = [(label = "projection-h8", cmd = "bin/projection.jl --hdiv=8 --reps=3"),
              (label = "projection-h8-offgrid",
               cmd = "bin/projection.jl --hdiv=8 --reps=3 --offgrid")]
clean = [(label = "clean-h$h",
          cmd = "bin/uniform_motion.jl --hdiv=$h --v=0.2 --tend=5 --every=0.0625")
         for h in (8, 12)]

(name = "uniform_motion",
 description = "TreeWaveOLDG M3: a particle in uniform motion, its series and clean time",
 pin = true,
 site = (symmetry = (time = "1:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = vcat(vec(series), projection, clean))

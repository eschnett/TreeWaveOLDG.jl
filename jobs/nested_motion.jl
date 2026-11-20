# INSPIRAL-PLAN.md's M3b: `bin/uniform_motion.jl` on nested boxes about
# M3's box (the finest cube of half-width `w0 = 3`, M3's side 6). First
# M3's convergence series to `t = 0.5` at `v = 0.2` on four levels, with
# the cost of each resolution, to set against M3's uniform box; then the
# clean time, the same particle at rest at `hₑ = d/8` to `t = 20` on 4
# and 5 levels, sampled every `1/16`, with the cost of each layout, for
# when anything arrives from which face (the first coarse-fine face at 3,
# the further ones at 6, 12, …, the outer faces at `2^(levels-1)·3`) and
# the net charge's drift as the box grows. At rest, because the window is
# fixed about the centre ("Nested boxes" under "Measured results" in
# `CODE.md`).
series = [(label = "series-n4-h$h",
           cmd = "bin/uniform_motion.jl --levels=4 --w0=3 --hdiv=$h --v=0.2 --tend=0.5 --every=0.125 --cost")
          for h in (4, 8, 12)]
clean = [(label = "clean-n$n-h8",
          cmd = "bin/uniform_motion.jl --levels=$n --w0=3 --hdiv=8 --v=0 --tend=20 --every=0.0625 --cost")
         for n in (4, 5)]

(name = "nested_motion",
 description = "TreeWaveOLDG M3b: uniform motion on nested boxes",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = vcat(series, clean))

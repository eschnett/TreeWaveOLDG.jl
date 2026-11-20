# INSPIRAL-PLAN.md's M4: `bin/prescribed.jl` on the layout chosen for M4
# to M7 (five nested cubes, the finest of half-width `w0 = 4`,
# `2 w0 hdiv` elements across each level): hyperbolic motion of proper
# acceleration 0.1 until its speed reaches 0.05, with the error against
# the exact retarded field every `1/32`, and the slow circle, `v = 0.05`
# at radius 0.5, from the retarded field, to `t = 8`, with the error
# against it every `1/4`; each at `d/4`, at orders 3 and 2, with the
# force every step and its cost ("Prescribed paths" under "Measured
# results" in `CODE.md`). A finer series takes a spec of its own with a
# longer time, as `dynamic_series.jl` is to `dynamic.jl`.
pr = "bin/prescribed.jl"
hyperbolic(h, o) = (label = "hyperbolic-h$h-o$o",
                    cmd = "$pr --hdiv=$h --order=$o --path=hyperbolic --g=0.1 --every=0.03125 --fevery=1 --cost")
circle(h, o) = (label = "circle-ret-h$h-o$o-t8",
                cmd = "$pr --hdiv=$h --order=$o --path=circle --data=retarded --rc=0.5 --tend=8 --every=0.25 --fevery=1 --cost")

(name = "prescribed",
 description = "TreeWaveOLDG M4: a particle on a prescribed path",
 pin = true,
 site = (symmetry = (time = "1:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = [hyperbolic(4, 3), hyperbolic(4, 2), circle(4, 3), circle(4, 2)])

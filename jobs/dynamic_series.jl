# INSPIRAL-PLAN.md's M6a, the rest of its series beside `dynamic.jl`'s
# `d/4`: the spring and the circle at `d/6`, `d/8` and `d/12` at
# `v = 0.05`, and at `d/8` at `v = 0.2`; and the abrupt hand-over
# (`onset = 0`) at `d/8` ("The equations of motion" under "Measured
# results" in `CODE.md`). One job per run, side by side:
#
#     simlaunch submit <study> jobs/dynamic_series.jl --site symmetry --run spring-h12-v0.05-on2
#
# The spring at `d/12` took more than the eight hours it was given then
# (about 0.87 time units an hour on 64 threads, job 606327).
dy = "bin/dynamic.jl"
spring(h, v, onset) = (label = "spring-h$h-v$v-on$onset",
                       cmd = "$dy --hdiv=$h --v=$v --tend=8 --onset=$onset")
circle(h, v) = (label = "circle-h$h-v$v", cmd = "$dy --hdiv=$h --v=$v --tend=8 --prescribed")

(name = "dynamic_series",
 description = "TreeWaveOLDG M6a: the spring-held particle against the circle, finer",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = [spring(6, "0.05", 2), circle(6, "0.05"),
        spring(8, "0.05", 2), circle(8, "0.05"), spring(8, "0.05", 0),
        spring(12, "0.05", 2), circle(12, "0.05"),
        spring(8, "0.2", 2), circle(8, "0.2")])

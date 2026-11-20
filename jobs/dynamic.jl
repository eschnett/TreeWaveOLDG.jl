# INSPIRAL-PLAN.md's M6a: `bin/dynamic.jl` on M4's layout (five nested
# cubes, the finest of half-width `w0 = 4`) at `d/4`: the particle held on
# M4's circle by a spring and moving under its forces, the force taking
# over over `onset = 2`, and the prescribed circle on the same grid and
# steps (`--prescribed`), both at `v = 0.05` from the circle's retarded
# field, to `t = 8`, with the force every step ("The equations of motion"
# under "Measured results" in `CODE.md`). `dynamic_series.jl` has the
# finer runs.
dy = "bin/dynamic.jl"

(name = "dynamic",
 description = "TreeWaveOLDG M6a: the spring-held particle against the circle, at d/4",
 pin = true,
 site = (symmetry = (time = "1:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = [(label = "spring-h4-v0.05-on2", cmd = "$dy --hdiv=4 --v=0.05 --tend=8 --onset=2"),
        (label = "circle-h4-v0.05", cmd = "$dy --hdiv=4 --v=0.05 --tend=8 --prescribed")])

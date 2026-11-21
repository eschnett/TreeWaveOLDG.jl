# INSPIRAL-PLAN.md's M6b, the short binary run (`bin/binary.jl`): the
# Newtonian binary of INSPIRAL.md's first run, `v = 0.2` at separation 1,
# from the retarded field of a past spun up from rest with `n = 4` over 10
# to the orbit at `−30`, the force taking over over one orbit with
# `n₂ = 4`, in the plan's ball (three levels at `w0 = 4`, `Lc = 16`,
# shells of radial width 1 to `R1 = 50`), to `t = 32`, two orbits. The
# runs watch the jets for an instability of the higher derivatives: one
# that grows faster on the finer grid (`d/6` against `d/4`) or with the
# smaller step (`cfl = 1/4` against `1/2`) is numerical, a drift that does
# not is physical ("The equations of motion" under "Measured results" in
# `CODE.md`). `circle-h4` runs the past's circle on the same grid and
# steps. `dmhgrid-h4` runs `binary-h4` to `t = 20`, past the hand-over,
# with the grid's own part of `δm_h` (`dmhgrid<k>`), which `binary-h4`
# did not record. One job per run, side by side:
#
#     simlaunch submit <study> jobs/binary.jl --site symmetry --run binary-h4
#
# `binary_smoke.toml` runs the driver for a few steps first.
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --tend=32 --every=1"

(name = "binary",
 description = "TreeWaveOLDG M6b: the Newtonian binary under its forces, two orbits",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 40 \"$d/output.log\"; done",),
 run = [(label = "binary-h4", cmd = "$bi --hdiv=4 --flux=flux-binary-h4.tsv"),
        (label = "binary-h4-dt2", cmd = "$bi --hdiv=4 --cfl=0.25"),
        (label = "binary-h6", cmd = "$bi --hdiv=6 --flux=flux-binary-h6.tsv"),
        (label = "circle-h4", cmd = "$bi --hdiv=4 --prescribed --flux=flux-circle-h4.tsv"),
        (label = "dmhgrid-h4",
         cmd = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --tend=20 --every=1 --hdiv=4 --jevery=0.25")])

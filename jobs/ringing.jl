# What rings the grid's field near a moving particle (`CODE.md`, "The
# binary" under "Measured results"): the past's circle of `binary.jl`,
# prescribed, at `d/4` and lower speeds, beside `binary.jl`'s `circle-h4`
# at `v = 0.2`, one AMD node each. The remainder's force rings at
# `ω hₑ ≈ 8` to 10 at `v = 0.2`. If that is the grid's own modes, the band
# stays where it is at any speed; if the moving source's projection beats
# against the elements, it moves as `v/hₑ`.
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/ringing.jl
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-v010
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-v005
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --every=1 --hdiv=4 --prescribed --tend=16"

(name = "ringing",
 description = "TreeWaveOLDG M6b: the field's ringing at the prescribed circle's speeds",
 pin = true,
 site = (symmetry = (time = "03:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "circle-v010", cmd = "$bi --v=0.1"),
        (label = "circle-v005", cmd = "$bi --v=0.05")])

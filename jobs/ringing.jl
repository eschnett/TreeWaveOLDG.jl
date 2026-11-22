# What rings the grid's field near a moving particle (`CODE.md`, "The
# binary" under "Measured results"): the past's circle of `binary.jl`,
# prescribed, at `d/4` and lower speeds, beside `binary.jl`'s `circle-h4`
# at `v = 0.2`, one AMD node each. The remainder's force rings at
# `ω hₑ ≈ 8` to 10 at `v = 0.2`. If that is the grid's own modes, the band
# stays where it is at any speed; if the moving source's projection beats
# against the elements, it moves as `v/hₑ`.
#
# `circle-w8` is `circle-h4` with the finest cube twice as wide, two
# levels at `w0 = 8` in the same cube (`Lc = 16`), its coarse-fine faces 8
# from the orbit's centre instead of 4. If the ringing is the initial
# data's error coming in from the coarser levels, it reaches the particle
# at `t ≈ 7.5` instead of 3.5. (It does not: `CODE.md`.)
#
# `circle-w8-win` is `circle-w8` with the window twice as wide,
# `R₁ = 2.5`, `R₂ = 5`, to `t = 10`. If the ringing starts where the
# source lives, between `R₁` and `R₂`, it arrives later.
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/ringing.jl
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-v010
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-v005
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-w8
#     simlaunch submit twoldg-<commit> jobs/ringing.jl --site symmetry --run circle-w8-win
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --every=1 --hdiv=4 --prescribed --tend=16"

(name = "ringing",
 description = "TreeWaveOLDG M6b: the field's ringing at the prescribed circle's speeds",
 pin = true,
 site = (symmetry = (time = "08:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "circle-v010", cmd = "$bi --v=0.1"),
        (label = "circle-v005", cmd = "$bi --v=0.05"),
        (label = "circle-w8",
         cmd = "bin/binary.jl --levels=2 --w0=8 --R1=50 --hshell=1 --every=1 --hdiv=4 --prescribed --tend=16"),
        (label = "circle-w8-win",
         cmd = "bin/binary.jl --levels=2 --w0=8 --R1=50 --hshell=1 --every=1 --hdiv=4 --prescribed --tend=10 --window=2.5,5")])

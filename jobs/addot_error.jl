# What the source with the predictor's `ä` costs (`CODE.md`, "The
# breathing instability" under "Measured results"): `bin/binary.jl`'s
# binary with `--source-addot=solved` and `predicted` side by side, at
# charges where the solved `ä` is stable and its loop's gain small
# (`q² = 0.005`, gain 0.26, and `0.01`, 0.51), at `d/2` and `d/4`. The
# differences in the orbit and the forces between the two are the
# predicted `ä`'s error there, and the same at both resolutions if they
# are that error and not the grid's. Each row also records `δä`, the
# solved `ä` against the predictor's, from which `bin/addot_error.jl`
# estimates the error without a reference, here to check it and at
# `q² = 0.2` (`q1-h<n>-pred`), where the solved `ä` is unstable.
#
# The layout is `jobs/instability.jl`'s: three levels at `w0 = 4`, shells
# to `R₁ = 50`, the orbit's plane halfway between element faces, `ℓ =
# q²/5`, the force taking over over `onset = 4`, to `t = 20`, about an
# orbit after the hand-over, a row every step. One AMD node per run:
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/addot_error.jl
#     simlaunch submit twoldg-<commit> jobs/addot_error.jl --site symmetry --run smoke
#     simlaunch submit twoldg-<commit> jobs/addot_error.jl --site symmetry --run q005-h2-solved
bb = "bin/binary.jl --R1=50 --hshell=1 --every=1 --jevery=0.125 --tend=20 --onset=4"
b3 = "$bb --levels=3 --w0=4"
h = Dict(2 => "--hdiv=2 --oz=0.25", 4 => "--hdiv=4 --oz=0.125")
q = Dict("q005" => "--q2=0.005 --ell=0.001", "q01" => "--q2=0.01 --ell=0.002",
         "q1" => "--q2=0.2 --ell=0.04")

(name = "addot_error",
 description = "TreeWaveOLDG: the error of the source with the predictor's ä",
 pin = true,
 site = (symmetry = (time = "12:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $(q["q1"]) --onset=0.0625 --order=2 --source-addot=predicted");
        [(label = "$qk-h$n-$s", cmd = "$b3 $(h[n]) $(q[qk]) --source-addot=$s")
         for qk in ("q005", "q01") for n in (2, 4) for s in ("solved", "predicted")];
        [(label = "q1-h$n-pred", cmd = "$b3 $(h[n]) $(q["q1"]) --source-addot=predicted")
         for n in (2, 4)]])

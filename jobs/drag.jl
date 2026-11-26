# The predictor's drag (`CODE.md`, "The predictor's drag" under "The
# breathing instability"): the predictor leaves out the radiation
# reaction, so its `ä` departs from the solved one by about `−0.035q²`
# toward `ȧ`, and its `a⃛` and `a⃜`, which the source takes, by `0.14q²`
# and `0.45q²` across themselves. `--predictor-drag=0.93` gives the
# predictor alone a `DipoleDrag` of 0.93 times the dipole's radiation
# reaction, the grid's `F_t`/dip. Whether that moves the wide window's
# loss rate to the standard one's, 2.5 % above it at `q² = 0.2` and
# 0.5 % at 0.05, and how much it moves the standard window's:
#
# - `win-q<k>-h2`: the window `2.5–5` with the predictor's `ä`, on two
#   levels at `w0 = 8`, `d/2`, at `q² = 0.05` and 0.2, beside
#   `jobs/inertia.jl`'s `win-q<k>-h2`;
# - `std-q1-h4`: the standard window with the predictor's `ä`, three
#   levels at `w0 = 4`, `d/4`, at 0.2, beside `jobs/inertia.jl`'s
#   `pred-q1-h4`;
# - `smoke`: `jobs/cures.jl`'s, a few steps at order 2, first.
#
# The rest is `jobs/inertia.jl`'s. One AMD node per run, about three
# hours at `d/2` and four at `d/4`:
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/drag.jl
#     simlaunch submit twoldg-<commit> jobs/drag.jl --site symmetry --run smoke
bb = "bin/binary.jl --R1=50 --hshell=1 --every=1 --jevery=0.125 --tend=20 --onset=4"
b3 = "$bb --levels=3 --w0=4"
b2 = "$bb --levels=2 --w0=8"
h = Dict(2 => "--hdiv=2 --oz=0.25", 4 => "--hdiv=4 --oz=0.125")
q = Dict("q2" => "--q2=0.05 --ell=0.01", "q1" => "--q2=0.2 --ell=0.04")
drag = "--source-addot=predicted --predictor-drag=0.93"

(name = "drag",
 description = "TreeWaveOLDG: the predictor with a dipole drag, the windows' loss rates",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $(q["q1"]) --onset=0.0625 --order=2 --predictor-drag=0.93");
        [(label = "win-$qk-h2", cmd = "$b2 $(h[2]) $(q[qk]) --window=2.5,5 $drag")
         for qk in ("q2", "q1")];
        (label = "std-q1-h4", cmd = "$b3 $(h[4]) $(q["q1"]) $drag")])

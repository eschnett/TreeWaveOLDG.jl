# The breathing instability of the dynamic binary at large charge
# (`CODE.md`, "The breathing instability" under "Measured results"):
# `bin/inspiral_movie.jl`'s binary at `q² = 0.2`, `ℓ = 0.04`, `d/4` went
# unstable (job 607865), an oscillation of the separation at `ω ≈ 7.7`
# growing `×3.7` per unit time as the force took over. `bin/binary.jl`'s
# binary with `--q2`, in `binary.jl`'s ball (three levels at `w0 = 4`,
# shells to `R1 = 50`), the orbit's plane halfway between element faces,
# the force taking over over `onset = 4` (`--onset`), so that the growth
# at full force shows from `t = 4` on, to `t = 20`; every step a row of
# the jets, the forces and the solve, with `δm_h/M` and its grid part
# `dmhgrid` every `1/8`. One AMD node per run:
#
# - `q<k>-h<n>`: `q² = 0.2`, `0.05`, `0.0125` (`k` = 1, 2, 3), with
#   `ℓ = q²/5`, so that `δm/M = 0.26` throughout, at `d/2` and `d/4`;
# - `q<k>-l04-h2`: the same charges at the movie's `ℓ = 0.04`;
# - `q1-h<n>-dt2`: `q² = 0.2` at half the step;
# - `q1-h2-w8` and `q1-h2-w8-win`: two levels at `w0 = 8`, so that the
#   window's ball stays on the finest level, with the window `1.25–2.5`
#   and twice as wide, `2.5–5`;
# - `q1-h2-o1`, `q1-h2-o2`: the near field carried to order 1 and 2
#   instead of 3;
# - `q1-h2-movie`: the movie's `onset = 2π/Ω = 15.7`, at `d/2`;
# - `q1-h2-pred`: the source written with the predictor's `ä` in place of
#   the solved one (`--source-addot=predicted`), which
#   `bin/breathing_loop.jl` predicts stable;
# - `lone-q1-h2`: one particle of the same charge and size on
#   `bin/dynamic.jl`'s spring circle of radius 0.5 at `v = 0.2`, the
#   binary's particle's, on three nested boxes at `w0 = 4` instead of the
#   ball, from the circle's retarded field.
#
# A run that goes unstable stops when the solve does; its rows to there
# are the result.
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/instability.jl
#     simlaunch submit twoldg-<commit> jobs/instability.jl --site symmetry --run smoke
#     simlaunch submit twoldg-<commit> jobs/instability.jl --site symmetry --run q1-h2
bb = "bin/binary.jl --R1=50 --hshell=1 --every=1 --jevery=0.125 --tend=20"
b3 = "$bb --levels=3 --w0=4"
b2 = "$bb --levels=2 --w0=8"
h2 = "--hdiv=2 --oz=0.25"
h4 = "--hdiv=4 --oz=0.125"
q1 = "--q2=0.2 --ell=0.04"
q2 = "--q2=0.05 --ell=0.01"
q3 = "--q2=0.0125 --ell=0.0025"
fast = "--onset=4"

(name = "instability",
 description = "TreeWaveOLDG: the breathing instability of the dynamic binary at large charge",
 pin = true,
 site = (symmetry = (time = "16:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $q1 --onset=0.0625 --order=2"),
        (label = "q1-h2", cmd = "$b3 $h2 $q1 $fast"),
        (label = "q2-h2", cmd = "$b3 $h2 $q2 $fast"),
        (label = "q3-h2", cmd = "$b3 $h2 $q3 $fast"),
        (label = "q1-h4", cmd = "$b3 $h4 $q1 $fast"),
        (label = "q2-h4", cmd = "$b3 $h4 $q2 $fast"),
        (label = "q3-h4", cmd = "$b3 $h4 $q3 $fast"),
        (label = "q2-l04-h2", cmd = "$b3 $h2 --q2=0.05 --ell=0.04 $fast"),
        (label = "q3-l04-h2", cmd = "$b3 $h2 --q2=0.0125 --ell=0.04 $fast"),
        (label = "q1-h2-dt2", cmd = "$b3 $h2 $q1 $fast --cfl=0.25"),
        (label = "q1-h4-dt2", cmd = "$b3 $h4 $q1 $fast --cfl=0.25"),
        (label = "q1-h2-w8", cmd = "$b2 $h2 $q1 $fast"),
        (label = "q1-h2-w8-win", cmd = "$b2 $h2 $q1 $fast --window=2.5,5"),
        (label = "q1-h2-o1", cmd = "$b3 $h2 $q1 $fast --order=1"),
        (label = "q1-h2-o2", cmd = "$b3 $h2 $q1 $fast --order=2"),
        (label = "q1-h2-movie", cmd = "$b3 $h2 $q1"),
        (label = "q1-h2-pred", cmd = "$b3 $h2 $q1 $fast --source-addot=predicted"),
        (label = "lone-q1-h2",
         cmd = "bin/dynamic.jl --hdiv=2 --levels=3 --w0=4 --v=0.2 --rc=0.5 --rx1=10 --rx2=14 --onset=4 --tend=20 --every=1 $q1 --oz=0.25")])

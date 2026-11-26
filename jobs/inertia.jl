# The predictor's inertia with the other charge's potential (`CODE.md`,
# "The release" under "The breathing instability"): whether it removes
# the bias of the predictor's `ä`, about `0.07q²` of itself before, and
# the released orbit's eccentricity, `d` rising to 1.0030 at `q² = 0.05`
# and 1.012 at 0.2 near `t = 7.5` before. Each run is one before it with
# the new predictor:
#
# - `pred-q<k>-h<n>`: the standard window with the predictor's `ä` in the
#   source, at `q² = 0.05` and 0.2, `d/2` and `d/4`, beside
#   `jobs/cures.jl`'s `std-q2-h<n>-pred` and `jobs/addot_error.jl`'s
#   `q1-h<n>-pred`;
# - `solved-q01-h4`: the solved `ä` at `q² = 0.01`, `d/4`, below the
#   breathing's threshold, beside `jobs/addot_error.jl`'s `q01-h4-solved`;
# - `smoke`: `jobs/cures.jl`'s, a few steps at order 2, first;
# - `win-q<k>-h2`: the window `2.5–5` with the predictor's `ä`, on
#   `jobs/cures.jl`'s two levels at `w0 = 8`, at `q² = 0.05` and 0.2,
#   `d/2`, beside its `win-q2-h2-predicted` and `win-q1-h2`. The standard
#   window's fits scatter by 0.1 to 0.4 per solve from the face
#   crossings, the wide window's by 0.004, so these read the bias; and
#   whether the windows still differ, by 2 % in `ΔE/∫P` at 0.2 before.
#
# The rest is `jobs/cures.jl`'s: three levels at `w0 = 4`, shells to
# `R₁ = 50`, the orbit's plane halfway between element faces,
# `ℓ = q²/5`, the force taking over over `onset = 4`, to `t = 20`, a row
# every step with the fit of the solved `ä`'s departure from the
# predictor's (`alpha<k>`, `beta<k>`). One AMD node per run, about five
# hours at `d/4`:
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/inertia.jl
#     simlaunch submit twoldg-<commit> jobs/inertia.jl --site symmetry --run smoke
bb = "bin/binary.jl --R1=50 --hshell=1 --every=1 --jevery=0.125 --tend=20 --onset=4"
b3 = "$bb --levels=3 --w0=4"
b2 = "$bb --levels=2 --w0=8"
h = Dict(2 => "--hdiv=2 --oz=0.25", 4 => "--hdiv=4 --oz=0.125")
q = Dict("q01" => "--q2=0.01 --ell=0.002", "q2" => "--q2=0.05 --ell=0.01",
         "q1" => "--q2=0.2 --ell=0.04")

(name = "inertia",
 description = "TreeWaveOLDG: the predictor's inertia with the other charge's potential, its bias and the release",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $(q["q1"]) --onset=0.0625 --order=2");
        [(label = "pred-$qk-h$n", cmd = "$b3 $(h[n]) $(q[qk]) --source-addot=predicted")
         for qk in ("q2", "q1") for n in (2, 4)];
        (label = "solved-q01-h4", cmd = "$b3 $(h[4]) $(q["q01"])");
        [(label = "win-$qk-h2",
          cmd = "$b2 $(h[2]) $(q[qk]) --window=2.5,5 --source-addot=predicted")
         for qk in ("q2", "q1")]])

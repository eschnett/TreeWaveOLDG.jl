# Two cures of the breathing instability side by side (`CODE.md`, "The
# breathing instability" under "Measured results"), each at the movie's
# `q² = 0.2`, where it must hold the binary stable, and at `q² = 0.01`,
# where `jobs/addot_error.jl`'s order-3 runs with the standard window
# are the reference for what it costs, at `d/2` and `d/4`:
#
# - `win-q<k>-h<n>`: the window twice as wide, `2.5–5`, with the
#   predictor's `ä` in the source. Each alone slows the growth or stops it
#   (`q1-h2-w8-win`, `q1-h2-pred` of `jobs/instability.jl`), and the wider
#   window cut the solved jets' ringing on the circle. Two levels at
#   `w0 = 8`, so that the window's ball stays on the finest level;
# - `o2-q<k>-h<n>`: the near field carried to order 2 instead of 3, with
#   the solved jets, which has no `ä` term to close the loop
#   (`q1-h2-o2`);
# - at `q² = 0.05` (`q2`), where the two windows' inspirals should differ
#   by less than at 0.2, which of them is right: the wide window with the
#   predicted and with the solved `ä` (`win-q2-h<n>-<s>`), which the
#   model leaves stable there, just (gain 0.81, its slowest
#   mode decaying at 0.05 per unit), and the standard window
#   with the predicted `ä` (`std-q2-h<n>-pred`), beside
#   `jobs/instability.jl`'s `q2-h<n>`, the standard window with the solved
#   `ä`, until its breathing mode grows;
# - the predictor's `ä` plus the running mean over `τ = 2` of its bias
#   against the solved one (`--source-addot=corrected`), which the model
#   holds at `q² = 0.2` with either window (slowest modes −0.25 and
#   −0.34), at `q² = 0.05` and 0.2, with the standard window (`cor-q<k>-std-h<n>`)
#   and the wide one (`cor-q<k>-win-h<n>`), beside the wide window's
#   solved `ä` at 0.05, which has no bias.
#
# The rest is `jobs/addot_error.jl`'s: shells to `R₁ = 50`, the orbit's
# plane halfway between element faces, `ℓ = q²/5`, the force taking over
# over `onset = 4`, to `t = 20`, a row every step with the positions,
# velocities and the predictor's `ä`. One AMD node per run; the wide
# window at `d/4` takes most of a day:
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/cures.jl
#     simlaunch submit twoldg-<commit> jobs/cures.jl --site symmetry --run smoke
#     simlaunch submit twoldg-<commit> jobs/cures.jl --site symmetry --run win-q1-h2
bb = "bin/binary.jl --R1=50 --hshell=1 --every=1 --jevery=0.125 --tend=20 --onset=4"
b3 = "$bb --levels=3 --w0=4"
b2 = "$bb --levels=2 --w0=8"
h = Dict(2 => "--hdiv=2 --oz=0.25", 4 => "--hdiv=4 --oz=0.125")
q = Dict("q01" => "--q2=0.01 --ell=0.002", "q2" => "--q2=0.05 --ell=0.01",
         "q1" => "--q2=0.2 --ell=0.04")

(name = "cures",
 description = "TreeWaveOLDG: the breathing instability's cures, a wider window with the predicted ä, order 2, and the bias-corrected ä",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 20 \"$d/output.log\"; done",),
 run = [(label = "smoke",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $(q["q1"]) --onset=0.0625 --order=2");
        (label = "smoke-cor",
         cmd = "bin/binary.jl --hdiv=2 --levels=1 --w0=4 --R1=8 --hshell=1 --rx1=10 --rx2=12 --tend=0.125 --jevery=0.0625 --oz=0.25 $(q["q1"]) --onset=0.0625 --source-addot=corrected --tau=2");
        [(label = "win-$qk-h$n",
          cmd = "$b2 $(h[n]) $(q[qk]) --window=2.5,5 --source-addot=predicted")
         for qk in ("q1", "q01") for n in (2, 4)];
        [(label = "o2-$qk-h$n", cmd = "$b3 $(h[n]) $(q[qk]) --order=2")
         for qk in ("q1", "q01") for n in (2, 4)];
        [(label = "win-q2-h$n-$s",
          cmd = "$b2 $(h[n]) $(q["q2"]) --window=2.5,5 --source-addot=$s")
         for n in (2, 4) for s in ("predicted", "solved")];
        [(label = "std-q2-h$n-pred", cmd = "$b3 $(h[n]) $(q["q2"]) --source-addot=predicted")
         for n in (2, 4)];
        [(label = "cor-$qk-$wk-h$n",
          cmd = "$(wk == "std" ? b3 : "$b2 --window=2.5,5") $(h[n]) $(q[qk]) --source-addot=corrected --tau=2")
         for qk in ("q2", "q1") for wk in ("std", "win") for n in (2, 4)]])

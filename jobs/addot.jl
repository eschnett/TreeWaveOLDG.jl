# The binary's solved `ä` at element faces (`CODE.md`, "The binary" under
# "Measured results"): two runs beside `binary.jl`'s `binary-h4`, on the
# same layout and step, one AMD node each.
#
# - `ell2-h4`: particles of twice the size, `ℓ = 2·10⁻³`, to `t = 16`. If
#   the jumps go as `(v/ℓω)²` times the force's relative jump, they fall
#   about fourfold.
# - `addot-h4`: the source written with the predictor's `ä` in place of
#   the solved one, to `t = 32`; against `binary-h4`, what the solved `ä`'s
#   jumps do to the forces and the orbit.
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/addot.jl
#     simlaunch submit twoldg-<commit> jobs/addot.jl --site symmetry --run ell2-h4 --run addot-h4
bi = "bin/binary.jl --levels=3 --w0=4 --R1=50 --hshell=1 --every=1 --hdiv=4"

(name = "addot",
 description = "TreeWaveOLDG M6b: the binary's solved ä at element faces",
 pin = true,
 site = (symmetry = (time = "12:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; tail -n 40 \"$d/output.log\"; done",),
 run = [(label = "ell2-h4", cmd = "$bi --tend=16 --ell=2e-3"),
        (label = "addot-h4", cmd = "$bi --tend=32 --source-addot=predicted --flux=flux-addot-h4.tsv")])

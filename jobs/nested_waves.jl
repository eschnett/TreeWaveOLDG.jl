# INSPIRAL-PLAN.md's M3b: `bin/nested_waves.jl`, the exact outgoing pulse
# `[g(r - t) - g(r + t)]/r` through nested boxes, the field alone, at
# wavelengths 16 (the first run's radiation) and 4 (shorter than any
# physical wave, standing in for the start-up junk), on the layout
# proposed for M4 to M7 (the finest cube of half-width `w0 = 4` at
# `hₑ = 1/8`, 64 elements across each level) with 4 and 5 levels, each to
# `t = 2^levels w0`, when the pulse has been to the periodic box's faces
# and back to the centre. What comes back into the finest cube after the
# pulse has left it is the coarse-fine faces' reflection, which arrives
# from the face at `w` no sooner than about `2w` ("Nested boxes" under
# "Measured results" in `CODE.md`).
runs = [(label = "waves-l$lambda-n$n",
         cmd = "bin/nested_waves.jl --levels=$n --w0=4 --hdiv=8 --lambda=$lambda --every=0.5")
        for n in (4, 5), lambda in (16, 4)]

(name = "nested_waves",
 description = "TreeWaveOLDG M3b: outgoing waves through nested boxes",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = vec(runs))

# INSPIRAL-PLAN.md's M8b: `bin/nested_ball.jl`, `nested_waves.jl`'s exact
# outgoing pulse through nested cubes inside the seven-patch ball, the
# field alone, at wavelengths 16 (the first run's radiation) and 4 (the
# start-up junk's stand-in). The cube holds the three finest levels of
# M3b's layout for M4 to M7 (the finest cube of half-width `w0 = 4` at
# `hₑ = 1/8`, 64 elements across each level, so `Lc = 16`), and the
# shells continue it to a radiative sphere at `R1 = 48` with radial
# elements of about `hshell = 2`, which resolves `λ = 16` at 8 elements
# per wavelength. Each run goes to `t = 2 R1`, when the sphere's return
# has reached the centre. Its errors in the balls of the cube's levels go
# beside `nested_waves.jl`'s on the nested boxes ("A refined cube" in
# `CODE.md`).
runs = [(label = "ball-l$lambda-n3",
         cmd = "bin/nested_ball.jl --levels=3 --w0=4 --hdiv=8 --lambda=$lambda --R1=48 --hshell=2 --every=0.5 --cost=yes")
        for lambda in (16, 4)]

(name = "nested_ball",
 description = "TreeWaveOLDG M8b: outgoing waves through nested cubes in the ball",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = runs)

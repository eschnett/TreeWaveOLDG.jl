# INSPIRAL-PLAN.md's M8a: `bin/sphere_waves.jl`, the exact outgoing pulse
# of `bin/nested_waves.jl` and its dipole through the seven-patch ball's
# radiative sphere, the field alone. What comes back to the centre after
# the pulse has left is what the sphere and the shells send back ("The
# radiative sphere" under "Measured results" in `CODE.md`). Each run is
# `(R1, Lc, n, m, lambda, ell)`: the sphere's radius, the cube's
# half-width, the roots along the cube's side and the shells' radius, the
# wavelength and the multipole.
cases = [(50, 16, 16, 5, 16, 0), (50, 16, 16, 5, 16, 1), (50, 16, 16, 5, 4, 0),
         (25, 8, 8, 3, 16, 0), (25, 8, 8, 3, 16, 1), (25, 8, 8, 3, 4, 0)]
runs = [(label = "sphere-R$R1-n$n-l$lambda-ell$ell",
         cmd = "bin/sphere_waves.jl --R1=$R1 --Lc=$Lc --n=$n --m=$m --E=4 --lambda=$lambda --ell=$ell --every=1 --radii=4,8,16")
        for (R1, Lc, n, m, lambda, ell) in cases]

(name = "sphere_waves",
 description = "TreeWaveOLDG M8a: outgoing waves through the radiative sphere",
 pin = true,
 site = (symmetry = (time = "24:00:00",),),
 after = (cmd = raw"for d in $SIMLAUNCH_JOBDIRS; do echo \"== $d\"; cat \"$d/output.log\"; done",),
 run = runs)

# INSPIRAL-PLAN.md's M6b, the flux check: `bin/prescribed.jl`'s circle at
# `v = 0.2` from the retarded field, in the plan's ball (three levels at
# `w0 = 4`, `Lc = 16`, shells of radial width 1 to `R1 = 50`), at `d/4`,
# `d/6` and `d/8`, to `t = 32`, two orbits, with the flux through the
# spheres of radius 4, 6, 8 and 12 in the cube every `1/16` beside the
# retarded field's at the same points (`--flux`). The retarded data reach
# beyond the sphere, so the flux is the circle's from `t = 0`; what the
# sphere sends back of the under-resolved error reaches the sphere of 12
# from about `(2R1 − R₂ − 12)/2 = 42`. One H200, one run after another:
#
#     simlaunch submit <study> jobs/flux.jl --site symmetry
#
# `bin/flux.jl 4=… 6=… 8=…` compares the three files.
pr = "bin/prescribed.jl --backend=cuda --path=circle --data=retarded"
ball = "--v=0.2 --levels=3 --w0=4 --order=3 --ball --R1=50 --hshell=1 --rx1=100 --rx2=110 --tend=32 --every=0.5"
flux(h) = (label = "flux-h$h",
           cmd = "$pr --hdiv=$h $ball --flux=flux-h$h.tsv --fradii=4,6,8,12")

(name = "flux",
 description = "TreeWaveOLDG M6b: the circle's flux through spheres against the retarded field's",
 env = "cuda",
 site = (symmetry = (time = "8:00:00", resources = (gpus = 1,), queue = "h200q",
                     sbatch_args = ["--mem=128G"]),),
 run = [flux(4), flux(6), flux(8)])

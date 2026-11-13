#!/bin/bash
# What the seven-patch ball's transitions send back, measured on one H200
# of Symmetry: the reflection scans of M8c ("Particles in the ball" in
# `CODE.md`). Before the sphere's return, at about `(2R1 − R₂)/2` for the
# under-resolved error and `2R1 − R₂` for a resolved wave, what comes back
# into the window is what the cube's coarse-fine faces and the cube–shell
# faces reflect; the scans vary the resolution on either side of them.
#
#     sbatch --export=ALL,SCAN=tests  bin/symmetry/gpu_scans.sh   # first
#     sbatch --export=ALL,SCAN=pulses bin/symmetry/gpu_scans.sh
#     sbatch --export=ALL,SCAN=circle,HSHELL=1 bin/symmetry/gpu_scans.sh
#     sbatch --export=ALL,SCAN=rest   bin/symmetry/gpu_scans.sh
#
# `SCAN=tests` builds the environment, precompiles it on the GPU node and
# runs `test/device_tests.jl` on CUDA, with the particles in the ball, and
# exits with the tests' status, so that the scans can wait for it
# (`--dependency=afterok:<id>`) and load what it compiled.
#
# `SCAN=pulses`: `bin/nested_ball.jl`'s outgoing pulse from the origin, in
# `Float64`, through three nested cubes (`w0 = 4`, `hₑ = 1/4`, so the
# cube's root element is 1 wide at `Lc = 16`) and shells to `R1 = 64`, for
# `λ = 4, 8, 16` (4, 8 and 16 coarse elements per wavelength at the face
# at 8) against shells of radial width 1, 2 and 4 (1:1, 2:1 and 4:1 at the
# cube–shell face), and graded from 1 to 4, to `t = 60`, before the
# sphere's under-resolved return at 64; the errors in the balls of radius
# 4, 8 and 16 every half time unit. `PULSE_LAMBDAS` and `PULSE_SHELLS`
# change the scan.
#
# `SCAN=circle`: `bin/prescribed.jl`'s circle at `v = 0.2`, `d/8`, order 3,
# from the retarded field, in the plan's ball (three levels at `w0 = 4`,
# `Lc = 16`) with shells of radial width `HSHELL` (default 1, twice the
# cube's root element width) to `R1 = 60`, to `t = 40`. The retarded data
# reach `rx1 = 48`, blended out by 54, so that what the blend sends in
# arrives after the run; the circle's radiation (`λ ≈ 16`) comes back from
# the cube–shell face at 16 into the ball of radius 8 from `t ≈ 24`, and
# into `R₁` from `t ≈ 30`. The error against the retarded field in `R₁`
# and in the balls of radius 4 and 8, every half time unit; the force every
# step.
#
# `SCAN=rest`: `bin/uniform_motion.jl`'s particle at rest at `d/8` in the
# ball of the CPU runs (`ball_motion.sh`, `w0 = 3`, `Lc = 12`) at
# `R1 = 72`, to `t = 80`, past the sphere's under-resolved return at about
# `(2R1 − R₂)/2 = 70.75`: the third point, beside 36 and 50, for how that
# return falls with `R1`.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`. The
# environment is one of its own that `Pkg.develop`s the checkout and adds
# CUDA, as `gpu.sh`'s. Each run's rows go to a file of their own in
# `$OUTDIR` as they are printed.

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=treewaveoldg-gpu-scans
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/gpu-scans-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_GPU_ENV:-$SCRATCH/env-gpu-scans}"
SCAN="${SCAN:?set SCAN to tests, pulses, circle or rest}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/gpu-scans-$SCAN-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENVDIR" "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG M8c reflection scan: $SCAN" 2

echo "# $(hostname) $(date -Iseconds) SCAN=$SCAN"
nvidia-smi --query-gpu=name,memory.total --format=csv
julia --version
JL=(julia --project="$ENVDIR" -t "${SLURM_CPUS_PER_TASK:-16}")

if [ "$SCAN" = tests ]; then
    simwatch_stage "environment"
    "${JL[@]}" -e "
        using Pkg
        Pkg.Registry.update()
        Pkg.develop(path = \"$REPO\")
        for p in (\"CUDA\", \"TreeAMR\", \"StaticArrays\", \"ForwardDiff\", \"Test\")
            p in keys(Pkg.project().dependencies) || Pkg.add(p)
        end
        \"IMEXRungeKutta\" in keys(Pkg.project().dependencies) ||
            Pkg.add(url = \"https://github.com/eschnett/IMEXRungeKutta.jl\", rev = \"main\")
        Pkg.instantiate()
        Pkg.precompile()
        using CUDA; CUDA.versioninfo()"
    simwatch_stage "device tests"
    TREEWAVEOLDG_TEST_BACKEND=cuda "${JL[@]}" "$REPO/test/device_tests.jl" \
        2>&1 | tee "$OUTDIR/device_tests.log"
    exit "${PIPESTATUS[0]}"
fi

simwatch_stage "scan $SCAN"
case $SCAN in
    pulses)
        for lambda in ${PULSE_LAMBDAS:-4 8 16}; do
            for shell in ${PULSE_SHELLS:-1 2 4 1:4}; do
                hshell=${shell#*:}
                face=""
                [ "$shell" != "$hshell" ] && face="--hface=${shell%%:*}"
                "${JL[@]}" "$REPO/bin/nested_ball.jl" --backend=cuda --type=f64 \
                    --levels=3 --w0=4 --hdiv=4 --lambda="$lambda" --R1=64 \
                    --hshell="$hshell" $face --tend=60 --every=0.5 --radii=4,8,16 \
                    > "$OUTDIR/pulse-l$lambda-h${shell/:/to}.tsv" \
                    || echo "=== failed: λ = $lambda, shells $shell ==="
            done
        done
        ;;
    circle)
        h=${HSHELL:-1}
        "${JL[@]}" "$REPO/bin/prescribed.jl" --backend=cuda --path=circle \
            --data=retarded --v=0.2 --hdiv=8 --levels=3 --w0=4 --order=3 --ball \
            --R1=60 --hshell="$h" --rx1=48 --rx2=54 --tend=40 --every=0.5 \
            --radii=4,8 --cost > "$OUTDIR/circle-h$h.tsv"
        ;;
    rest)
        "${JL[@]}" "$REPO/bin/uniform_motion.jl" --backend=cuda --ball --R1=72 \
            --levels=3 --w0=3 --hdiv=8 --v=0 --tend=80 --every=0.25 --cost \
            > "$OUTDIR/rest-R72-h8.tsv"
        ;;
    *)
        echo "unknown SCAN=$SCAN"
        exit 1
        ;;
esac

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

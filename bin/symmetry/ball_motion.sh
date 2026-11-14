#!/bin/bash
# INSPIRAL-PLAN.md's M8c on one Symmetry AMD node, 64 threads:
# `bin/uniform_motion.jl --ball` in the seven-patch ball's cube, beside
# the same runs on nested boxes. The cube holds M3b's three finest levels
# about M3's box (the finest cube of half-width `w0 = 3`, so the cube's
# half-width is 12), with shells of radial elements twice the cube's root
# width to a Sommerfeld sphere at `R1` (`bin/ball.jl`). First M3b's
# convergence series to `t = 0.5` at `v = 0.2`, in the ball and on the
# periodic box of the cube's three levels, whose numbers the ball has to
# give inside the box's clean time (about 4.75 in the `R₂` ball); then the
# particle at rest at `hₑ = d/8`, to `t = 60`, far past it, for each `R1`
# and once on the box: the error in the `R₂` ball, the force, and the
# mean of `Π_r`, which the net charge drives at `Q/(γV)` in the periodic
# box and which should stay flat in the ball, until what the sphere sends
# back arrives, from about `2R1 − R₂`. The boxes run with the current
# code: M3b's own runs (`nested_motion.sh`, jobs 587559, 587798 and
# 587799) predate M4's averaging and M5's lab slice, which change the
# remainder, so they are no reference.
#
#     bash bin/symmetry/ball_motion.sh setup     # on the login node, once
#     sbatch bin/symmetry/ball_motion.sh
#
# `TREEWAVEOLDG_SERIES` (resolutions, `hdiv`), `_CLEAN` (the at-rest
# runs' resolutions), `_R1` (their spheres; the series takes the first),
# `_CLEAN_TEND`, `_LEVELS` and `_W0` change the runs; an empty `_SERIES`
# or `_CLEAN` skips those runs, and an empty `_BOXES` the boxes' runs (to
# split the spheres over jobs without running the box twice). `2 w0 hdiv` must be a multiple of 8, so
# at `w0 = 3` the series is `d/4, d/8, d/12`.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `nested_motion.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it.
#
# Measured (jobs 596453 to 596455, "Particles in the ball" in `CODE.md`):
# the series takes about an hour; at `d/8` and `R1 = 50` the ball has
# `5.5·10⁷` nodes, `3.5·10⁷` of them in the shells, a step takes 2.3 s
# on an EPYC 7543, and a run to `t = 60`, 15 360 steps, about 10 hours;
# at `R1 = 36` about 8.4, and the boxes about 5.7, so split the spheres
# over jobs with `TREEWAVEOLDG_R1` and run the boxes with one of them. Threads are pinned by Julia (`JULIA_EXCLUSIVE=1`
# with `srun --cpu-bind=none`, `-t 64,0`). Each run's rows go to a file
# of their own in `$OUTDIR` as they are printed, so a job that runs out
# of time keeps what it measured, and are echoed at the end of the job
# log.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=treewaveoldg-m8c-motion
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m8c-motion-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env-m8c}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m8c-motion-${SLURM_JOB_ID:-local}}"
# `-`, not `:-`: set and empty skips the runs.
SERIES="${TREEWAVEOLDG_SERIES-4 8 12}"
CLEAN="${TREEWAVEOLDG_CLEAN-8}"
R1S="${TREEWAVEOLDG_R1:-36 50}"
CLEAN_TEND="${TREEWAVEOLDG_CLEAN_TEND:-60}"
LEVELS="${TREEWAVEOLDG_LEVELS:-3}"
W0="${TREEWAVEOLDG_W0:-3}"
BOXES="${TREEWAVEOLDG_BOXES-yes}"

if [ "${1:-}" = setup ]; then
    mkdir -p "$ENVDIR"
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$ENVDIR" -e "
        using Pkg
        Pkg.Registry.update()
        Pkg.develop(path = \"$REPO\")
        Pkg.instantiate()"
    exit 0
fi
mkdir -p "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG M8c: uniform motion in the ball" 3

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
run() {  # run <outfile> <script> <args…>
    local out=$1 script=$2; shift 2
    srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/$script" "$@" \
        > "$OUTDIR/$out"
}

R1FIRST=${R1S%% *}
simwatch_stage "convergence series"
for h in $SERIES; do
    run "series-R$R1FIRST-h$h.tsv" uniform_motion.jl --ball --R1="$R1FIRST" \
        --levels="$LEVELS" --w0="$W0" --hdiv="$h" --v=0.2 --tend=0.5 --every=0.125 --cost
    if [ -n "$BOXES" ]; then
        run "series-box-h$h.tsv" uniform_motion.jl --levels="$LEVELS" --w0="$W0" \
            --hdiv="$h" --v=0.2 --tend=0.5 --every=0.125 --cost
    fi
done

simwatch_stage "at rest, past the boxes' clean time"
for h in $CLEAN; do
    for R in $R1S; do
        run "rest-R$R-h$h.tsv" uniform_motion.jl --ball --R1="$R" --levels="$LEVELS" \
            --w0="$W0" --hdiv="$h" --v=0 --tend="$CLEAN_TEND" --every=0.0625 --cost
    done
    if [ -n "$BOXES" ]; then
        run "rest-box-h$h.tsv" uniform_motion.jl --levels="$LEVELS" --w0="$W0" \
            --hdiv="$h" --v=0 --tend="$CLEAN_TEND" --every=0.0625 --cost
    fi
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

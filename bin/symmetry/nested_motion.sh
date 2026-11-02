#!/bin/bash
# INSPIRAL-PLAN.md's M3b on one Symmetry AMD node, 64 threads:
# `bin/uniform_motion.jl` on nested boxes about M3's box (the finest cube
# of half-width `w0 = 3`, M3's side 6). First M3's convergence series to
# `t = 0.5` at `v = 0.2` on four levels, with the cost of each resolution,
# to set against M3's uniform box; then the clean time, the same particle
# at rest at `hₑ = d/8` to `t = 20` on 4 and 5 levels, sampled every
# `1/16`, with the cost of each layout, for when anything arrives from
# which face (the first coarse-fine face at 3, the further ones at 6,
# 12, …, the outer faces at `2^(levels-1)·3`) and the net charge's drift
# as the box grows. At rest, because the window is fixed about the
# centre: at `v = 0.2` the particle leaves the `R₁` ball at `t ≈ 5.7` and
# the finest cube at `t = 15`, and the first is what job 587559 saw
# (`CODE.md`, "Nested boxes" under "Measured results").
#
#     bash bin/symmetry/nested_motion.sh setup     # on the login node, once
#     sbatch bin/symmetry/nested_motion.sh
#
# `TREEWAVEOLDG_SERIES` (resolutions, `hdiv`), `_SERIES_LEVELS`,
# `_LEVELS` (the clean-time runs' level counts), `_CLEAN` (their
# resolutions), `_CLEAN_V` (their speed), `_CLEAN_TEND` and `_W0` change
# the runs; an empty `_SERIES` or `_CLEAN` skips those runs. On nested
# boxes `2 w0 hdiv` must be a multiple of 8, so at `w0 = 3` the series is
# `d/4, d/8, d/12`.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `uniform_motion.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it.
#
# Measured in job 587559 on an EPYC 7543 node: the series takes about 17
# minutes, and a clean-time run on four levels at `d/8` about 1.1 s a
# step, so about 1h40 to `t = 20`; five levels add a quarter. Split
# them over several jobs with the variables above if `amdq` is full.
# Threads are pinned by Julia (`JULIA_EXCLUSIVE=1` with
# `srun --cpu-bind=none`, `-t 64,0`). Each run's rows go to a file of
# their own in `$OUTDIR` as they are printed, so a job that runs out of
# time keeps what it measured, and are echoed at the end of the job log.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=treewaveoldg-m3b-motion
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m3b-motion-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m3b-motion-${SLURM_JOB_ID:-local}}"
# `-`, not `:-`: set and empty skips the runs.
SERIES="${TREEWAVEOLDG_SERIES-4 8 12}"
SERIES_LEVELS="${TREEWAVEOLDG_SERIES_LEVELS:-4}"
LEVELS="${TREEWAVEOLDG_LEVELS:-4 5}"
CLEAN="${TREEWAVEOLDG_CLEAN-8}"
CLEAN_V="${TREEWAVEOLDG_CLEAN_V:-0}"
CLEAN_TEND="${TREEWAVEOLDG_CLEAN_TEND:-20}"
W0="${TREEWAVEOLDG_W0:-3}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M3b: uniform motion on nested boxes" 3

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

simwatch_stage "convergence series"
for n in $SERIES_LEVELS; do
    for h in $SERIES; do
        run "series-n$n-h$h.tsv" uniform_motion.jl --levels="$n" --w0="$W0" --hdiv="$h" \
            --v=0.2 --tend=0.5 --every=0.125 --cost
    done
done

simwatch_stage "clean time"
for h in $CLEAN; do
    for n in $LEVELS; do
        run "clean-n$n-h$h.tsv" uniform_motion.jl --levels="$n" --w0="$W0" --hdiv="$h" \
            --v="$CLEAN_V" --tend="$CLEAN_TEND" --every=0.0625 --cost
    done
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

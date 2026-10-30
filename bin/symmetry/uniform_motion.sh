#!/bin/bash
# INSPIRAL-PLAN.md's M3 on one Symmetry AMD node, 64 threads:
# `bin/uniform_motion.jl` as a convergence series, `hₑ = d/4 … d/12` at
# `v = 0` and `v = 0.2` to `t = 0.5`, with the cost of each resolution;
# `bin/projection.jl` on the first run's source; and the clean time, the
# same particle at `hₑ = d/8` and `d/12` to `t = 5`, past the time the
# error of the box's faces reaches the centre.
#
#     bash bin/symmetry/uniform_motion.sh setup     # on the login node, once
#     sbatch bin/symmetry/uniform_motion.sh
#
# `TREEWAVEOLDG_SERIES`, `_SPEEDS`, `_TEND` and `_CLEAN` change the series'
# resolutions, speeds and end time and the clean-time resolutions; an
# empty `_SERIES` or `_CLEAN` skips those runs.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `benchmark.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it.
#
# Threads are pinned by Julia (`JULIA_EXCLUSIVE=1` with
# `srun --cpu-bind=none`, `-t 64,0`). Each run's rows go to a file of
# their own in `$OUTDIR` as they are printed, so a job that runs out of
# time keeps what it measured, and are echoed at the end of the job log.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-m3
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m3-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m3-${SLURM_JOB_ID:-local}}"
# `-`, not `:-`: set and empty skips the runs.
SERIES="${TREEWAVEOLDG_SERIES-4 6 8 10 12}"
SPEEDS="${TREEWAVEOLDG_SPEEDS-0.2 0.0}"
TEND="${TREEWAVEOLDG_TEND:-0.5}"
CLEAN="${TREEWAVEOLDG_CLEAN-8 12}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M3: uniform motion" 4

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
for v in $SPEEDS; do
    for h in $SERIES; do
        run "series-v$v-h$h.tsv" uniform_motion.jl --hdiv="$h" --v="$v" --tend="$TEND" \
            --every=0.125 --cost
    done
done

simwatch_stage "projection cost"
run projection-h8.tsv projection.jl --hdiv=8 --reps=3
run projection-h8-offgrid.tsv projection.jl --hdiv=8 --reps=3 --offgrid

simwatch_stage "clean time"
for h in $CLEAN; do
    run "clean-h$h.tsv" uniform_motion.jl --hdiv="$h" --v=0.2 --tend=5 --every=0.0625
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

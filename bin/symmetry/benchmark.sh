#!/bin/bash
# What one Symmetry AMD node does with TreeWaveOLDG: `bin/benchmark.jl` at
# 1, 8, 16, 32 and 64 threads, 3D, P = 3, 16.8M nodes in two block sizes
# (4096 blocks of 16³ and 512 of 32³).
#
#     sbatch bin/symmetry/benchmark.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`. The
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, so the package's own environment is never touched, and it is
# precompiled here, on the node type that runs it.
#
# To compare against another version in the same job, on the same node
# (two nominally identical nodes have measured 20% apart), set
# `TREEWAVEOLDG_BASELINE_REPO` to its checkout and
# `TREEWAVEOLDG_BASELINE_SCRIPT` to a script that prints the same rows
# there; it runs first, at the same thread counts and sizes.
#
# Threads are pinned by Julia (`JULIA_EXCLUSIVE=1` with
# `srun --cpu-bind=none`, and `-t N,0` so that no interactive thread is
# added), and every array's pages are first written by the thread that
# owns them. Below 64 threads, pinning fills one NUMA domain after
# another, so only the 64-thread rows compare like with like. The rows
# are kept in `$OUTDIR` and echoed at the end of the job log.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-bench
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-bench-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
BASE_REPO="${TREEWAVEOLDG_BASELINE_REPO:-}"
BASE_SCRIPT="${TREEWAVEOLDG_BASELINE_SCRIPT:-}"
SCRATCH=/mnt/beegfs/eschnetter/claude
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/treewaveoldg-bench-env}"
BASE_ENVDIR="$ENVDIR-baseline"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/treewaveoldg-bench-${SLURM_JOB_ID:-local}}"
SCAN="${TREEWAVEOLDG_SCAN:-4:16,8:8}"
THREADS="${TREEWAVEOLDG_THREADS:-1 8 16 32 64}"
mkdir -p "$ENVDIR" "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG benchmark" 3

simwatch_stage "setting up the environments"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
setup() {  # setup <envdir> <checkout>
    julia --project="$1" -e "
        using Pkg
        Pkg.Registry.update()
        Pkg.develop(path = \"$2\")
        Pkg.instantiate()
        Pkg.precompile()"
}
setup "$ENVDIR" "$REPO"
if [ -n "$BASE_REPO" ]; then
    mkdir -p "$BASE_ENVDIR"
    setup "$BASE_ENVDIR" "$BASE_REPO"
fi

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
bench() {  # bench <envdir> <script> <threads> <outfile>
    srun --cpu-bind=none julia -t "$3,0" --project="$1" "$2" \
        --D=3 --P=3 --scan="$SCAN" --reps=5 > "$OUTDIR/$4"
}

simwatch_stage "baseline"
if [ -n "$BASE_REPO" ]; then
    for t in $THREADS; do
        bench "$BASE_ENVDIR" "$BASE_SCRIPT" "$t" "baseline-$t.tsv"
    done
fi

simwatch_stage "thread scan"
for t in $THREADS; do
    bench "$ENVDIR" "$REPO/bin/benchmark.jl" "$t" "threads-$t.tsv"
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

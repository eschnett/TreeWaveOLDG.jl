#!/bin/bash
# What one Symmetry AMD node does with TreeWaveOLDG: `bin/benchmark.jl` at
# 1, 8, 16, 32 and 64 threads, 3D, P = 3, 16.8M nodes in two block sizes
# (4096 blocks of 16³ and 512 of 32³).
#
#     bash bin/symmetry/benchmark.sh setup      # on the login node, once
#     sbatch bin/symmetry/benchmark.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. The environment is one of its
# own under scratch that `Pkg.develop`s the checkout, so the package's own
# environment is never touched. `setup` builds it on the login node, which
# reaches GitHub (IMEXRungeKutta comes from there), without precompiling;
# the job precompiles it on the node type that runs it, since the login
# node is Intel and its cache would be rejected there.
#
# To compare against another version in the same job, on the same node
# (two nominally identical nodes have measured 20% apart), set
# `TREEWAVEOLDG_BASELINE_REPO` to its checkout and
# `TREEWAVEOLDG_BASELINE_SCRIPT` to a script that prints the same rows
# there; it runs first, at the same thread counts and sizes. Packages
# that script loads by name besides the package go in
# `TREEWAVEOLDG_BASELINE_PACKAGES`, separated by spaces, for `setup`.
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

setup() {  # setup <envdir> <checkout> [<package>…]
    local env=$1 repo=$2; shift 2
    local pkgs=""
    for p in "$@"; do pkgs="$pkgs\"$p\", "; done
    mkdir -p "$env"
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$env" -e "
        using Pkg
        Pkg.Registry.update()
        Pkg.develop(path = \"$repo\")
        isempty([$pkgs]) || Pkg.add([$pkgs])
        Pkg.instantiate()"
}
if [ "${1:-}" = setup ]; then
    setup "$ENVDIR" "$REPO"
    if [ -n "$BASE_REPO" ]; then
        # shellcheck disable=SC2086
        setup "$BASE_ENVDIR" "$BASE_REPO" ${TREEWAVEOLDG_BASELINE_PACKAGES:-}
    fi
    exit 0
fi
mkdir -p "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG benchmark" 3

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'
if [ -n "$BASE_REPO" ]; then
    julia --project="$BASE_ENVDIR" -e 'using Pkg; Pkg.precompile()'
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

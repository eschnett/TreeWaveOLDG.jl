#!/bin/bash
# What one Symmetry AMD node does with the thirteen-patch ball:
# `bin/ball_benchmark.jl` at 1, 8, 16, 32 and 64 threads, `P = 3`, `E = 4`,
# on its benchmark ball: the cube `[−4, 4]³`, `Rm = 12`, `R1 = 100`, 16
# elements across each patch angularly and `hₑ = 1/2` radially, roots
# `(4, 4, 44)`, 1.9·10⁷ nodes; and the same with the inner shells' radial
# elements four times as wide, `(4, 1, 44)`.
#
#     bash bin/symmetry/ball_benchmark.sh setup      # on the login node, once
#     sbatch bin/symmetry/ball_benchmark.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, as for
# `benchmark.sh`, whose environment it shares by default
# (`TREEWAVEOLDG_ENV`). Threads are pinned by Julia (`JULIA_EXCLUSIVE=1`
# with `srun --cpu-bind=none`, `-t N,0`). The rows are kept in `$OUTDIR`
# and echoed at the end of the job log.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-ball-bench
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-ball-bench-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/treewaveoldg-bench-env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/treewaveoldg-ball-bench-${SLURM_JOB_ID:-local}}"
SCAN="${TREEWAVEOLDG_SCAN:-4:4:44,4:1:44}"
THREADS="${TREEWAVEOLDG_THREADS:-1 8 16 32 64}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG ball benchmark" 3

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1

simwatch_stage "thread scan"
for t in $THREADS; do
    srun --cpu-bind=none julia -t "$t,0" --project="$ENVDIR" \
        "$REPO/bin/ball_benchmark.jl" --E=4 --Lc=4 --Rm=12 --R1=100 \
        --scan="$SCAN" --reps=5 > "$OUTDIR/threads-$t.tsv"
done

# The flat box of about the same size on the same node, for the ratio:
# `bin/benchmark.jl`'s 4096 blocks of `16³`, 16.8M nodes.
simwatch_stage "flat box"
srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/benchmark.jl" \
    --D=3 --P=3 --scan=4:16 --reps=5 > "$OUTDIR/flat-64.tsv"

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

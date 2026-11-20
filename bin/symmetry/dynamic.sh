#!/bin/bash
# INSPIRAL-PLAN.md's M6a on one Symmetry AMD node, 64 threads:
# `bin/dynamic.jl` on M4's layout (five nested cubes, the finest of
# half-width `w0 = 4`), each run given as `motion:hdiv:v[:onset]` in
# `TREEWAVEOLDG_RUNS`: `spring`, the particle held on M4's circle by a
# spring and moving under its forces, the force taking over over `onset`
# (default 2; 0 at once), or `circle`, the prescribed circle on the same
# grid and steps (`--prescribed`), both from the circle's retarded field,
# to `TREEWAVEOLDG_TEND` (default 8), with the force every step.
# `TREEWAVEOLDG_EXTRA` is passed on to every run.
#
#     bash bin/symmetry/dynamic.sh setup     # on the login node, once
#     sbatch bin/symmetry/dynamic.sh
#     TREEWAVEOLDG_RUNS="spring:12:0.05" sbatch --partition=amdq --time=24:00:00 bin/symmetry/dynamic.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. The environment is one of its
# own under scratch (`TREEWAVEOLDG_ENV`) that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it; two jobs must not
# precompile it at once, so stagger them (`--dependency=after:<id>+15`).
# The times are in `CODE.md`, "The equations of motion" under "Measured
# results". Each run's rows go to a file of their own in `$OUTDIR` as
# they are printed, so a job that runs out of time keeps what it
# measured, and are echoed at the end of the job log.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-m6a
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m6a-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env-m6a}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m6a-${SLURM_JOB_ID:-local}}"
RUNS="${TREEWAVEOLDG_RUNS:-spring:4:0.05 circle:4:0.05}"
TEND="${TREEWAVEOLDG_TEND:-8}"
EXTRA="${TREEWAVEOLDG_EXTRA:-}"

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
nruns=$(echo $RUNS | wc -w)
simwatch_begin "$OUTDIR" "TreeWaveOLDG M6a: the spring-held particle" $((nruns + 1))

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
for run in $RUNS; do
    IFS=: read -r motion hdiv v onset <<< "$run"
    onset="${onset:-2}"
    simwatch_stage "$motion at d/$hdiv, v = $v"
    case $motion in
        spring) args=(--onset="$onset"); label="-on$onset" ;;
        circle) args=(--prescribed); label="" ;;
        *) echo "unknown motion $motion"; exit 1 ;;
    esac
    # shellcheck disable=SC2086
    srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/dynamic.jl" \
        --hdiv="$hdiv" --v="$v" --tend="$TEND" "${args[@]}" $EXTRA \
        > "$OUTDIR/$motion-h$hdiv-v$v$label.tsv"
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

#!/bin/bash
# INSPIRAL-PLAN.md's M3b on one Symmetry AMD node, 64 threads:
# `bin/nested_waves.jl`, the exact outgoing pulse `[g(r - t) - g(r + t)]/r`
# through nested boxes, the field alone, at wavelengths 16 (the first
# run's radiation) and 4 (shorter than any physical wave, standing in for
# the start-up junk), on the layout proposed for M4 to M7 (the finest
# cube of half-width `w0 = 4` at `hₑ = 1/8`, 64 elements across each
# level) with 4 and 5 levels, each to `t = 2^levels w0`, when the pulse
# has been to the periodic box's faces and back to the centre. What comes
# back into the finest cube after the pulse has left it is the coarse-fine
# faces' reflection, which arrives from the face at `w` no sooner than
# about `2w`.
#
#     bash bin/symmetry/nested_waves.sh setup     # on the login node, once
#     sbatch bin/symmetry/nested_waves.sh
#
# `TREEWAVEOLDG_LAMBDAS`, `_LEVELS`, `_W0`, `_HDIV` and `_EVERY` change
# the runs.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `uniform_motion.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it.
#
# The runs are estimated from M3's costs, not measured, at about eight
# hours together on `amdq` (five levels have about 75M nodes and take
# 32768 steps to `t = 128`); split them over several jobs with the
# variables above if `amdq` is full. Threads are pinned by Julia
# (`JULIA_EXCLUSIVE=1` with `srun --cpu-bind=none`, `-t 64,0`). Each run's
# rows go to a file of their own in `$OUTDIR` as they are printed, so a
# job that runs out of time keeps what it measured, and are echoed at the
# end of the job log.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=treewaveoldg-m3b-waves
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m3b-waves-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m3b-waves-${SLURM_JOB_ID:-local}}"
LAMBDAS="${TREEWAVEOLDG_LAMBDAS:-16 4}"
LEVELS="${TREEWAVEOLDG_LEVELS:-4 5}"
W0="${TREEWAVEOLDG_W0:-4}"
HDIV="${TREEWAVEOLDG_HDIV:-8}"
EVERY="${TREEWAVEOLDG_EVERY:-0.5}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M3b: outgoing waves through nested boxes" 2

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

simwatch_stage "outgoing waves"
for lambda in $LAMBDAS; do
    for n in $LEVELS; do
        run "waves-l$lambda-n$n.tsv" nested_waves.jl --levels="$n" --w0="$W0" \
            --hdiv="$HDIV" --lambda="$lambda" --every="$EVERY"
    done
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

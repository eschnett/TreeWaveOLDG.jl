#!/bin/bash
# INSPIRAL-PLAN.md's M8b on one Symmetry AMD node, 64 threads:
# `bin/nested_ball.jl`, `nested_waves.jl`'s exact outgoing pulse through
# nested cubes inside the seven-patch ball, the field alone, at wavelengths
# 16 (the first run's radiation) and 4 (the start-up junk's stand-in). The
# cube holds the three finest levels of M3b's layout for M4 to M7 (the
# finest cube of half-width `w0 = 4` at `hₑ = 1/8`, 64 elements across each
# level, so `Lc = 16`), and the shells continue it to a radiative sphere at
# `R1 = 48` with radial elements of about `hshell = 2`, which resolves
# `λ = 16` at 8 elements per wavelength (`INSPIRAL-PLAN.md`, M8's
# estimate). Each run goes to `t = 2 R1`, when the sphere's return has
# reached the centre. Its errors in the balls of the cube's levels go beside
# `bin/symmetry/nested_waves.sh`'s on the nested boxes, whose levels inside
# the cube are the same.
#
#     bash bin/symmetry/nested_ball.sh setup     # on the login node, once
#     sbatch bin/symmetry/nested_ball.sh
#
# `TREEWAVEOLDG_LAMBDAS`, `_LEVELS`, `_W0`, `_HDIV`, `_R1`, `_HSHELL` and
# `_EVERY` change the runs.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `nested_waves.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it.
#
# The runs are estimated, not measured: three levels in the cube and the
# shells are about 7·10⁷ nodes and take about 25000 steps to `t = 96`, at
# about twice the nested boxes' cost per node (`--cost` on the laptop), so
# several hours each on `amdq`. The shells' truncation tables are built in
# `BigFloat` on all threads first. Threads are pinned by Julia
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
#SBATCH --job-name=treewaveoldg-m8b-ball
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m8b-ball-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m8b-ball-${SLURM_JOB_ID:-local}}"
LAMBDAS="${TREEWAVEOLDG_LAMBDAS:-16 4}"
LEVELS="${TREEWAVEOLDG_LEVELS:-3}"
W0="${TREEWAVEOLDG_W0:-4}"
HDIV="${TREEWAVEOLDG_HDIV:-8}"
R1="${TREEWAVEOLDG_R1:-48}"
HSHELL="${TREEWAVEOLDG_HSHELL:-2}"
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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M8b: outgoing waves through nested cubes in the ball" 2

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

simwatch_stage "outgoing waves in the ball"
for lambda in $LAMBDAS; do
    for n in $LEVELS; do
        run "ball-l$lambda-n$n.tsv" nested_ball.jl --levels="$n" --w0="$W0" \
            --hdiv="$HDIV" --lambda="$lambda" --R1="$R1" --hshell="$HSHELL" \
            --every="$EVERY" --cost=yes
    done
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

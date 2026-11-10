#!/bin/bash
# INSPIRAL-PLAN.md's M8a on one Symmetry AMD node, 64 threads:
# `bin/sphere_waves.jl`, the exact outgoing pulse of `bin/nested_waves.jl`
# and its dipole through the seven-patch ball's radiative sphere, the
# field alone. What comes back to the centre after the pulse has left is
# what the sphere and the shells send back; the nested boxes' numbers for
# the same pulse are under "Nested boxes" in `CODE.md`'s "Measured
# results".
#
#     bash bin/symmetry/sphere_waves.sh setup     # on the login node, once
#     sbatch bin/symmetry/sphere_waves.sh
#
# `TREEWAVEOLDG_RUNS` is a space-separated list of runs, each
# `R1:Lc:n:m:lambda:ell` (the sphere's radius, the cube's half-width, the
# roots along the cube's side and the shells' radius, the wavelength and
# the angular order), by default the two balls of M8a at `E = 4`:
#
# - `R1 = 50`, `Lc = 16`, `n = 16`, `m = 5`: the cube is the three finest
#   levels' outermost one of M3b's layout for M4 to M7 (`hₑ = 1/2`), the
#   shells replace its two outer levels, 4.8·10⁷ nodes; `hₑ = 1.56` along
#   the sphere at the centre of a shell's face, 1.7 radially;
# - `R1 = 25`, `Lc = 8`, `n = 8`, `m = 3`: the same spacings at half the
#   radius, `kR1` halved, 6.8·10⁶ nodes;
#
# each at `λ = 16` for `ℓ = 0` and `ℓ = 1`, and at `λ = 4` for `ℓ = 0`.
# `_EVERY` and `_RADII` change the sampling.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `nested_waves.sh`, the
# environment is one of its own under scratch (`TREEWAVEOLDG_ENV`) that
# `Pkg.develop`s the checkout, built on the login node by `setup` without
# precompiling, and precompiled by the job on the node type that runs it.
# Several jobs at once, each given a part of the runs through
# `sbatch --export=ALL,TREEWAVEOLDG_RUNS=…`, wait on each other's
# precompilation where their nodes are of one type.
#
# Measured on EPYC 7543 nodes (`CODE.md`, "The radiative sphere"): on the
# larger ball 1¾ hours of serial setup and 10 hours of steps per run, 28
# GB; on the smaller one 15 minutes and 55 minutes, 5 GB. So give each run
# of the larger ball a job of its own. Each run's rows go to a file of
# their own in `$OUTDIR` as they are printed, so a job that runs out of
# time keeps what it measured, and are echoed at the end of the job log.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=treewaveoldg-m8a-sphere
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m8a-sphere-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m8a-sphere-${SLURM_JOB_ID:-local}}"
RUNS="${TREEWAVEOLDG_RUNS:-50:16:16:5:16:0 50:16:16:5:16:1 50:16:16:5:4:0 25:8:8:3:16:0 25:8:8:3:16:1 25:8:8:3:4:0}"
EVERY="${TREEWAVEOLDG_EVERY:-1}"
RADII="${TREEWAVEOLDG_RADII:-4,8,16}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M8a: outgoing waves through the radiative sphere" 2

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
echo "# runs: $RUNS"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1

simwatch_stage "outgoing waves"
for spec in $RUNS; do
    IFS=: read -r R1 Lc n m lambda ell <<< "$spec"
    out="sphere-R$R1-n$n-l$lambda-ell$ell.tsv"
    echo "# $(date -Iseconds) $out"
    srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/sphere_waves.jl" \
        --R1="$R1" --Lc="$Lc" --n="$n" --m="$m" --E=4 --lambda="$lambda" \
        --ell="$ell" --every="$EVERY" --radii="$RADII" > "$OUTDIR/$out"
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

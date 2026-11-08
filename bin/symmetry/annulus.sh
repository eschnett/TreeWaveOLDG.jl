#!/bin/bash
# The four-patch annulus, TreeAMR's M17 acceptance, on one Symmetry AMD
# node: `bin/annulus.jl`, the rotating Neumann Bessel mode on TreeAMR's
# ring of four patches, its convergence over `P = 2 … 5` and root scales
# 1, 2, 4, 8, with and without truncation, and the energy's loss at
# `cfl = 1/2, 1/4, 1/8` ("The annulus" under "Measured results" in
# `CODE.md`).
#
#     bash bin/symmetry/annulus.sh setup     # on the login node, once
#     sbatch bin/symmetry/annulus.sh
#
# `TREEWAVEOLDG_PS`, `_SCALES`, `_M` and `_TURNS` change the runs.
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `uniform_motion.sh`, the
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout, built on the login node by `setup` without precompiling, and
# precompiled by the job on the node type that runs it. TreeAMR 0.2.8,
# with the forests of patches (its M17), comes from the registry.
#
# The runs are small (the largest has about 74 000 nodes) and take
# minutes; each set's rows go to a file of their own in `$OUTDIR` as they
# are printed, and are echoed at the end of the job log. Threads are
# pinned by Julia (`JULIA_EXCLUSIVE=1` with `srun --cpu-bind=none`,
# `-t 64,0`).

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-annulus
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-annulus/annulus-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-annulus
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/annulus-${SLURM_JOB_ID:-local}}"
PS="${TREEWAVEOLDG_PS:-2,3,4,5}"
SCALES="${TREEWAVEOLDG_SCALES:-1,2,4,8}"
M="${TREEWAVEOLDG_M:-2}"
TURNS="${TREEWAVEOLDG_TURNS:-0.25}"

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
simwatch_begin "$OUTDIR" "TreeWaveOLDG: the four-patch annulus" 3

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
run() {  # run <outfile> <args…>
    local out=$1; shift
    srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/annulus.jl" "$@" \
        > "$OUTDIR/$out"
}

simwatch_stage "convergence"
run convergence.tsv --P="$PS" --scales="$SCALES" --truncate=yes,no --m="$M" \
    --turns="$TURNS"

simwatch_stage "energy against the step"
run energy.tsv --P=3 --scales=2 --truncate=yes,no --m="$M" --turns=1 \
    --cfls=0.5,0.25,0.125

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

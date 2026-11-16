#!/bin/bash
# The package's test suite on one Symmetry AMD node, one way per job:
#
#     sbatch --export=ALL,MODE=plain bin/symmetry/suite.sh
#     sbatch --export=ALL,MODE=t4 bin/symmetry/suite.sh    # --threads=4
#     sbatch --export=ALL,MODE=cb bin/symmetry/suite.sh    # --check-bounds=yes
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced without `.git` and `Manifest*.toml`. The plain run takes about
# 51 minutes and the bounds-checked one more than an hour, hence `amdq`.
# The code is compiled for the node's own CPU: with
# `JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all"` Pkg took
# the depot's caches of other targets as current and KernelAbstractions
# failed to precompile against them (jobs 599622–599624). The MPI test
# starts its three ranks with MPICH's local launcher, which needs the
# release-gather off on these nodes.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#SBATCH --job-name=treewaveoldg-suite
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-suite-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
MODE="${MODE:-plain}"
OUTDIR="/mnt/beegfs/eschnetter/claude/treewaveoldg-suite-$MODE-${SLURM_JOB_ID:-local}"
mkdir -p "$OUTDIR"
export HYDRA_LAUNCHER=fork
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0
export TREEWAVEOLDG_TEST_MPI_CONCURRENT=1

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG suite ($MODE)" 1
simwatch_stage "the suite"

cd "$REPO"
echo "# $(hostname) $(date -Iseconds) MODE=$MODE"
lscpu | grep "Model name"
case $MODE in
    plain) args='' ;;
    t4) args='julia_args = ["--threads=4"]' ;;
    cb) args='julia_args = ["--check-bounds=yes"]' ;;
    *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test(; $args)"

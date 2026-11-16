#!/bin/bash
# Two H200s, one problem: the device tests on one GPU, then
# `bin/distributed_gpu.jl` over MPI on two ranks of one GPU each, against
# the same problems on one GPU alone ("Running distributed" in
# `CODE.md`).
#
#     sbatch bin/symmetry/distributed_gpu.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`. The
# environment is one of its own under scratch
# (`TREEWAVEOLDG_MPIGPU_ENV`) that `Pkg.develop`s the checkout and adds
# CUDA and MPI; it is built and precompiled in the job, on the GPU node.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-mpi-gpu
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-mpi-gpu-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude
ENVDIR="${TREEWAVEOLDG_MPIGPU_ENV:-$SCRATCH/treewaveoldg-mpi-gpu-env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/treewaveoldg-mpi-gpu-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENVDIR" "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG on two H200s" 3

simwatch_stage "environment"
echo "# $(hostname) $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.Registry.update()
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"MPI\", \"TreeAMR\", \"StaticArrays\", \"ForwardDiff\", \"Test\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    \"IMEXRungeKutta\" in keys(Pkg.project().dependencies) ||
        Pkg.add(url = \"https://github.com/eschnett/IMEXRungeKutta.jl\", rev = \"main\")
    Pkg.instantiate()
    Pkg.precompile()
    using CUDA; CUDA.versioninfo()"

# MPICH_jll on Symmetry: local processes, not Slurm's launcher, and no
# release-gather in shared memory, which segfaults in the first collective.
export HYDRA_LAUNCHER=fork
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

simwatch_stage "device tests"
echo "=== device tests, CUDA against the host ==="
TREEWAVEOLDG_TEST_BACKEND=cuda julia --project="$ENVDIR" -t 8 "$REPO/test/device_tests.jl" \
    2>&1 | tee "$OUTDIR/device_tests.log" || echo "=== the device tests failed ==="

simwatch_stage "two GPUs"
echo "=== two ranks, one H200 each, against one H200 ==="
julia --project="$ENVDIR" -e "
    using MPI
    cmd = \`\$(MPI.mpiexec()) -n 2 \$(Base.julia_cmd()) -t 4,0 --project=$ENVDIR
           $REPO/bin/distributed_gpu.jl --backend=cuda\`
    run(cmd)" 2>&1 | tee "$OUTDIR/distributed_gpu.tsv"

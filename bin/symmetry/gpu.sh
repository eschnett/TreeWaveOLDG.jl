#!/bin/bash
# What one H200 does with TreeWaveOLDG: the device tests against the host
# in Float64 and Float32, the flat box of `benchmark.sh` and the benchmark
# ball of `ball_benchmark.sh` on the device, and the far rule of the
# source projection, the gate for moving the particles there (`PLAN.md`,
# "Particles on a device").
#
#     sbatch bin/symmetry/gpu.sh
#     sbatch --partition=h200q bin/symmetry/gpu.sh        # when debug is full
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`. The
# environment is one of its own under scratch that `Pkg.develop`s the
# checkout and adds CUDA: neither the package nor its tests may depend on
# a device package, since CI has none. It is built and precompiled in the
# job, on the GPU node. The rows are kept in `$OUTDIR` and echoed at the
# end of the job log; "Threads and devices" in `CODE.md` has the tables.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-gpu
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-gpu-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude
ENVDIR="${TREEWAVEOLDG_GPU_ENV:-$SCRATCH/treewaveoldg-gpu-env}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/treewaveoldg-gpu-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENVDIR" "$OUTDIR"

. "$REPO/bin/symmetry/simwatch.sh"
simwatch_begin "$OUTDIR" "TreeWaveOLDG on an H200" 5

simwatch_stage "environment"
echo "# $(hostname) $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
julia --version
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.Registry.update()
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"TreeAMR\", \"StaticArrays\", \"ForwardDiff\", \"Test\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    \"IMEXRungeKutta\" in keys(Pkg.project().dependencies) ||
        Pkg.add(url = \"https://github.com/eschnett/IMEXRungeKutta.jl\", rev = \"main\")
    Pkg.instantiate()
    Pkg.precompile()
    using CUDA; CUDA.versioninfo()"

JL=(julia --project="$ENVDIR" -t 8)

simwatch_stage "device tests"
echo "=== device tests, CUDA against the host ==="
# A failure is reported and the benchmarks still run.
TREEWAVEOLDG_TEST_BACKEND=cuda "${JL[@]}" "$REPO/test/device_tests.jl" \
    2>&1 | tee "$OUTDIR/device_tests.log" || echo "=== the device tests failed ==="

# `TREEWAVEOLDG_BENCH=0` stops here.
if [ "${TREEWAVEOLDG_BENCH:-1}" = 0 ]; then
    exit 0
fi

simwatch_stage "flat box"
for T in f64 f32; do
    "${JL[@]}" "$REPO/bin/benchmark.jl" --backend=cuda --type=$T --D=3 --P=3 \
        --scan="${TREEWAVEOLDG_SCAN:-4:16,8:8}" --reps=5 > "$OUTDIR/flat-cuda-$T.tsv" \
        || echo "=== failed: the flat box at $T ==="
done

simwatch_stage "benchmark ball"
for T in f64 f32; do
    "${JL[@]}" "$REPO/bin/ball_benchmark.jl" --backend=cuda --type=$T \
        --scan="${TREEWAVEOLDG_BALL_SCAN:-4:4:44}" --reps=5 > "$OUTDIR/ball-cuda-$T.tsv" \
        || echo "=== failed: the benchmark ball at $T ==="
done

simwatch_stage "far rule"
for grid in "" --offgrid; do
    "${JL[@]}" "$REPO/bin/far_rule.jl" --backend=cuda --hdiv=8 --order=3 --reps=3 \
        $grid > "$OUTDIR/far-cuda${grid:+-offgrid}.tsv" \
        || echo "=== failed: the far rule $grid ==="
done

echo "=== results ($OUTDIR) ==="
for f in "$OUTDIR"/*.tsv; do
    echo "--- $(basename "$f")"
    cat "$f"
done

#!/bin/bash
# INSPIRAL-PLAN.md's M4 on one Symmetry AMD node, 64 threads:
# `bin/prescribed.jl` on the layout chosen for M4 to M7 (five nested
# cubes, the finest of half-width `w0 = 4`, `2 w0 hdiv` elements across
# each level), each run given as `path:hdiv:order[:tend]` in
# `TREEWAVEOLDG_RUNS`: the slow circle, `v = 0.05` at radius
# `TREEWAVEOLDG_RC`, to `tend` (default `TREEWAVEOLDG_TEND`) with the
# force every step,
# from instantaneous data (`circle`) or from the retarded field
# (`circle-ret`, with the error against it every `1/4`), and hyperbolic
# motion (`hyperbolic`) of proper acceleration `TREEWAVEOLDG_G` until its
# speed reaches 0.05, or until `tend` where the run names one, with the
# error against the exact retarded field every `1/32`. Each run ends with
# its cost (`--cost`). With `TREEWAVEOLDG_EXTRA=--reading`, nothing is
# evolved: each run reads the exact remainder's force at `tend` (M5).
# `TREEWAVEOLDG_EXTRA` is passed on to every run, for instance `--plain`
# for the source and the force without the averaging, or `--both` for
# the force read both ways.
#
#     bash bin/symmetry/prescribed.sh setup     # on the login node, once
#     sbatch bin/symmetry/prescribed.sh
#     TREEWAVEOLDG_RUNS="circle:12:3" sbatch --partition=amdq … bin/symmetry/prescribed.sh
#
# Run from a checkout of its own (`TREEWAVEOLDG_REPO`, default `$PWD`),
# rsynced to the cluster without `.git` and `Manifest*.toml`, with the
# same environment variables both times. As in `nested_motion.sh`, the
# environment is one of its own under scratch (`TREEWAVEOLDG_ENV`) that
# `Pkg.develop`s the checkout, built on the login node by `setup` without
# precompiling, and precompiled by the job on the node type that runs it;
# two jobs must not precompile it at once, so stagger them
# (`--dependency=after:<id>+15`).
#
# The times are in `CODE.md`, "Prescribed paths" under "Measured
# results"; a step takes about 0.75 s at `d/8`, 2 s at `d/12` and 4.5 s
# at `d/16` on an EPYC 7543, and a circle to `t = 8` takes `256·hdiv`
# steps, so `d/16` needs about 5.5 hours on `amdq`. Split the runs over
# jobs with `TREEWAVEOLDG_RUNS`. Threads are pinned by Julia
# (`JULIA_EXCLUSIVE=1` with `srun --cpu-bind=none`, `-t 64,0`). Each run's
# rows go to a file of their own in `$OUTDIR` as they are printed, so a
# job that runs out of time keeps what it measured, and are echoed at the
# end of the job log.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewaveoldg-m4
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral/m4-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEWAVEOLDG_REPO:-$PWD}"
SCRATCH=/mnt/beegfs/eschnetter/claude/treewaveoldg-inspiral
ENVDIR="${TREEWAVEOLDG_ENV:-$SCRATCH/env-m4}"
OUTDIR="${TREEWAVEOLDG_OUT:-$SCRATCH/m4-${SLURM_JOB_ID:-local}}"
RUNS="${TREEWAVEOLDG_RUNS:-hyperbolic:4:3 hyperbolic:4:2 circle-ret:4:3 circle-ret:4:2}"
RC="${TREEWAVEOLDG_RC:-0.5}"
G="${TREEWAVEOLDG_G:-0.1}"
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
simwatch_begin "$OUTDIR" "TreeWaveOLDG M4: prescribed paths" $((nruns + 1))

simwatch_stage "precompiling"
echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.precompile()'

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
for run in $RUNS; do
    IFS=: read -r path hdiv order tend <<< "$run"
    # The circle's files name their end time, and hyperbolic motion's
    # where the run gives one.
    label=""
    hyptend=()
    if [ "$path" = hyperbolic ]; then
        [ -n "$tend" ] && { label="-t$tend"; hyptend=(--tend="$tend"); }
    else
        tend="${tend:-$TEND}"
        label="-t$tend"
    fi
    simwatch_stage "$path at d/$hdiv, order $order"
    case $path in
        circle) args=(--path=circle --rc="$RC" --tend="$tend" --every=0.25 --fevery=1) ;;
        circle-ret) args=(--path=circle --data=retarded --rc="$RC" --tend="$tend"
                          --every=0.25 --fevery=1) ;;
        hyperbolic) args=(--path=hyperbolic --g="$G" --every=0.03125 --fevery=1
                          "${hyptend[@]}") ;;
        *) echo "unknown path $path"; exit 1 ;;
    esac
    # shellcheck disable=SC2086
    srun --cpu-bind=none julia -t 64,0 --project="$ENVDIR" "$REPO/bin/prescribed.jl" \
        --hdiv="$hdiv" --order="$order" "${args[@]}" --cost $EXTRA \
        > "$OUTDIR/$path-h$hdiv-o$order$label.tsv"
done

for f in "$OUTDIR"/*.tsv; do
    echo "== $(basename "$f")"
    cat "$f"
done

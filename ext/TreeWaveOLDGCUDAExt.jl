# The source passes on CUDA with a register cap ("On a device" in
# CODE.md). The source at order 3 takes every register a thread may have,
# 255, and spills; capped at 128 a thread spills more, but twice as many
# threads fit on a multiprocessor, and the far rule's source pass takes
# a fifth less time on an H200. KernelAbstractions' CUDA back end passes
# no register cap, so these passes are launched through `@cuda` here, the
# rest of the package's kernels as before. Removing this file, and the
# extension in Project.toml, gives the uncapped launch back.
module TreeWaveOLDGCUDAExt

using CUDA: CUDA, CUDABackend, @cuda, cudaconvert, blockIdx, blockDim, threadIdx
import TreeWaveOLDG

const MAXREGS = 128
const THREADS = 128

function capped_points!(body, n, args)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= n && body(i, args...)
    return nothing
end

function TreeWaveOLDG.launch_points!(::CUDABackend, body, n, args)
    n == 0 && return nothing
    dargs = cudaconvert(args)
    kernel = @cuda launch=false maxregs=MAXREGS capped_points!(body, n, dargs)
    threads = min(THREADS, CUDA.maxthreads(kernel))
    kernel(body, n, dargs; threads=threads, blocks=cld(n, threads))
    return nothing
end

end

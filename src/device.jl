# Where the work runs. As in TreeWave, the storage decides: the backend is
# said once, to the field sets, and everything this package keeps for its
# own kernels — the per-block spacings and origins — has to follow it
# there. The element needs no copy: it is a tuple of tuples, so it travels
# as a kernel argument.

"""
    to_backend(backend, a)

`a` on `backend`: itself on the CPU, a fresh device array copied from it
otherwise.
"""
to_backend(::CPU, a::AbstractArray) = a

function to_backend(backend::Backend, a::AbstractArray)
    dev = allocate(backend, eltype(a), size(a))
    copyto!(dev, a)
    return dev
end

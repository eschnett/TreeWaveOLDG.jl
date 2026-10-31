# The operations this package performs that a software floating-point type
# does not provide. Copied from TreeWave's `src/precision.jl`, which has
# the reasons; only what is used here is kept.

"""
    ceilint(x)

`ceil(Int, x)` for a type that may not define `Int(::AbstractFloat)`:
MultiFloats.jl does not, and `BigFloat` is the conversion every float
type has. Taken only once the value is an exact integer, so only ever
exact. Host-side, for step counts.
"""
ceilint(x) = _toint(ceil(x))

_toint(y::Base.IEEEFloat) = Int(y)
_toint(y) = Int(BigFloat(y))

"""
    floorint(x)

`floor(Int, x)`, through the same fallback as [`ceilint`](@ref).
"""
floorint(x) = _toint(floor(x))

"""
    roundint(x)

`round(Int, x)`, through the same fallback as [`ceilint`](@ref).
"""
roundint(x) = _toint(round(x))

"""
    chunk_count(t_end, chunk)

The number of chunks to `t_end`: `⌈t_end / chunk⌉`, except that a quotient
within a few ulp of an integer `m` gives `m`, so that a run meant to be a
whole number of chunks is not given a sliver of a chunk more by rounding
(`0.33 / 0.03 == 11.000000000000002`). TreeWave's rule; pair it with
[`chunk_bounds`](@ref), whose last chunk ends at `t_end` exactly.
"""
function chunk_count(t_end::T, chunk::T) where {T}
    r = t_end / chunk
    m = round(r)
    tol = 4 * (eps(r) + eps(t_end) / chunk)
    return m >= 1 && abs(r - m) <= tol ? roundint(m) : ceilint(r)
end

"""
    chunk_bounds(c, nchunks, t_end, chunk) -> (tstart, t)

The span of chunk `c` of [`chunk_count`](@ref)`(t_end, chunk) == nchunks`:
from `(c - 1) · chunk` to `c · chunk`, both capped at `t_end`, the last one
ending at `t_end` exactly.
"""
function chunk_bounds(c::Integer, nchunks::Integer, t_end::T, chunk::T) where {T}
    tstart = min((c - 1) * chunk, t_end)
    t = c == nchunks ? t_end : min(c * chunk, t_end)
    return tstart, t
end

"""
    wrap(x, L)

`x` reduced into `[0, L)` for positive `L`, spelled `x - L·floor(x/L)`
rather than `mod`, which goes through `rem` and is not defined for
MultiFloats types.
"""
wrap(x, L) = x - L * floor(x / L)

"""
    tofloat64(x)

`x` as a `Float64`, through `BigFloat` for a type that converts to
nothing else. The bridge to the blast wave's `Float64` reference table.
"""
tofloat64(x::Base.IEEEFloat) = Float64(x)
tofloat64(x) = Float64(BigFloat(x))

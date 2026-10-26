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

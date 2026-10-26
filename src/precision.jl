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

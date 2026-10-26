# The reference the kernels are checked against: the same operators as
# dense matrices over a whole periodic line, built from the element's
# tables by a loop that shares nothing with the kernels but the tables,
# and the package's state rearranged into global arrays to apply them to.

# The wide derivative `Dm`, the truncation projector `Πm` and the
# quadrature weights `w` on a periodic line of `K` elements over cells of
# spacing `h`.
function global_operators(el::Element{T,P1}, K, h) where {T,P1}
    n = K * P1
    Dm = zeros(T, n, n)
    Πm = zeros(T, n, n)
    w = zeros(T, n)
    for e in 0:(K - 1), i in 0:(P1 - 1)
        r = e * P1 + i + 1
        w[r] = h * P1 / 2 * el.weights[i + 1]
        for j in -1:P1
            Dm[r, mod(e * P1 + j, n) + 1] += el.stencil[i + 1][j + 2] / h
        end
        for j in 0:(P1 - 1)
            Πm[r, e * P1 + j + 1] = el.truncate[i + 1][j + 1]
        end
    end
    return Dm, Πm, w
end

# The one-dimensional operator `A` acting along dimension `d` of a
# `D`-dimensional array flattened with dimension 1 fastest.
function along(A, d, D)
    n = size(A, 1)
    Id = Matrix{eltype(A)}(I, n, n)
    return reduce(kron, [e == d ? A : Id for e in D:-1:1])
end

# Variable `v` of the state vector `u` of `fs`, as one global array over
# the whole box, `n` nodes per dimension. A block's nodes start at its
# origin divided by the spacing, which is an exact integer on a uniform
# box.
function global_array(fs::FieldSet{T,D}, u, v) where {T,D}
    forest = fs.forest
    N = forest.N
    h = minimum_spacing(forest)
    n = round(Int, (forest.extents[1][2] - forest.extents[1][1]) / h)
    ua = statearray(u, fs)
    A = zeros(T, ntuple(_ -> n, D))
    for b in 1:nblocks(fs)
        origin = block_origin(forest, blockkey(fs, b))
        g = ntuple(d -> round(Int, (origin[d] - forest.extents[d][1]) / h), D)
        for I in CartesianIndices(ntuple(_ -> N, D))
            A[ntuple(d -> g[d] + I[d], D)...] = ua[Tuple(I)..., v, b]
        end
    end
    return A
end

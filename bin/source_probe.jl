# What one evaluation of the particles' source costs on a CUDA device, and
# why: registers, local memory and nanoseconds per point, for the source as
# the far rule calls it and for variants of the call.
#
#     julia --project=<env with CUDA> -t 8 bin/source_probe.jl [--n=2^21] [--reps=5]
#
# The source is `bin/far_rule.jl`'s, two charges of the first run at
# `v = 0.2` (or the first charge alone, `K = 1`), at orders 0 to 3, at `n`
# points in the window's ball: nine tenths in its inner ball, then a tenth
# in the transition, in that order, as a warp of the far rule meets them.
# Each variant is one kernel launch over the points, a work item per
# point. Prints a header row, then one tab-separated row per variant,
# order and particle count:
#
#     variant order K registers local_bytes threads ns_per_point max_rel_diff
#
# `max_rel_diff` is against the host's evaluation of the same points, over
# their largest value. The variants (`CODE.md`, "On a device", has what
# they showed):
#
# - `call`: `S(x)` with the source's `SVector` tables, a kernel argument,
#   its call `@noinline`, as the far rule called it first;
# - `inline`: the same body inlined into the kernel;
# - `tables`, `tables_inline`: the source as `project_far!` passes it now,
#   its coefficient tables in a device array of their own
#   (`device_source`), called and inlined;
# - `tables_const`: the order a constant, so that only its code compiles;
# - at order 3 with both particles, `tables_transition`, every point in
#   the window's transition, where the source carries `∇W`, and
#   `sweep_r<maxregs>_t<threads>`, a register cap and a block size.

using CUDA
using TreeWaveOLDG
using TreeWaveOLDG: SVector, device_source, window_terms, _sum_particles, _inner_term, _transition_term

function option(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, "="; limit=2)[2]
    end
    return default
end

# The body of `(s::ParticleSource)(x)`, inlined.
@inline function inline_source(s::ParticleSource{T,K}, x) where {T,K}
    W, ∇W, ∇²W = window_terms(s.window, x)
    iszero(W) && return zero(W)
    iszero(∇W) && iszero(∇²W) && return W * _sum_particles(_inner_term, Val(K), s, x)
    return _sum_particles(_transition_term, Val(K), s, x, W, ∇W, ∇²W)
end

# `S(x)` with the order a constant, so that the lower orders' code alone
# is compiled (the source's `order` is a run-time field).
@inline function const_source(s::ParticleSource{T,K}, x, ::Val{O}) where {T,K,O}
    W, ∇W, ∇²W = window_terms(s.window, x)
    iszero(W) && return zero(W)
    inner(i, s, x) = TreeWaveOLDG.source_terms(s.particles[i], O, x)[2]
    function transition(i, s, x, W, ∇W, ∇²W)
        Ψ, bracket, ∇WΨ = TreeWaveOLDG.source_terms(s.particles[i], O, x, ∇W)
        return W * bracket - 2∇WΨ - Ψ * ∇²W
    end
    iszero(∇W) && iszero(∇²W) && return W * _sum_particles(inner, Val(K), s, x)
    return _sum_particles(transition, Val(K), s, x, W, ∇W, ∇²W)
end

function k_const(out, S, xs, o)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(out) && @inbounds out[i] = const_source(S, xs[i], o)
    return nothing
end

function k_call(out, S, xs)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(out) && @inbounds out[i] = S(xs[i])
    return nothing
end
function k_inline(out, S, xs)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(out) && @inbounds out[i] = inline_source(S, xs[i])
    return nothing
end
function source(order, K)
    c = SVector(2.5, 2.5, 2.5)
    q, Ω = sqrt(0.005), 0.4
    m₁ = ParticleModel(; Q=q, ℓ=1e-3, M=1.0)
    m₂ = ParticleModel(; Q=-q, ℓ=1e-3, M=1.0)
    r = SVector(0.5, 0.0, 0.0)
    v = SVector(0.0, Ω / 2, 0.0)
    a = -Ω^2 * r
    tr₁ = Trajectory(; z=c + r, v=v, a=a, ȧ=-Ω^2 * v, ä=-Ω^2 * a, a⃛=Ω^4 * v, a⃜=Ω^4 * a)
    tr₂ = Trajectory(; z=c - r, v=-v, a=-a, ȧ=Ω^2 * v, ä=Ω^2 * a, a⃛=-Ω^4 * v,
                     a⃜=-Ω^4 * a)
    w = Window(c, 1.25, 2.5; continuity=4)
    models, trajs = K == 1 ? ((m₁,), (tr₁,)) : ((m₁, m₂), (tr₁, tr₂))
    return ParticleSource(models, w, trajs; order=order), c
end

# `n` points in the window's ball, deterministic: nine tenths within its
# inner radius, a tenth in the transition.
function points(c, n)
    xs = [begin
                θ = 2π * mod(0.618034 * k, 1)
                μ = 2 * mod(0.7548777 * k, 1) - 1
                ρ = k % 10 == 0 ? 1.25 + 1.25 * mod(0.5698403 * k, 1) :
                    1.25 * cbrt(mod(0.4301597 * k, 1))
                s = sqrt(1 - μ^2)
                c + ρ * SVector(s * cos(θ), s * sin(θ), μ)
            end for k in 1:n]
    # The inner ball's points first and the transition's last, as a warp
    # of the far rule meets them: neighbouring points of one element.
    return sort(xs; by=x -> sum(abs2, x - c) > 1.25^2)
end

function probe(variant, kernel, arg, xs, out, host, reps; maxregs=nothing, nthreads=256,
               extra=())
    k = maxregs === nothing ? (@cuda launch=false kernel(out, cudaconvert(arg), xs, extra...)) :
        (@cuda launch=false maxregs=maxregs kernel(out, cudaconvert(arg), xs, extra...))
    threads = min(nthreads, CUDA.maxthreads(k))
    blocks = cld(length(out), threads)
    a = cudaconvert(arg)
    k(out, a, xs, extra...; threads, blocks)
    CUDA.synchronize()
    t = minimum(1:reps) do _
        CUDA.@elapsed k(out, a, xs, extra...; threads, blocks)
    end
    diff = maximum(abs.(Array(out) .- host)) / maximum(abs.(host))
    return (CUDA.registers(k), CUDA.memory(k).local, threads, 1e9 * t / length(out), diff)
end

function main()
    n = Int(eval(Meta.parse(option("n", "2^21"))))
    reps = parse(Int, option("reps", "5"))
    CUDA.versioninfo()
    println(join(("variant", "order", "K", "registers", "local_bytes", "threads",
                  "ns_per_point", "max_rel_diff"), '\t'))
    for K in (1, 2), order in 0:3
        S, c = source(order, K)
        xh = points(c, n)
        host = Vector{Float64}(undef, n)
        Threads.@threads for i in 1:n
            host[i] = S(xh[i])
        end
        xs = CuArray(xh)
        out = CuArray{Float64}(undef, n)
        St = device_source(S, CUDABackend())
        for (name, kernel, arg) in (("call", k_call, S), ("inline", k_inline, S),
                                    ("tables", k_call, St),
                                    ("tables_inline", k_inline, St))
            row = try
                probe(name, kernel, arg, xs, out, host, reps)
            catch err
                (replace(first(sprint(showerror, err), 200), r"\s+" => " "),)
            end
            println(join((name, order, K, row...), '\t'))
            flush(stdout)
        end
        name = "tables_const"
        row = try
            probe(name, k_const, St, xs, out, host, reps; extra=(Val(order),))
        catch err
            (replace(first(sprint(showerror, err), 200), r"\s+" => " "),)
        end
        println(join((name, order, K, row...), '\t'))
        if order == 3 && K == 2
            # Every point in the window's transition, where the source
            # carries `∇W` as a further partial.
            xt = [c + (1.3 + 1.1 * mod(0.381966 * k, 1)) *
                      SVector(cos(k), sin(2k), cos(3k)) / sqrt(cos(k)^2 + sin(2k)^2 + cos(3k)^2)
                  for k in 1:n]
            hostt = [S(x) for x in xt]
            row = probe("tables_transition", k_call, St, CuArray(xt), out, hostt, reps)
            println(join(("tables_transition", order, K, row...), '\t'))
            for maxregs in (nothing, 128), nthreads in (128, 256)
                name = "sweep_r$(something(maxregs, 255))_t$nthreads"
                row = try
                    probe(name, k_const, St, xs, out, host, reps; maxregs, nthreads,
                          extra=(Val(order),))
                catch err
                    (replace(first(sprint(showerror, err), 200), r"\s+" => " "),)
                end
                println(join((name, order, K, row...), '\t'))
                flush(stdout)
            end
        end
    end
    t = @elapsed begin
        S, c = source(3, 2)
        xh = points(c, 2^16)
        s = Threads.Atomic{Float64}(0)
        Threads.@threads for i in eachindex(xh)
            Threads.atomic_add!(s, S(xh[i]))
        end
    end
    println("# host, $(Threads.nthreads()) threads, order 3, K = 2: ",
            1e9 * t / 2^16, " ns per point (including the setup)")
    return nothing
end

main()

# The right-hand side. The kernels are a variable-coefficient stencil
# indexed by each point's place in its element, read across element faces
# inside a block and across block faces through one ghost plane; every one
# of those index computations can be off by one in a way that still runs.
# The reference is the dense global operator of `reference.jl`.

using LinearAlgebra: I, norm, dot
using IMEXRungeKutta: IMEXProblem, RK4, init, solve!, step!

# A smooth periodic test state that is not separable and not a single
# mode, so that a transposed index or a wrong neighbour shows.
function testdata(D)
    return function (x, v)
        s = zero(x[1])
        for d in 1:D
            s += sinpi(2 * d * x[d] + v / 3) * cospi(2 * x[mod1(d + 1, D)])
        end
        return s
    end
end

@testset "The right-hand side is the global OLDG operator, in 1D, 2D and 3D" begin
    # Guards every index the kernels compute: the element a point is in,
    # its window, the ghost plane at block faces, the dimension a
    # derivative runs along, and the tensor-product truncation of the
    # initial `v`. The right-hand side itself never truncates.
    for (D, P, E, roots) in ((1, 3, 1, 3), (2, 2, 2, 3), (2, 3, 1, 2), (3, 1, 1, 3)),
        truncate in (true, false)

        forest = wave_forest(Float64, Val(D), P, E; roots=roots)
        el = Element(Float64, P)
        p = WaveProblem(forest, el; truncate=truncate)
        u = initial_state(p, testdata(D))
        du = similar(u)
        wave_rhs!(du, u, p, 0.0)

        Dm, Πm, _ = global_operators(el, roots * E, minimum_spacing(forest))
        lap = sum(along(Dm, d, D) * along(Dm, d, D) for d in 1:D)
        Π = truncate ? reduce(kron, fill(Πm, D)) : I
        U = vec(global_array(p.fs, u, 1))
        V = vec(global_array(p.fs, u, 2))
        @test vec(global_array(p.fs, du, 1)) == V
        ref = lap * U
        @test norm(vec(global_array(p.fs, du, 2)) - ref, Inf) < 1e-12 * norm(ref, Inf)

        # The initial `v` is projected when truncating, and only then.
        raw = FieldSet(forest, 2; G=1)
        fill_at_nodes!(testdata(D), raw, el)
        r = statevector(raw)
        gather!(r, raw)
        R = vec(global_array(raw, r, 2))
        if truncate
            @test norm(V - Π * R, Inf) < 1e-13
            @test norm(V - R, Inf) > 1e-6
        else
            @test V == R
        end
    end
end

@testset "The right-hand side neither writes u nor remembers a call" begin
    # The contract the integrator relies on, as in TreeWave: a stage that
    # mutated `u`, or a working array carried between calls, would
    # corrupt RK4 silently.
    forest = wave_forest(Float64, Val(2), 2, 2; roots=2)
    p = WaveProblem(forest, Element(Float64, 2))
    u = initial_state(p, testdata(2))
    u0 = copy(u)
    du1, du2 = similar(u), similar(u)
    wave_rhs!(du1, u, p, 0.0)
    wave_rhs!(du2, 2 .* u, p, 0.0)
    wave_rhs!(du2, u, p, 0.0)
    @test u == u0
    @test du1 == du2
end

# The largest change truncation would make to the `v` of state `u`.
function untruncated(u, p)
    w = truncate_state!(copy(u), p)
    return norm(w .- u, Inf)
end

@testset "Truncation is the tensor-product projector, applied in place" begin
    # Guards the per-element kernel: it applies `Π` one dimension at a
    # time as a rank-one update, so a wrong line, a dimension skipped or an
    # element read across its face would leave a projection that is not
    # `Π ⊗ … ⊗ Π`. The dense reference is built from `el.truncate`, the
    # matrix the kernel does not read.
    for (D, P, E, roots) in ((1, 3, 2, 2), (2, 2, 2, 2), (2, 4, 2, 1), (3, 3, 1, 2))
        forest = wave_forest(Float64, Val(D), P, E; roots=roots)
        el = Element(Float64, P)
        p = WaveProblem(forest, el; truncate=false)
        q = WaveProblem(forest, el)
        u = initial_state(p, testdata(D))
        w = truncate_state!(copy(u), q)
        _, Πm, _ = global_operators(el, roots * E, minimum_spacing(forest))
        Π = reduce(kron, fill(Πm, D))
        V = vec(global_array(p.fs, u, 2))
        @test norm(vec(global_array(p.fs, w, 2)) - Π * V, Inf) < 1e-13 * norm(V, Inf)
        @test global_array(p.fs, w, 1) == global_array(p.fs, u, 1)
        @test untruncated(w, q) < 1e-14 * norm(V, Inf)
    end
end

@testset "Every stage and every step has v truncated" begin
    # Guards the truncation itself: the top Legendre mode of `v` is zero
    # in every state the right-hand side is evaluated at and in every
    # step's result — set to zero there, not merely kept at zero by the
    # linearity of the stages. The second half shows that it is the
    # limiter doing it: without the hooks the stages carry the mode.
    for (D, P) in ((1, 3), (2, 2))
        forest = wave_forest(Float64, Val(D), P, 2; roots=2)
        p = WaveProblem(forest, Element(Float64, P))
        u0 = initial_state(p, testdata(D))
        @test untruncated(u0, p) < 1e-13
        dt = oldg_timestep(p; cfl=1 // 2)
        for truncated in (true, false)
            worst = Ref(0.0)
            function rhs!(du, u, p, t)
                worst[] = max(worst[], untruncated(u, p))
                wave_rhs!(du, u, p, t)
                return nothing
            end
            limiters = truncated ?
                       (stage_limiter=truncate_stage!, step_limiter=truncate_stage!) :
                       (;)
            integ = init(IMEXProblem(rhs!, nothing, u0, (0.0, 10dt), p), RK4();
                         dt=dt, limiters...)
            steps = 0.0
            while integ.nstep < integ.nsteps
                step!(integ)
                steps = max(steps, untruncated(integ.u, p))
            end
            if truncated
                @test worst[] < 1e-13
                @test steps < 1e-13
                # `wave_integrator` installs exactly these limiters.
                integ′ = wave_integrator(p, u0, 0.0, 10dt; dt=dt)
                solve!(integ′)
                @test integ′.u == integ.u
            else
                @test worst[] > 1e-6
                @test steps > 1e-6
            end
        end
    end
end

@testset "The stage arithmetic by block owner is bitwise the broadcast" begin
    # Guards `state_partition`: a partition that missed or doubled an
    # entry is refused by IMEXRungeKutta, but one that gave a block to the
    # wrong thread would only be slower; one that combined entries
    # differently would change the bits. Both paths must agree exactly,
    # at whatever thread count the suite runs.
    for (D, P) in ((1, 3), (2, 2), (3, 1))
        forest = wave_forest(Float64, Val(D), P, 2; roots=3)
        p = WaveProblem(forest, Element(Float64, P))
        u0 = initial_state(p, testdata(D))
        part = state_partition(p.fs, u0)
        @test length(part) == Threads.nthreads()
        @test sort(reduce(vcat, collect.(part))) == eachindex(u0)
        dt = oldg_timestep(p; cfl=1 // 2)
        owner = wave_integrator(p, u0, 0.0, 5dt; dt=dt)
        bcast = init(IMEXProblem(wave_rhs!, nothing, u0, (0.0, 5dt), p), RK4(); dt=dt,
                     stage_limiter=truncate_stage!, step_limiter=truncate_stage!)
        solve!(owner)
        solve!(bcast)
        @test owner.u == bcast.u
    end
end

@testset "Energy is conserved up to the integrator's own loss" begin
    # Summation by parts makes the discrete energy an exact invariant of
    # the semi-discrete system, with or without truncation, so the change
    # over a run is RK4's: never a gain, and shrinking at its fifth-order
    # rate (`|R(iy)|² = 1 - y⁶/72 + …` per step, `1/dt` steps) as the step
    # shrinks. A stencil that was not summation by parts would leave a
    # change that does not go away. (The loss itself is not small here:
    # 200 steps of a few modes at `ω dt ≈ 0.2` lose up to 1%.)
    for (D, P, truncate) in ((1, 3, true), (2, 2, true), (2, 3, false))
        losses = map((1, 1 // 2)) do cfl
            forest = wave_forest(Float64, Val(D), P, 2; roots=2)
            p = WaveProblem(forest, Element(Float64, P); truncate=truncate)
            u0 = initial_state(p, testdata(D))
            dt = oldg_timestep(p; cfl=cfl)
            integ = solve!(wave_integrator(p, u0, 0.0, 200dt; dt=dt))
            E0, E1 = wave_energy(p, u0), wave_energy(p, integ.u)
            @test E1 <= E0
            (E0 - E1) / E0
        end
        @test 0 < losses[2] < losses[1]
        # 200 steps both times, so half the time: a factor 2^6 per step.
        @test losses[1] / losses[2] > 2^5
    end
end

@testset "RK4 is stable up to cfl ≈ 5/√D for every order" begin
    # Guards `oldg_timestep`'s scaling `hₑ/(P+1)²`: the largest
    # eigenvalue of the semi-discrete Laplacian, truncated as the limiter
    # truncates it, found by power iteration through the package's own
    # right-hand side and `truncate_state!`, puts RK4's limit on the
    # imaginary axis, `|λ dt| ≤ 2√2`, at the same `cfl` for every `P`. A
    # step that scaled as `hₑ/(P+1)` would let the limit fall as `1/P`.
    for D in (1, 2), P in (1, 2, 4)
        forest = wave_forest(Float64, Val(D), P, 2; roots=2)
        p = WaveProblem(forest, Element(Float64, P))
        x = initial_state(p, (x, v) -> v == 1 ? sinpi(37 * sum(x) + 0.1) : 0.0)
        y = similar(x)
        # `u` and `v` interleave block by block in the state vector.
        u = selectdim(statearray(x, p.fs), D + 1, 1)
        lap = selectdim(statearray(y, p.fs), D + 1, 2)
        λ = 0.0
        for _ in 1:300
            wave_rhs!(y, x, p, 0.0)
            truncate_state!(y, p)
            λ = norm(lap) / norm(u)
            u .= lap ./ norm(lap)
        end
        cflmax = 2 * sqrt(2) / sqrt(λ) / oldg_timestep(p; cfl=1)
        @test 4.8 < cflmax * sqrt(D) < 5.8
    end
end

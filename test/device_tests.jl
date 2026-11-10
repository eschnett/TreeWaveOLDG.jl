# Running on a device: every driver and problem the suite measures with
# runs on a device and reproduces the host run at the same type, to
# rounding. A device that built a different table, read a different ghost
# or took a different step would still run, and only this comparison would
# tell.
#
# The backend is the field sets' (`WaveProblem(…; backend)`), and the
# integrator follows the state: a device state takes IMEXRungeKutta's
# broadcast path, since `state_partition` hands it no partition. A device
# package is deliberately not a dependency of the package or of the tests;
# add one to an environment of your own and name it:
#
#     julia --project=<env> -e 'using Pkg; Pkg.develop(path = ".");
#                                Pkg.add(["Metal", "KernelAbstractions",
#                                         "TreeAMR", "IMEXRungeKutta",
#                                         "StaticArrays", "ForwardDiff"])'
#     TREEWAVEOLDG_TEST_BACKEND=metal julia --project=<env> test/device_tests.jl
#
# Unset, the CPU stands in for the device, which is what CI does: the
# comparisons are then exact by construction, and what is exercised is the
# plumbing, the backend reaching every set and table. See "Threads and
# devices" in `CODE.md`.

using Test
using TreeAMR
using TreeWaveOLDG
# KernelAbstractions through TreeAMR, so that the tests need no dependency of
# their own for it.
using TreeAMR.KernelAbstractions: CPU, Backend, get_backend, supports_float64
using LinearAlgebra: norm

const DEVICE_NAME = lowercase(get(ENV, "TREEWAVEOLDG_TEST_BACKEND", ""))

if DEVICE_NAME == "cuda"
    using CUDA
elseif DEVICE_NAME == "metal"
    using Metal
elseif !isempty(DEVICE_NAME)
    error("TREEWAVEOLDG_TEST_BACKEND must be \"cuda\" or \"metal\", got " *
          "\"$DEVICE_NAME\"")
end

# The device, or the CPU standing in for it.
const DEVICE = let
    device = if DEVICE_NAME == "cuda"
        CUDA.functional() ? CUDABackend() : nothing
    elseif DEVICE_NAME == "metal"
        Metal.functional() ? MetalBackend() : nothing
    end
    if device === nothing && !isempty(DEVICE_NAME)
        @info "TREEWAVEOLDG_TEST_BACKEND=$DEVICE_NAME is not functional here; " *
              "the CPU stands in for it"
    end
    device === nothing ? CPU() : device
end

const DEVICE_LABEL = string(nameof(typeof(DEVICE)))

# `Float64` where the device has it, the type the host suite makes its
# claims at; `Float32` on Metal, which has nothing else, and on CUDA as
# well. On the CPU stand-in one type, since the comparison is exact there.
const DEVICE_TYPES = DEVICE isa CPU ? (Float64,) :
                     supports_float64(DEVICE) ? (Float64, Float32) : (Float32,)

# The agreement asked of a result: a device may contract `a*b + c` into an
# `fma` where the host does not, and its `sin` and `exp` are not Julia's,
# so it differs from the host in the last bits of every node. An error
# norm is a difference of order-one states and carries a relative noise of
# `eps(T)/err`, hence `sqrt(eps(T))`, as in TreeHydro's device tests.
device_close(dev, host, ::Type{T}) where {T} = isapprox(dev, host; rtol=sqrt(eps(T)))

# Two arrays agree in the norm, to the same tolerance.
device_close_arrays(dev, host, ::Type{T}) where {T} =
    norm(Array(dev) .- host) ≤ sqrt(eps(T)) * norm(host)

@testset "on $DEVICE_LABEL" begin
    @testset "the run returns its state on the device ($DEVICE_LABEL)" begin
        # Guards against a problem that silently builds its sets on the
        # host whatever it was asked for.
        T = first(DEVICE_TYPES)
        forest = wave_forest(T, Val(3), 3, 2; roots=2)
        p = WaveProblem(forest, Element(T, 3); backend=DEVICE)
        u = initial_state(p, wave_exact(3, one(T), 1, zero(T)))
        @test get_backend(u) isa typeof(DEVICE)
        @test get_backend(p.spacings) isa typeof(DEVICE)
    end

    for T in DEVICE_TYPES
        @testset "a periodic box's run on the device gives the host's ($T)" begin
            # Guards against a kernel or the integrator's broadcast path
            # computing something else on the device.
            dev = wave_errors(T, Val(3); P=3, E=2, roots=2, backend=DEVICE)
            host = wave_errors(T, Val(3); P=3, E=2, roots=2)
            @test dev.nsteps == host.nsteps
            @test device_close(dev.l2, host.l2, T)
            @test device_close(dev.energy, host.energy, T)
        end

        @testset "a refined box's run on the device gives the host's ($T)" begin
            # Guards against TreeAMR's coarse-fine element exchange or the
            # refined forest's spacings differing on the device.
            dev = wave_errors(T, Val(3); P=3, E=2, roots=4, refined=true,
                              backend=DEVICE)
            host = wave_errors(T, Val(3); P=3, E=2, roots=4, refined=true)
            @test dev.nsteps == host.nsteps
            @test device_close(dev.l2, host.l2, T)
            @test device_close(dev.energy, host.energy, T)
        end

        @testset "the thirteen-patch ball's step on the device is the host's ($T)" begin
            # Its right-hand side, its truncation, its step and its energy.
            # Guards against the truncation tables (per element on the
            # inner shells, factored on the outer ones) or the coefficients
            # failing to reach the device, the radiative ghosts reading the
            # wrong face, and the step being taken from other speeds.
            forest, geo = thirteenpatch_forest(T, 3, 2; roots=(2, 1, 1), Lc=1, Rm=2,
                                               R1=3)
            el = Element(T, 3)
            mk(backend) = WaveProblem(forest, el; geometry=geo, backend=backend,
                                      boundary=RadiativeBoundary())
            pd, ph = mk(DEVICE), mk(CPU())
            uh = T.(sin.(1:statelength(ph.fs)))
            truncate_state!(uh, ph)
            ud = similar(statevector(pd.fs))
            copyto!(ud, uh)
            duh, dud = similar(uh), similar(ud)
            wave_rhs!(duh, uh, ph, zero(T))
            wave_rhs!(dud, ud, pd, zero(T))
            @test device_close_arrays(dud, duh, T)
            wh, wd = copy(duh), copy(dud)
            truncate_state!(wh, ph)
            truncate_state!(wd, pd)
            @test device_close_arrays(wd, wh, T)
            @test oldg_timestep(pd; cfl=T(1//2)) == oldg_timestep(ph; cfl=T(1//2))
            @test device_close(wave_energy(pd, ud), wave_energy(ph, uh), T)
        end

        @testset "the seven-patch plane wave on the device gives the host's ($T)" begin
            # Guards against the Dirichlet hook (`WithPatch`, the shells'
            # positions) or the full geometry's norm differing on the device.
            dev = sevenpatch_errors(T; P=2, E=2, roots=(2, 1), tend=T(1//8),
                                    backend=DEVICE)
            host = sevenpatch_errors(T; P=2, E=2, roots=(2, 1), tend=T(1//8))
            @test dev.nsteps == host.nsteps
            @test device_close(dev.l2, host.l2, T)
            @test device_close(dev.energy, host.energy, T)
        end

        @testset "the radiative sphere on the device gives the host's ($T)" begin
            # A pulse through the thirteen-patch ball's radiative sphere.
            # Guards against the radiative kernels or the boundary's energy
            # term differing on the device.
            kw = (; P=3, E=2, roots=(2, 1, 1), Rm=2, R1=3, tend=T(1//2))
            dev = radiative_errors(T; kw..., backend=DEVICE)
            host = radiative_errors(T; kw...)
            @test dev.nsteps == host.nsteps
            @test device_close(dev.l2, host.l2, T)
            @test device_close(dev.energy, host.energy, T)
        end
    end
end

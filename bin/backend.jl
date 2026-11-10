# `--backend=` and `--type=` for the benchmarks in this directory, as in
# TreeWave's `bin/backend.jl`. The device package is loaded only when asked
# for: neither CUDA nor Metal is a dependency of this package or of
# TreeAMR, so a script run on a device runs against an environment of its
# own that adds one (`bin/symmetry/gpu.sh` builds it). The load is a
# `Core.eval` into `Main`, and the work then runs through `invokelatest`,
# since a script's `main` is already running when the device package adds
# its methods: in `main`'s world every dispatch on the new backend would
# fall through to KernelAbstractions' generic methods.

using TreeWaveOLDG
using TreeWaveOLDG.TreeAMR.KernelAbstractions: Backend, CPU, supports_float64,
                                               synchronize

const BACKENDNAMES = Dict("cuda" => (:CUDA, :CUDABackend),
                          "metal" => (:Metal, :MetalBackend))

# The backend `name` asks for: `"cpu"`, `"cuda"` or `"metal"`.
function resolvebackend(name::AbstractString)
    name == "cpu" && return CPU()
    haskey(BACKENDNAMES, name) ||
        error("unknown --backend=$name; expected cpu, cuda or metal")
    pkg, ctor = BACKENDNAMES[name]
    Base.identify_package(String(pkg)) === nothing && error(
        "--backend=$name needs $pkg in this environment, which is deliberately not " *
        "a dependency of TreeWaveOLDG: run the script against a project that " *
        "`Pkg.develop`s the package and adds $pkg")
    Core.eval(Main, :(using $pkg))
    Core.eval(Main, :($pkg.functional())) || error(
        "$pkg loaded but reports no functional device, so --backend=$name has " *
        "nothing to run on")
    return Core.eval(Main, :($ctor()))
end

# The number type `--type=f64|f32` asks for.
function resolvetype(name::AbstractString)
    name == "f64" && return Float64
    name == "f32" && return Float32
    return error("unknown --type=$name; expected f64 or f32")
end

# `f(backend, T)` on the backend and type the flags ask for, in the latest
# world, after refusing `Float64` on a device without it.
function withbackend(f, name::AbstractString, typename::AbstractString)
    backend = resolvebackend(name)
    T = resolvetype(typename)
    return Base.invokelatest() do
        T === Float64 && !supports_float64(backend) && error(
            "$(nameof(typeof(backend))) has no Float64, so --type=f64 cannot run on " *
            "it; pass --type=f32")
        f(backend, T)
    end
end

# The least of `reps` timed calls of `f` after a warm-up, waiting for the
# device to finish each one, since a launch or a broadcast returns before
# the device has run it.
function best(f, backend, reps)
    f()
    synchronize(backend)
    return minimum(1:reps) do _
        @elapsed begin
            f()
            synchronize(backend)
        end
    end
end

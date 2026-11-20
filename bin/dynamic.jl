# A particle that moves under its forces, INSPIRAL-PLAN.md's M6a
# (INSPIRAL.md, "Validation", step 6): held on M4's circle by a spring
# (`SpringForce`), from the circle's retarded field, the force taking over
# from the circle over `onset`, at the thread count Julia was started
# with:
#
#     julia -t N,0 --project=. bin/dynamic.jl [--hdiv=8] [--levels=5] \
#         [--w0=4] [--order=3] [--tend=8] [--cfl=0.5] [--v=0.05] [--rc=0.5] \
#         [--rx1=24] [--rx2=30] [--onset=2] [--n2=4] [--every=1] \
#         [--source-tol=0.1] [--prescribed] [--backend=cpu|cuda]
#
# The layout, the particle, the window and the data are
# `bin/prescribed.jl`'s circle with `--data=retarded`, so that its runs
# compare with M4's ("Prescribed paths" under "Measured results" in
# `CODE.md`); `--prescribed` runs that circle instead, on the same grid and
# steps. Every `every` steps it prints, as `bin/prescribed.jl` does, the
# remainder's force along the particle's own acceleration (`Fr`, positive
# inward), velocity (`Ft`) and their normal (`Fn`); the particle's radius
# and speed about the circle's centre less the circle's (`dr`, `dv`); and
# for the dynamic particle what the step's last solve did (`iterations`,
# `projections`), the `ä` block of its Jacobian, `cond_aa` and
# `dmh = ‖J_ää − MγI‖/M`, the grid's self-inertia over the mass, and the
# wall-clock seconds since the start. The radial direction is toward the
# circle's centre, which is the circle's `-a/|a|`, so that a report solves
# nothing. `--source-tol` is `Dynamic`'s.

include(joinpath(@__DIR__, "prescribed.jl"))
using LinearAlgebra: I, cond, opnorm

function dynamic_main()
    withbackend((backend, _) -> dynamic_evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function dynamic_evolve(backend)
    hdiv = parse(Int, option("hdiv", "8"))
    levels = parse(Int, option("levels", "5"))
    w0 = parse(Float64, option("w0", "4"))
    order = parse(Int, option("order", "3"))
    tend = parse(Float64, option("tend", "8"))
    cfl = parse(Float64, option("cfl", "0.5"))
    v = parse(Float64, option("v", "0.05"))
    rc = parse(Float64, option("rc", "0.5"))
    rx1 = parse(Float64, option("rx1", "24"))
    rx2 = parse(Float64, option("rx2", "30"))
    onset = parse(Float64, option("onset", "2"))
    n2 = parse(Int, option("n2", "4"))
    every = parse(Int, option("every", "1"))
    source_tol = parse(Float64, option("source-tol", "0.1"))
    prescribed = "--prescribed" in ARGS
    start = time()
    P = 3
    forest, L = box(hdiv, levels, w0)
    wave = WaveProblem(forest, Element(Float64, P); backend=backend)
    c = SVector(L / 2, L / 2, L / 2)
    m = ParticleModel(; Q=sqrt(0.005), ℓ=1e-3, M=1.0)
    w = Window(c, 1.25, 2.5; continuity=P + 1)
    centre = c + OFFSET
    F, circle = spring_circle(m, centre, rc, v)
    motion = prescribed ? circle : Dynamic((F,), circle; onset, n2, source_tol)
    p = InspiralProblem(wave, (m,), w, motion; order)
    retarded = Window(c, rx1, rx2; continuity=4)
    u0 = prescribed ? inspiral_initial_state(p; retarded) :
         dynamic_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt)
    tag = (prescribed ? "circle" : "spring", hdiv, levels, w0, order)
    println(join(("threads", "motion", "hdiv", "levels", "w0", "order", "t", "quantity",
                  "value"), '\t'))
    tr0 = circle.path(0.0)[1]
    emit(tag..., 0.0, "nodes", statelength(p.wave.fs) ÷ 2)
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    emit(tag..., 0.0, "rr", m.Q^2 / 12π * dot(tr0.ȧ, tr0.v / norm(tr0.v)))
    emit(tag..., 0.0, "Omega", v / rc)
    emit(tag..., 0.0, "onset", onset)
    emit(tag..., 0.0, "setup_seconds", time() - start)
    function report(u, t)
        (z, vz), = particle_state(u, p)
        Fg = particle_forces(p, u, t)[1]
        ρ = (centre - z) / norm(centre - z)
        τ = vz / norm(vz)
        n = SVector(ρ[2] * τ[3] - ρ[3] * τ[2], ρ[3] * τ[1] - ρ[1] * τ[3],
                    ρ[1] * τ[2] - ρ[2] * τ[1])
        emit(tag..., t, "Fr", dot(Fg, ρ))
        emit(tag..., t, "Ft", dot(Fg, τ))
        emit(tag..., t, "Fn", dot(Fg, n))
        emit(tag..., t, "dr", norm(z - centre) - rc)
        emit(tag..., t, "dv", norm(vz) - v)
        if !prescribed && motion.scratch.last !== nothing
            l = motion.scratch.last
            emit(tag..., t, "iterations", l.iterations)
            emit(tag..., t, "projections", l.projections)
            J = l.jacobian[7:9, 7:9]
            γ = 1 / sqrt(1 - dot(vz, vz))
            emit(tag..., t, "cond_aa", cond(J))
            emit(tag..., t, "dmh", opnorm(J - m.M * γ * I) / m.M)
        end
        emit(tag..., t, "seconds", time() - start)
        return nothing
    end
    report(integ.u, integ.t)
    while integ.nstep < integ.nsteps
        step!(integ)
        (integ.nstep % every == 0 || integ.nstep == integ.nsteps) || continue
        report(integ.u, integ.t)
    end
    return nothing
end

dynamic_main()

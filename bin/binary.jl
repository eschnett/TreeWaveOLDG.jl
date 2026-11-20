# The Newtonian binary of INSPIRAL.md's "Parameters for a first run",
# INSPIRAL-PLAN.md's M6b ("A short binary run"), at the thread count Julia
# was started with:
#
#     julia -t N,0 --project=. bin/binary.jl [--hdiv=4] [--levels=3] \
#         [--w0=4] [--R1=50] [--hshell=1] [--order=3] [--tend=32] \
#         [--cfl=0.5] [--v=0.2] [--T=30] [--ramp=10] [--n=4] [--onset=2π/Ω] \
#         [--n2=4] [--rx1=100] [--rx2=110] [--every=1] [--jevery=0.125] \
#         [--source-tol=0.1] [--flux=FILE] [--fradii=4,6,8,12] [--prescribed]
#
# Two particles of `M = 1`, `Q = ±√0.005`, `ℓ = 10⁻³`, at separation 1, each
# at speed `v`, on `binary_circular_motion`'s circle about the centre
# `o = (1.23, 0.71, 0.93)·10⁻²`, off the element faces of the orbit's
# plane (`bin/prescribed.jl`'s `OFFSET`), held there by Newton's force
# with the `G` that makes the predictor's force centripetal
# (`newtonian_binary`). The data are the retarded field, out to `rx1`,
# of a past at rest before `−T − ramp` and spun up with the smoothstep of
# order `n` to the orbit at `−T` (`CODE.md`, "Assumed pasts"), and the
# force takes over from that past over `onset`, one orbit by default, with
# `n2` (`Dynamic`). The layout is `bin/prescribed.jl`'s `--ball`
# (`bin/ball.jl`): `levels` nested cubes, the finest of half-width `w0` at
# `hₑ = 1/hdiv`, in the seven-patch ball's cube, shells of radial width
# about `hshell` to a Sommerfeld sphere at `R1`; the window `R₁ = 1.25`,
# `R₂ = 2.5`, about `o`. `--prescribed` runs the past's circle instead, on
# the same grid and steps.
#
# Every `every` steps it prints one row per quantity, `bin/dynamic.jl`'s
# columns:
#
# - the orbit (`orbit_diagnostics`): `d`, `ddot`, `phase`, `omega`,
#   `energy`, the centre of mass's distance from `o` (`com`) and the
#   momentum's norm (`momentum`);
# - per particle `k`, the remainder's force along the inward radius
#   (`Fr<k>`, toward the centre of mass), the velocity (`Ft<k>`) and their
#   normal (`Fn<k>`);
# - per particle, the jets as solved at the step's last stage, at `t`,
#   `a`, `ȧ`, `ä` (`a<k>x`, …,
#   `ad<k>x`, …, `add<k>x`, …), and the predictor's `a⃛` and `a⃜`
#   (`a3<k>x`, …, `a4<k>x`, …), the instabilities in the higher
#   derivatives that self-force schemes have reported showing first there;
#   and `jerkgap<k>`, `|(ä(t) − ä(t − Δt))/Δt − (a⃛(t) + a⃛(t − Δt))/2|`
#   over `|a⃛(t)|`, the solved `ä`'s rate against the predictor's `a⃛`
#   across the step;
# - what the step's last solve did: `iterations`, `projections`,
#   `residual`, and `dmh<k>`, `‖J_ää − MγI‖/M` of particle `k` with the
#   Jacobian it used, which may have been kept; every `jevery` in time the
#   step's first solve finds it afresh (`renew_jacobian!`), and that
#   step's row is `dmhfresh<k>`, with `dmhcross`, the coupling of the two
#   `ä` blocks;
# - the wall-clock seconds since the start.
#
# At `t = 0` also `G`, `K`, the leading-order `ddot0` (`dipole_ddot`),
# `nodes`, `steps`, `dt` and `setup_seconds`. With `--flux=FILE` the flux
# through spheres in the cube about `o` of radii `--fradii`, every
# `--fluxevery` (`bin/waves_io.jl`), with the retarded field's beside it
# only with `--prescribed`.

include(joinpath(@__DIR__, "prescribed.jl"))
using TreeWaveOLDG: lorentz_factor
using LinearAlgebra: I, cross, opnorm

function binary_main()
    withbackend((backend, _) -> binary_evolve(backend), option("backend", "cpu"), "f64")
    return nothing
end

function binary_evolve(backend)
    hdiv = parse(Int, option("hdiv", "4"))
    levels = parse(Int, option("levels", "3"))
    w0 = parse(Float64, option("w0", "4"))
    R1 = parse(Float64, option("R1", "50"))
    hs = option("hshell", "1")
    order = parse(Int, option("order", "3"))
    tend = parse(Float64, option("tend", "32"))
    cfl = parse(Float64, option("cfl", "0.5"))
    v = parse(Float64, option("v", "0.2"))
    d0 = 1.0
    Ω = 2v / d0
    T = parse(Float64, option("T", "30"))
    ramp = parse(Float64, option("ramp", "10"))
    n = parse(Int, option("n", "4"))
    onset = parse(Float64, option("onset", string(2π / Ω)))
    n2 = parse(Int, option("n2", "4"))
    rx1 = parse(Float64, option("rx1", "100"))
    rx2 = parse(Float64, option("rx2", "110"))
    every = parse(Int, option("every", "1"))
    jevery = parse(Float64, option("jevery", "0.125"))
    source_tol = parse(Float64, option("source-tol", "0.1"))
    prescribed = "--prescribed" in ARGS
    start = time()
    P = 3
    wave = ball_wave(hdiv, levels, w0; R1, hshell=parse(Float64, hs), P, backend)
    o = OFFSET
    q = sqrt(0.005)
    ms = (ParticleModel(; Q=q, ℓ=1e-3, M=1.0), ParticleModel(; Q=-q, ℓ=1e-3, M=1.0))
    f, past = newtonian_binary(ms, d0, v; centre=o, past=T, ramp, n)
    motion = prescribed ? past :
             Dynamic((NoForce(), NoForce()), past; interaction=f, onset, n2, source_tol)
    p = InspiralProblem(wave, ms, Window(o, 1.25, 2.5; continuity=P + 1), motion; order)
    retarded = Window(o, rx1, rx2; continuity=4)
    u0 = prescribed ? inspiral_initial_state(p; retarded) :
         dynamic_initial_state(p; retarded)
    dt = oldg_timestep(p.wave; cfl)
    integ = inspiral_integrator(p, u0, 0.0, tend; dt)
    tag = (prescribed ? "circle" : "binary", hdiv, levels, w0, order)
    println(join(("threads", "motion", "hdiv", "levels", "w0", "order", "t", "quantity",
                  "value"), '\t'))
    emit(tag..., 0.0, "G", f.G)
    emit(tag..., 0.0, "K", binding_constant(ms, f))
    emit(tag..., 0.0, "ddot0", dipole_ddot(ms, f, d0))
    emit(tag..., 0.0, "Omega", Ω)
    emit(tag..., 0.0, "onset", onset)
    emit(tag..., 0.0, "nodes", statelength(p.wave.fs) ÷ 2)
    emit(tag..., 0.0, "steps", integ.nsteps)
    emit(tag..., 0.0, "dt", integ.dt)
    emit(tag..., 0.0, "setup_seconds", time() - start)
    fluxes = flux_writer(p, o, prescribed ? retarded : nothing)
    kf = max(1, round(Int, parse(Float64, option("fluxevery", "0.0625")) / integ.dt))
    write_flux(fluxes, p, integ.u, integ.t, prescribed ? retarded : nothing)
    previous = Ref{Any}(nothing)        # the last report's time and jets
    nextfresh = Ref(0.0)
    renewed = Ref(false)                # the step's Jacobian was found afresh
    function report(u, t)
        zv = particle_state(u, p)
        zs, vs = map(first, zv), map(last, zv)
        od = orbit_diagnostics(ms, f, zs, vs)
        for (name, value) in (("d", od.d), ("ddot", od.ddot), ("phase", od.phase),
                              ("omega", od.omega), ("energy", od.energy),
                              ("com", norm(od.centre - o)), ("momentum", norm(od.momentum)))
            emit(tag..., t, name, value)
        end
        Fg = particle_forces(p, u, t)
        for k in 1:2
            ρ = (od.centre - zs[k]) / norm(od.centre - zs[k])
            τ = vs[k] / norm(vs[k])
            emit(tag..., t, "Fr$k", dot(Fg[k], ρ))
            emit(tag..., t, "Ft$k", dot(Fg[k], τ))
            emit(tag..., t, "Fn$k", dot(Fg[k], cross(ρ, τ)))
        end
        jets = prescribed ? past.path(t) :
               (motion.scratch.last === nothing ? past.path(t) : motion.scratch.last.trajs)
        for k in 1:2, (name, s) in (("a", :a), ("ad", :ȧ), ("add", :ä), ("a3", :a⃛),
                                     ("a4", :a⃜))
            x = getfield(jets[k], s)
            for (i, xyz) in enumerate("xyz")
                emit(tag..., t, "$name$k$xyz", x[i])
            end
        end
        if previous[] !== nothing
            t₋, jets₋ = previous[]
            for k in 1:2
                rate = (jets[k].ä - jets₋[k].ä) / (t - t₋)
                gap = norm(rate - (jets[k].a⃛ + jets₋[k].a⃛) / 2) / norm(jets[k].a⃛)
                emit(tag..., t, "jerkgap$k", gap)
            end
        end
        previous[] = (t, jets)
        if !prescribed && motion.scratch.last !== nothing
            l = motion.scratch.last
            emit(tag..., t, "iterations", l.iterations)
            emit(tag..., t, "projections", l.projections)
            emit(tag..., t, "residual", l.residual)
            # A Jacobian found afresh at the step's first stage is kept
            # through its others, as the chord iteration converges with it.
            J = l.jacobian
            fresh = renewed[]
            for (k, blk) in ((1, 7:9), (2, 16:18))
                γ = lorentz_factor(vs[k])
                dmh = opnorm(J[blk, blk] - ms[k].M * γ * I) / ms[k].M
                emit(tag..., t, fresh ? "dmhfresh$k" : "dmh$k", dmh)
            end
            fresh && emit(tag..., t, "dmhcross", opnorm(J[7:9, 16:18]) / ms[1].M)
        end
        emit(tag..., t, "seconds", time() - start)
        return nothing
    end
    report(integ.u, integ.t)
    while integ.nstep < integ.nsteps
        renewed[] = !prescribed && integ.t >= nextfresh[]
        if renewed[]
            renew_jacobian!(motion)
            nextfresh[] += jevery
        end
        step!(integ)
        (integ.nstep % kf == 0 || integ.nstep == integ.nsteps) &&
            write_flux(fluxes, p, integ.u, integ.t, prescribed ? retarded : nothing)
        (integ.nstep % every == 0 || integ.nstep == integ.nsteps) || continue
        report(integ.u, integ.t)
    end
    fluxes === nothing || close(fluxes.io)
    return nothing
end

binary_main()

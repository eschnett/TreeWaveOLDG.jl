# The seven-patch ball in place of the nested boxes' periodic box, for the
# particle drivers' `--ball` (INSPIRAL-PLAN.md, M8c), included by
# `bin/uniform_motion.jl` and `bin/prescribed.jl`.
#
# The ball's cube is the nested boxes' outermost level: `levels` nested
# cubes about the origin, the finest of half-width `w0` at `hₑ = 1/hdiv`,
# each coarser one twice as wide at twice the element width, the coarsest
# the cube itself, of half-width `Lc = 2^(levels-1) w0`, and `2 w0 hdiv`
# elements across every level, a multiple of `4E` (`E` is 4 where that
# fits, else 2), as `nested_cubes` builds them in the cube's logical
# coordinates. The six shells run from the cube to a sphere of radius
# `R1` under Sommerfeld's condition (`RadiativeBoundary()`), in radial
# elements of width about `hshell`, by default twice the cube's root
# element width, so that the cube–shell faces send back what a 2:1
# coarse-fine face does ("The seven-patch ball" under "Measured results"
# in `CODE.md`). The particles' frame is the cube, and the centre is the
# origin ("Particles in the ball" in `CODE.md`).
#
# `ball_layout` returns the numbers: `E`, the cube's roots across `n`, its
# half-width `Lc`, the shells' radial roots `m` and their element width
# `hshell`. `ball_wave` builds the wave problem on that ball, on `backend`.

function ball_layout(hdiv, levels, w0; R1, hshell=nothing)
    across = 2w0 * hdiv
    isinteger(across) || error("2 w0 hdiv = $across elements across a level is not whole")
    across = round(Int, across)
    fits = [e for e in (4, 2) if across % (4e) == 0]
    isempty(fits) && error(
        "2 w0 hdiv = $across elements across a level is not a multiple of 4E for E = 2 " *
        "or 4, so the nested cubes' faces would not fall on block faces; choose another " *
        "w0 or hdiv")
    E = first(fits)
    n = across ÷ E
    Lc = 2^(levels - 1) * w0
    R1 > sqrt(3) * Lc || error("R1 = $R1 must exceed √3 Lc = $(sqrt(3) * Lc)")
    hroot = 2Lc / across
    h = hshell === nothing ? 2hroot : hshell
    m = max(1, ceil(Int, (R1 - Lc) / (h * E)))
    return (; E, n, Lc, m, hshell=(R1 - Lc) / (m * E))
end

function ball_wave(hdiv, levels, w0; R1, hshell=nothing, P=3,
                   backend=TreeWaveOLDG.TreeAMR.KernelAbstractions.CPU())
    (; E, n, Lc, m) = ball_layout(hdiv, levels, w0; R1, hshell)
    refined = levels == 1 ? false : nested_cubes(Val(3), n / 2^levels, levels)
    forest, geo = sevenpatch_forest(Float64, P, E; roots=(n, m), Lc=Lc, R1=R1,
                                    refined=refined)
    return WaveProblem(forest, Element(Float64, P); geometry=geo,
                       boundary=RadiativeBoundary(), backend=backend)
end

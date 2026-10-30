(* The particle's own near-field terms acting on it: the charge-weighted
   averages of orders 1-3 over a particle at rest. INSPIRAL.md, "The near
   field of an accelerated particle" (the self-force of these terms) and
   "Equations of motion"; CODE.md, "The near field".

   Run alone with `wolframscript -file derivations/selfforce.wl`.

   At rest the force of a field f on the particle is -Q ∇̄f = -∫ ρ₀ ∇f,
   and its potential average f̄ = (1/Q) ∫ ρ₀ f enters the mass
   m = M₀ + Q Φ̄ₑ. Per unit charge, Q = 1, so that forces scale as Q² and
   averages as Q.

   Defines:
     selfForce[N]       -∫ ρ₀ ∇Φ⁽ᴺ⁾, a vector in av, addv
     selfPotential[N]   ∫ ρ₀ Φ⁽ᴺ⁾
     cAddot, cA2A       the order-3 force, cAddot ä + cA2A a² a
     cPotential2        the order-2 potential average, cPotential2 a² *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];
needs["nearfield"];

Print["selfforce.wl"];

(* ∫ over the ball r < ℓ of a polynomial in x1, x2, x3: each monomial is
   its angular moment times a radial integral. *)
sphereMoment[p_List] := If[AnyTrue[p, OddQ], 0,
    2 Times @@ (Gamma[(# + 1)/2] & /@ p)/Gamma[(Total[p] + 3)/2]];
ballIntegral[f_] := Total[
    (#[[2]] sphereMoment[#[[1]]] ell^(Total[#[[1]]] + 3)/(Total[#[[1]]] + 3)) & /@
        CoefficientRules[Expand[f], xv]];

rho0x = Expand[rho0in[Sqrt[qq]]];
check["the ball integral of ρ₀ is 1", Simplify[ballIntegral[rho0x]] == 1];

phiIn[nOrder_] := phiIn[nOrder] = Expand[phiN[nOrder] /. inRules];
selfForce[nOrder_] := selfForce[nOrder] =
    Simplify[-(ballIntegral[rho0x D[phiIn[nOrder], #]] & /@ xv)];
selfPotential[nOrder_] := selfPotential[nOrder] =
    Simplify[ballIntegral[rho0x phiIn[nOrder]]];

check["order 1 pushes back with -δm a", zeroQ[selfForce[1] + deltaM av]];
check["order 2 exerts no force", selfForce[2] === {0, 0, 0}];
cAddot = Simplify[Coefficient[selfForce[3][[1]], add1]];
cA2A = Simplify[Coefficient[selfForce[3][[1]], a1^3]];
check["order 3 is cAddot ä + cA2A a² a",
    zeroQ[selfForce[3] - (cAddot addv + cA2A (av . av) av)]];
check["the ä force is -(ä/24π) ∬ρρ′|x - x′|", zeroQ[cAddot + rhoRhoR/(24 Pi)]];

check["orders 1 and 3 have no potential average",
    selfPotential[1] === 0 && selfPotential[3] === 0];
cPotential2 = Simplify[Coefficient[selfPotential[2], a1^2]];
check["order 2's potential average is cPotential2 a²",
    zeroQ[selfPotential[2] - cPotential2 av . av]];

Print[utf8["  δm = "], deltaM, " = ", N[deltaM /. ell -> 1], utf8[" Q²/ℓ"]];
Print[utf8["  order 3: ä coefficient "], cAddot, " = ", N[cAddot /. ell -> 1], utf8[" Q²ℓ"]];
Print[utf8["  order 3: a² a coefficient "], cA2A, " = ", N[cA2A /. ell -> 1], utf8[" Q²ℓ"]];
Print[utf8["  order 2: potential average "], cPotential2, " = ",
    N[cPotential2 /. ell -> 1], utf8[" Q ℓ a²"]];

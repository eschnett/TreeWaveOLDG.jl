(* The lab slice, as far as it is carried ahead of INSPIRAL-PLAN.md's M5:
   the t′² term of the boosted static field Φ_C. INSPIRAL.md, "Moving
   particles and the lab slice"; CODE.md, "The near field" and
   "Prescribed paths".

   Run alone with `wolframscript -file derivations/labslice.wl`.

   A lab point at r = x - z sits at the rest-frame position y = Λ(v) r and
   rest-frame time t′ = -γ v·r = -v·y. In the instantaneous rest frame the
   particle is at rest at t′ = 0 and then accelerates with α′, so its
   static field at t′ is Φ₀(|y - α′t′²/2|) to second order, and the lab
   slice picks up the t′² term

       T₂ = (t′²/2) ∂²_{t′} Φ₀(|y - α′t′²/2|) = -½ (v·y)² (α′·∇Φ₀)
          = -(v·y)² (α′·y) Φ₀′(q),     q = |y|², Φ₀′ = dΦ₀/dq,

   which outside the particle is -(Q/8π) v² (v̂·n̂)² (α′·n̂), of order r⁰.
   It counts as order 1: it holds one acceleration. The rest of the
   re-expansion (the t′ terms of the near field, and the near field's own
   t′² terms) is M5's.

   Here y is xv, α′ is av, and v is vv, all symbols; the Laplacian is in y.
   Asserted: T₂ is that t′² coefficient, inside and outside; its
   Laplacian is

       ∇²T₂ = -[2 v² (α′·y) Φ₀′ + 4 (v·y)(v·α′) Φ₀′
                + (v·y)² (α′·y) (8 Φ₀″ + ρ₀′)],

   which uses ∇²Φ₀ = ρ₀ differentiated in q, so that no term of size
   Q/ℓ³ appears; and its force on the particle at rest.

   Defines:
     tp2Radial          -Φ₀′(q), the radial factor of (v·y)² (α′·y)
     tp2LapRadial[j]    the Laplacian's radial factors of v² (α′·y),
                        (v·y)(v·α′) and (v·y)² (α′·y)
     tp2In, tp2Out      T₂ inside and outside, in q
     cTp2V2, cTp2VA     its force -∫ρ₀∇T₂ = cTp2V2 v² α′ + cTp2VA (v·α′) v *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];
needs["selfforce"];

Print["labslice.wl"];

vv = {v1, v2, v3};
grad[f_] := D[f, #] & /@ xv;

phiX["in"] = Expand[phi0in[Sqrt[qq]]];
phiX["out"] = phi0out[Sqrt[qq]];
rhoX["in"] = Expand[rho0in[Sqrt[qq]]];
rhoX["out"] = 0;

(* Φ₀ and ρ₀ as functions of q, and their q derivatives. *)
phiQ["in"][q_] = Expand[phi0in[Sqrt[q]]];
phiQ["out"][q_] = phi0out[Sqrt[q]];
rhoQ["in"][q_] = Expand[rho0in[Sqrt[q]]];
rhoQ["out"][q_] = 0;
dphi[side_] := D[phiQ[side][q], q];
d2phi[side_] := D[phiQ[side][q], {q, 2}];
drho[side_] := D[rhoQ[side][q], q];

tp2Radial[side_] := -dphi[side];
tp2Of[side_] := (vv . xv)^2 (av . xv) tp2Radial[side] /. q -> qq;

Do[
    With[{shifted = phiX[side] /. Thread[xv -> xv - av tau^2/2]},
        check["T₂ is the t′² term of the displaced Φ₀ " <> side,
            zeroQ[1/2 (vv . xv)^2 (D[shifted, {tau, 2}] /. tau -> 0) - tp2Of[side]]];
        check["T₂ = -½(v·y)²(α′·∇Φ₀) " <> side,
            zeroQ[-1/2 (vv . xv)^2 (av . grad[phiX[side]]) - tp2Of[side]]]],
    {side, {"in", "out"}}];

check["outside, T₂ = -(1/8π) v²(v̂·n̂)²(α′·n̂) per unit charge",
    zeroQ[tp2Of["out"] + 1/(8 Pi) (vv . xv)^2 (av . xv)/qq^(3/2)]];

tp2LapRadial[side_] := {-2 dphi[side], -4 dphi[side],
    -(8 d2phi[side] + drho[side])};
tp2LapBasis = {(vv . vv) (av . xv), (vv . xv) (vv . av), (vv . xv)^2 (av . xv)};

Do[
    check["∇²T₂ in closed form " <> side,
        zeroQ[lap3[tp2Of[side]] - (tp2LapRadial[side] /. q -> qq) . tp2LapBasis]],
    {side, {"in", "out"}}];

tp2In = Expand[(vv . xv)^2 (av . xv) tp2Radial["in"]];
tp2Out = (vv . xv)^2 (av . xv) tp2Radial["out"];

(* Its force on the particle at rest, -∫ ρ₀ ∇T₂, per unit charge squared.
   T₂ is a polynomial inside, so `ballIntegral` of selfforce.wl applies. *)
tp2Force = Simplify[-(ballIntegral[rhoX["in"] D[tp2Of["in"], #]] & /@ xv)];
(* With v along x: α′ along y gives cTp2V2, α′ along x cTp2V2 + cTp2VA. *)
cTp2V2 = Simplify[tp2Force[[2]] /. {v1 -> 1, v2 -> 0, v3 -> 0, a1 -> 0, a2 -> 1, a3 -> 0}];
cTp2VA = Simplify[(tp2Force[[1]] /. {v1 -> 1, v2 -> 0, v3 -> 0, a1 -> 1, a2 -> 0,
    a3 -> 0}) - cTp2V2];
check["T₂'s force is cTp2V2 v² α′ + cTp2VA (v·α′) v",
    zeroQ[tp2Force - (cTp2V2 (vv . vv) av + cTp2VA (vv . av) vv)]];
Print[utf8["  T₂'s force: v² α′ coefficient "], cTp2V2, " = ", N[cTp2V2 /. ell -> 1],
    utf8[" Q²/ℓ, (v·α′) v coefficient "], cTp2VA, " = ", N[cTp2VA /. ell -> 1],
    utf8[" Q²/ℓ"]];
Print[utf8["  relative to δm: "], Simplify[cTp2V2/deltaM], ", ", Simplify[cTp2VA/deltaM]];

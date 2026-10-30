(* The near field of an accelerated extended particle, orders 1-3, and
   the Laplacian of each order from the near-zone hierarchy.
   INSPIRAL.md, "The near field of an accelerated particle"; CODE.md,
   "The near field".

   Run alone with `wolframscript -file derivations/nearfield.wl`. The
   terms are derived here from the retardation series, independently of
   INSPIRAL.md's results, and then compared with them.

   For a particle momentarily at rest at the origin, with r = x - z and
   s = x′ - z, the time-symmetric field is the even-n part of

       Φ = -(1/4π) Σₙ ((-1)ⁿ/n!) ∫ Rⁿ⁻¹ ∂ₜⁿρ(t, x′) d³x′,   R = |x - x′|,

   with ρ(t, x′) = ρ₀(|Λ(v(t)) (s - Δz(t))|), the profile carried along
   the trajectory Δz(t) = ½at² + ⅙ȧt³ + (1/24)ät⁴ and Lorentz-contracted
   along v(t) = Δz′(t). The order N of a term counts derivatives of the
   velocity: a counts 1, ȧ 2, ä 3 (the bookkeeping parameter `eps`).

   Every integral reduces to the radial potentials through

       ∫ R^k ∂^α ρ₀ = ∂^α B_k(|r|)
       ∫ R^k sᵢ F   = rᵢ ∫ R^k F - ∂ᵢ ∫ R^{k+2} F / (k+2)    (any F)

   The second holds because (r - s)ᵢ R^k = ∂_{rᵢ} R^{k+2}/(k+2). The
   radial potentials enter as abstract functions bk[k][q] of q = |r|²,
   so that the terms come out as polynomials in r, a, ä times
   q-derivatives of the B_k, the form the generator emits.

   Defines:
     phiN[N]   order N, in terms of bk
     lapN[N]   its Laplacian from the hierarchy ∇²Φₙ = ∂ₜ²Φₙ₋₂
     inRules, outRules   bk -> the interior and exterior closed forms
   with vectors xv (= r), av (a), adv (ȧ), addv (ä). *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];
needs["profile"];

Print["nearfield.wl"];

xv = {x1, x2, x3};
sv = {s1, s2, s3};
av = {a1, a2, a3};
adv = {ad1, ad2, ad3};
addv = {add1, add2, add3};
qq = xv . xv;
id3 = IdentityMatrix[3];

(* The trajectory, the contraction Λ - I ≈ ½vvᵀ (the next term,
   (3/8)v² vvᵀ, is order 4), and δ with ρ(t, x′) = ρ₀(s + δ). *)
dz = eps av t^2/2 + eps^2 adv t^3/6 + eps^3 addv t^4/24;
vt = D[dz, t];
lam = id3 + 1/2 Outer[Times, vt, vt];
delta = (lam - id3) . sv - lam . dz;

(* ρ₀(s + δ) to third order in eps, with ∂^α ρ₀ written dd[α]. *)
rhoSeries = Expand[Normal[Series[rho @@ (sv + delta), {eps, 0, 3}]] /. {
    Derivative[i_, j_, k_][rho][__] :> dd[i, j, k],
    rho[__] :> dd[0, 0, 0]}];

(* Coefficient of eps^N tⁿ, which is ∂ₜⁿρ / n! at order N. *)
rhoCoeff[nOrder_, n_] := Coefficient[Coefficient[rhoSeries, eps, nOrder], t, n];

(* ∫ R^k s^p ∂^α ρ₀ d³s, through the two identities. *)
integ1[k_, p_List, al_List] := integ1[k, p, al] =
    If[Total[p] == 0,
        D[bk[k][qq], Sequence @@ Transpose[{xv, al}]],
        With[{i = First[FirstPosition[p, _?Positive]]},
        With[{pm = p - UnitVector[3, i]},
            xv[[i]] integ1[k, pm, al] - D[integ1[k + 2, pm, al], xv[[i]]]/(k + 2)]]];

termInteg[k_, term_] := Module[{d, c},
    d = First[Cases[term, dd[__], {0, Infinity}]];
    c = term/d;
    Total[(#[[2]] integ1[k, #[[1]], List @@ d]) & /@ CoefficientRules[c, sv]]];

integ[k_, e_] := With[{ex = Expand[e]},
    Which[ex === 0, 0,
          Head[ex] === Plus, Total[termInteg[k, #] & /@ List @@ ex],
          True, termInteg[k, ex]]];

(* Φ⁽ᴺ⁾ = -(1/4π) Σ_{n even} ∫ Rⁿ⁻¹ [∂ₜⁿρ/n!]_N, and from
   ∇²Rⁿ⁻¹ = n(n-1) Rⁿ⁻³ its Laplacian, which is ∂ₜ² of the n-2 term. *)
phiN[nOrder_] := phiN[nOrder] = Expand[
    Sum[-1/(4 Pi) integ[n - 1, rhoCoeff[nOrder, n]], {n, 2, 2 nOrder, 2}]];
lapN[nOrder_] := lapN[nOrder] = Expand[
    Sum[-n (n - 1)/(4 Pi) integ[n - 3, rhoCoeff[nOrder, n]], {n, 2, 2 nOrder, 2}]];

inRules = Table[bk[k] -> Function[{q}, Evaluate[Expand[bin[k][Sqrt[q]]]]],
    {k, {-1, 1, 3, 5, 7}}];
outRules = Table[bk[k] -> Function[{q}, Evaluate[Expand[bout[k][Sqrt[q]]]]],
    {k, {-1, 1, 3, 5, 7}}];

lap3[f_] := Total[D[f, {#, 2}] & /@ xv];

(* The hierarchy. Inside, both sides are polynomials in r; outside,
   rational in r and |r|. *)
Do[
    check["∇²Φ⁽" <> ToString[nOrder] <> "⁾ = ∂ₜ² of the lower terms, inside",
        Expand[lap3[phiN[nOrder] /. inRules] - (lapN[nOrder] /. inRules)] === 0];
    check["∇²Φ⁽" <> ToString[nOrder] <> "⁾ = ∂ₜ² of the lower terms, outside",
        zeroQ[lap3[phiN[nOrder] /. outRules] - (lapN[nOrder] /. outRules)]],
    {nOrder, 3}];

(* Order 1 against the order-0 field directly: its Laplacian is the
   order-1 part of ∂ₜ²Φ_C, -(a·∇)Φ₀. *)
dirD[c_, f_] := c . (D[f, #] & /@ xv);
check["∇²Φ⁽¹⁾ = -(a·∇)Φ₀",
    Expand[(lapN[1] + dirD[av, bk[-1][qq]] (-1/(4 Pi))) /. inRules] === 0];

(* INSPIRAL.md's results, in the same notation. *)
bB[k_] := bk[k][qq];
inspiral[1] = 1/(8 Pi) dirD[av, bB[1]];
inspiral[2] = -1/(8 Pi) (av . xv) dirD[av, bB[1]] + 1/(96 Pi) dirD[av, dirD[av, bB[3]]];
inspiral[3] = 1/(96 Pi) dirD[addv, bB[3]] + 1/(16 Pi) (av . av) dirD[av, bB[3]] +
    1/(16 Pi) (av . xv) dirD[av, dirD[av, bB[3]]] -
    7/(960 Pi) dirD[av, dirD[av, dirD[av, bB[5]]]];
Do[
    check["order " <> ToString[nOrder] <> " equals INSPIRAL.md's, inside",
        Expand[(phiN[nOrder] - inspiral[nOrder]) /. inRules] === 0];
    check["order " <> ToString[nOrder] <> " equals INSPIRAL.md's, outside",
        zeroQ[(phiN[nOrder] - inspiral[nOrder]) /. outRules]],
    {nOrder, 3}];

check["no ȧ in orders 1-3",
    And @@ Table[FreeQ[phiN[nOrder], ad1 | ad2 | ad3], {nOrder, 3}]];

(* The point limits, ℓ → 0 outside, against INSPIRAL.md's. *)
rN = Sqrt[qq];
nv = xv/rN;
pointRules = outRules /. ell -> 0;
point[nOrder_] := Simplify[phiN[nOrder] /. pointRules, rN > 0];
check["order 1 point limit is (Q/8π) a·n̂",
    zeroQ[point[1] - 1/(8 Pi) av . nv]];
check["order 2 point limit is (Q r/32π)[a² - 3(a·n̂)²]",
    zeroQ[point[2] - rN/(32 Pi) (av . av - 3 (av . nv)^2)]];
check["order 3 point limit is (Q r²/64π)[2(ä·n̂) + 3a²(a·n̂) + 5(a·n̂)³]",
    zeroQ[point[3] - rN^2/(64 Pi) (2 addv . nv + 3 (av . av) (av . nv) + 5 (av . nv)^3)]];

(* Uniform acceleration g along z: ȧ = 0 and ä = -3g²a. The time-symmetric
   field at t = 0 is -(Q/4πr) Σₖ (-g r/2)^k P_k(cos θ). *)
hyperbolicRules = {a1 -> 0, a2 -> 0, a3 -> g, add1 -> 0, add2 -> 0, add3 -> -3 g^3,
    x1 -> rh Sin[th], x2 -> 0, x3 -> rh Cos[th]};
Do[
    check["order " <> ToString[k] <> " is the hyperbolic-motion term k = " <> ToString[k],
        zeroQ[Simplify[(point[k] /. hyperbolicRules) +
            1/(4 Pi rh) (-g rh/2)^k LegendreP[k, Cos[th]], rh > 0 && 0 < th < Pi]]],
    {k, 3}];

(* What the rest-frame terms miss before the lab-slice re-expansion
   (INSPIRAL.md, "Moving particles and the lab slice"). Until M5 the
   particle field is Φ₀ and the near-field terms evaluated at
   y = Λ(v)(x - z) with α′ = γ²Λ(v)a. Along a trajectory through rest,
   its ∂ₜ² at t = 0 reproduces the hierarchy's Laplacians at orders 1 and
   2, but at order 3 it is off by (a·r)²(a·∇Φ₀): ∂ₜ² of the t′² term of
   the re-expansion, ½(v·r)² (-a·∇Φ₀), which vanishes at v = 0 but whose
   second time derivative does not. So at rest the bracket keeps a part
   (a·r)²(a·∇Φ₀), which behaves like r⁰, even with order 3 carried.
   Checked in the point limit at one exact point, with the exact γ and
   Λ along the trajectory. *)
Module[{num, xn, an, adn, addn, fieldAt, zt, vv, at, gam, lamt, yt, alpha,
        psi, d2, rn},
    num = {x1 -> 3/10, x2 -> -1/5, x3 -> 1/2, a1 -> 2/7, a2 -> 1/3, a3 -> -1/2,
        ad1 -> 1/4, ad2 -> -1/6, ad3 -> 1/5, add1 -> -1/3, add2 -> 1/7, add3 -> 2/9};
    {xn, an, adn, addn} = {xv, av, adv, addv} /. num;
    fieldAt[y_, al_] := -1/(4 Pi Sqrt[y . y]) +
        Sum[point[n] /. Join[Thread[av -> al], Thread[xv -> y]], {n, 1, 2}];
    zt = eps an t^2/2 + eps^2 adn t^3/6 + eps^3 addn t^4/24;
    vv = D[zt, t];
    at = D[vv, t];
    gam = 1/Sqrt[1 - vv . vv];
    lamt = id3 + (gam - 1)/(vv . vv) Outer[Times, vv, vv];
    yt = lamt . (xn - zt);
    alpha = gam^2 lamt . at;
    psi = fieldAt[yt, alpha];
    d2 = Normal[Series[Normal[Series[D[psi, {t, 2}], {t, 0, 0}]], {eps, 0, 3}]];
    rn = Sqrt[xn . xn];
    check["at rest, ∂ₜ² of the rest-frame terms gives ∇²Φ⁽¹⁾ and ∇²Φ⁽²⁾",
        And @@ Table[zeroQ[Coefficient[d2, eps, n] - (lapN[n] /. pointRules /. num)],
            {n, 2}]];
    check["at rest, it misses (a·r)²(a·∇Φ₀) at order 3",
        zeroQ[Coefficient[d2, eps, 3] - (lapN[3] /. pointRules /. num)
            - (an . xn)^2 (an . xn)/(4 Pi rn^3)]]];

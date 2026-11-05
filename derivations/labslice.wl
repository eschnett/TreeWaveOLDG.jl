(* The lab slice: the particle field on the lab's time slice, order by
   order. INSPIRAL.md, "Moving particles and the lab slice";
   INSPIRAL-PLAN.md, M5; CODE.md, "The near field" and "Prescribed paths".

   Run alone with `wolframscript -file derivations/labslice.wl`.

   A lab point at r = x - z sits at the rest-frame position y = Λ(v) r and
   rest-frame time t′ = -γ v·r = -v·y. The first part is the t′² term of
   Φ_C that M4 carried ahead of M5. In the instantaneous rest frame the
   particle is at rest at t′ = 0 and then accelerates with α′, so its
   static field at t′ is Φ₀(|y - α′t′²/2|) to second order, and the lab
   slice picks up the t′² term

       T₂ = (t′²/2) ∂²_{t′} Φ₀(|y - α′t′²/2|) = -½ (v·y)² (α′·∇Φ₀)
          = -(v·y)² (α′·y) Φ₀′(q),     q = |y|², Φ₀′ = dΦ₀/dq,

   which outside the particle is -(Q/8π) v² (v̂·n̂)² (α′·n̂), of order r⁰.
   It counts as order 1: it holds one acceleration.

   Here y is xv, α′ is av, and v is vv, all symbols; the Laplacian is in y.
   Asserted: T₂ is that t′² coefficient, inside and outside; its
   Laplacian is

       ∇²T₂ = -[2 v² (α′·y) Φ₀′ + 4 (v·y)(v·α′) Φ₀′
                + (v·y)² (α′·y) (8 Φ₀″ + ρ₀′)],

   which uses ∇²Φ₀ = ρ₀ differentiated in q, so that no term of size
   Q/ℓ³ appears; and its force on the particle at rest.

   The second part is M5's: the whole re-expansion, orders 1-3. It is
   not T₂ and the t′ terms of the rest-frame near field alone, because
   the particle's density ρ₀(|Λ(v)(x - z)|) is rigid on the lab's slices
   (INSPIRAL.md, "Fields and particles"), not on the rest frame's. So the
   rest frame's density history is that of the lab-sliced particle, and
   the field is its time-symmetric field in the rest frame, from the same
   retardation series as nearfield.wl at any t′,

       Φ(t′, y) = -(1/4π) Σ_{n even} Σ_m C(m, n) t′^{m-n} ∫ Rⁿ⁻¹ c_m,
       ρ(t′, s) = Σ_m c_m(s) t′^m,

   evaluated at t′ = -v·y. The rest frame's worldline is
   Z′(τ) = ½α′τ² + ⅙Bτ³ + (1/24)Cτ⁴ in its coordinate time τ, with
   α′ = av, B = adv, C = addv: the rest-frame forms of the lab's
   acceleration and its derivatives (nearfield.wl's a, ȧ, ä, which are
   these). The density at an event P = (t′, s) is ρ₀ of P's distance from
   the worldline in the rest frame of the worldline's event E that is
   simultaneous with P in the lab, n·(P - E) = 0 with the lab's
   4-velocity n ∝ (1, -v), that is

       τ_E = t′ + v·s - v·Z′(τ_E),
       ρ = ρ₀(|s + δ|),   s + δ = Δx + (γ_E - 1)(V̂·Δx)V̂ - γ_E V Δt,

   with V = Z′′(τ_E), Δx = s - Z′(τ_E), Δt = t′ - τ_E. Everything is
   polynomial in v: the lab velocity enters only through the slicing.

   The lab-slice term of order N is what this adds to the rest-frame
   near field phiN[N] of nearfield.wl, Λ_N(y) = Φ⁽ᴺ⁾(-v·y, y) - phiN[N].
   It vanishes at v = 0, and at order 1 it is T₂ plus the field of the
   density ½(v·y)²(α′·∇ρ₀), by which the lab-sliced density differs from
   the rest-frame-sliced one: T₂ alone leaves the bracket a residual
   (v·y)²(α′·y)ρ₀′ inside the particle, which is not small.

   Asserted besides: the density's displacement δ against the lab's own
   formula, at an exact velocity with γ = 5/4; the rest-frame near field
   at t′ = 0 and v = 0; Λ_N = 0 at v = 0; Λ₁ = T₂ + that field; each Λ_N
   is its decomposition into rotation invariants times radial functions
   of q, and its Laplacian its generated closed form, inside and outside;
   and orders 0-3 in the point limit are the series of the exact field of
   uniform acceleration, the Born field, on the tilted slice t′ = -v·y,
   at several points and velocities.

   Defines:
     tp2Radial          -Φ₀′(q), the radial factor of (v·y)² (α′·y)
     tp2LapRadial[j]    the Laplacian's radial factors of v² (α′·y),
                        (v·y)(v·α′) and (v·y)² (α′·y)
     tp2In, tp2Out      T₂ inside and outside, in q
     cTp2V2, cTp2VA     its force -∫ρ₀∇T₂ = cTp2V2 v² α′ + cTp2VA (v·α′) v
     labN[N]            Λ_N in terms of nearfield.wl's bk, in components
     labInvNames        the invariants' names, in the order of the
                        exponent vectors
     labTerms[N], labLapTerms[N]
                        Λ_N and ∇²Λ_N as lists of {exponents, radial
                        factor}, the factor a function of q through bk
     labRadial[f, side] a radial factor, inside, outside or as a point, in q
   and asserts that Λ₁ exerts no force on the particle at rest, where T₂
   alone exerts cTp2V2 v² α′ + cTp2VA (v·α′) v. *)

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

(* ---- M5: the whole re-expansion, orders 1-3. ---- *)

(* The rest frame's worldline, and the event E on it that the lab sees
   simultaneous with (t, s): τ_E = t + v·s - v·Z′(τ_E), iterated to ε³. *)
truncEps[e_] := Normal[Series[e, {eps, 0, 3}]];
labZ[tau_] := eps av tau^2/2 + eps^2 adv tau^3/6 + eps^3 addv tau^4/24;
labV[tau_] := eps av tau + eps^2 adv tau^2/2 + eps^3 addv tau^3/6;
labTau = t + vv . sv;
Do[labTau = Expand[truncEps[t + vv . sv - vv . labZ[labTau]]], {3}];

(* s + δ: P's position in E's rest frame, with γ_E - 1 = V²/2 and
   (γ_E - 1)V̂V̂ᵀ = VVᵀ/2 to ε³. *)
labDelta = Module[{zE = Expand[truncEps[labZ[labTau]]], vE = Expand[truncEps[labV[labTau]]],
        dx, dt},
    dx = sv - zE;
    dt = t - labTau;
    Expand[truncEps[-zE + 1/2 vE (vE . dx) - (1 + vE . vE/2) vE dt]]];

(* Against the lab's own formula: the event P and the worldline, boosted
   to the lab by -v; E where the lab time is P's; and |Λ(v_E)(x_P - z_E)|²
   with the lab's velocity v_E there, at an exact v with γ = 5/4. *)
Module[{vn = {2/5, 1/5, 2/5}, gam = 5/4, lamn, num, tauS, tL, zL, tP, xP, sol, ts, vL, rr,
        gL, lab, ours, sn, tn},
    lamn = IdentityMatrix[3] + (gam - 1)/(vn . vn) Outer[Times, vn, vn];
    num = {a1 -> 2/7, a2 -> -1/3, a3 -> 1/2, ad1 -> 1/5, ad2 -> 1/4, ad3 -> -1/6,
        add1 -> -1/3, add2 -> 1/7, add3 -> 2/9};
    {sn, tn} = {{3/10, -1/5, 1/4}, 1/7};
    tL[tau_] := gam (tau + vn . labZ[tau]) /. num;
    zL[tau_] := lamn . labZ[tau] + gam vn tau /. num;
    tP = gam (tn + vn . sn);
    xP = lamn . sn + gam vn tn;
    sol = Sum[c[j] eps^j, {j, 0, 3}];
    ts = Solve[Thread[CoefficientList[truncEps[tL[sol] - tP], eps] == 0], c /@ Range[0, 3]];
    sol = sol /. First[ts];
    vL = truncEps[(D[zL[tau], tau]/D[tL[tau], tau]) /. tau -> sol];
    rr = truncEps[xP - zL[sol]];
    gL = 1/Sqrt[1 - vL . vL];
    lab = truncEps[rr . rr + gL^2 (vL . rr)^2];
    ours = truncEps[(sv + labDelta) . (sv + labDelta) /. Join[num, Thread[sv -> sn],
        Thread[vv -> vn], {t -> tn}]];
    check["the lab-sliced density's displacement agrees with the lab's formula",
        zeroQ[Expand[lab - ours]]]];

(* ρ₀(s + δ) by Taylor's formula to ε³, with ∂^α ρ₀ written dd[α] as in
   nearfield.wl. *)
labD[k_] := Coefficient[labDelta, eps, k];
taylor1[w_] := Sum[w[[i]] dd @@ UnitVector[3, i], {i, 3}];
taylor2[w1_, w2_] := Sum[w1[[i]] w2[[j]] dd @@ (UnitVector[3, i] + UnitVector[3, j]),
    {i, 3}, {j, 3}];
taylor3[w1_, w2_, w3_] := Sum[w1[[i]] w2[[j]] w3[[k]]
    dd @@ (UnitVector[3, i] + UnitVector[3, j] + UnitVector[3, k]), {i, 3}, {j, 3}, {k, 3}];
labRho[1] = Expand[taylor1[labD[1]]];
labRho[2] = Expand[taylor1[labD[2]] + 1/2 taylor2[labD[1], labD[1]]];
labRho[3] = Expand[taylor1[labD[3]] + taylor2[labD[1], labD[2]] +
    1/6 taylor3[labD[1], labD[1], labD[1]]];

(* The time-symmetric field at rest-frame time tp, and Λ_N. *)
labFull[nOrder_] := labFull[nOrder] = Module[{mx = Exponent[labRho[nOrder], t]},
    Expand[Sum[-1/(4 Pi) Binomial[m, n] tp^(m - n) integ[n - 1, Coefficient[labRho[nOrder], t, m]],
        {m, 0, mx}, {n, 0, m, 2}]]];
labN[nOrder_] := labN[nOrder] = Expand[(labFull[nOrder] /. tp -> -vv . xv) - phiN[nOrder]];

check["at t′ = 0 and v = 0 it is the rest-frame near field",
    And @@ Table[Expand[(labFull[n] /. tp -> 0 /. Thread[vv -> 0]) - phiN[n]] === 0, {n, 3}]];
check["Λ_N vanishes at v = 0",
    And @@ Table[Expand[labN[n] /. Thread[vv -> 0]] === 0, {n, 3}]];
check["Λ₁ is T₂ plus the field of ½(v·y)²(α′·∇ρ₀)",
    Expand[labN[1] - 1/(8 Pi) (vv . xv)^2 (av . grad[bk[-1][qq]]) +
        1/(4 Pi) integ[-1, Expand[1/2 (vv . sv)^2 taylor1[av]]]] === 0];

(* The radial potentials that Λ_N uses, inside, outside and as a point. *)
labKs = Union[Cases[Table[labN[n], {n, 3}], bk[k_] :> k, Infinity, Heads -> True]];
labRules["in"] = Table[bk[k] -> Function[{q}, Evaluate[Expand[bin[k][Sqrt[q]]]]], {k, labKs}];
labRules["out"] = Table[bk[k] -> Function[{q}, Evaluate[Expand[bout[k][Sqrt[q]]]]], {k, labKs}];
labRules["point"] = {bk[k_] :> Function[{q}, q^(k/2)]};

(* The invariants: four linear in y, six constant. Each counts powers of
   v, α′, B and C; the weight of α′, B, C is 1, 2, 3, and order N is
   weight N. With at most one B or C at order 3 no Gram relation holds
   among them, so the decomposition is unique. *)
labInvNames = {"vy", "ay", "by", "cy", "vv", "va", "aa", "bv", "ab", "cv"};
labInvSyms = Symbol["inv" <> #] & /@ labInvNames;
labInvValues = {vv . xv, av . xv, adv . xv, addv . xv, vv . vv, vv . av, av . av,
    vv . adv, av . adv, vv . addv};
labInvDegs = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}, {2,0,0,0}, {1,1,0,0},
    {0,2,0,0}, {1,0,1,0}, {0,1,1,0}, {1,0,0,1}};
labYLinear = 4;
labMonomial[e_List] := Times @@ (labInvSyms^e);
labMonomialValue[e_List] := Times @@ (labInvValues^e);

(* All exponent vectors with the given degrees in v, α′, B and C. *)
labEnum[target_, i_] := If[i > Length[labInvDegs], If[target == {0,0,0,0}, {{}}, {}],
    Flatten[Table[Prepend[#, e] & /@ labEnum[target - e labInvDegs[[i]], i + 1],
        {e, 0, Min[If[labInvDegs[[i, #]] > 0, Floor[target[[#]]/labInvDegs[[i, #]]],
            Infinity] & /@ Range[4]]}], 1]];

(* Decomposition by homogeneous block, matched at y = (0, 0, X) and v in
   the xz-plane, with the radial potentials' derivatives at X² as
   symbols bkd[k, j]. *)
labSpecial = {x1 -> 0, x2 -> 0, x3 -> X, v1 -> V1, v2 -> 0, v3 -> V3,
    a1 -> A1, a2 -> A2, a3 -> A3, ad1 -> B1, ad2 -> B2, ad3 -> B3,
    add1 -> C1, add2 -> C2, add3 -> C3};
labCompVars = {V1, V3, A1, A2, A3, B1, B2, B3, C1, C2, C3};
labScale = {v1 -> lv v1, v2 -> lv v2, v3 -> lv v3, a1 -> la a1, a2 -> la a2, a3 -> la a3,
    ad1 -> lb ad1, ad2 -> lb ad2, ad3 -> lb ad3, add1 -> lc add1, add2 -> lc add2,
    add3 -> lc add3};
toBkd[e_] := e /. Derivative[j_][bk[k_]][_] :> bkd[k, j] /. bk[k_][_] :> bkd[k, 0];
fromBkd[e_] := e /. bkd[k_, j_] :> D[bk[k][q], {q, j}];
labDecompose[expr_] := Module[{blocks, res = {}},
    blocks = CoefficientRules[Expand[expr /. labScale], {lv, la, lb, lc}];
    Do[Module[{es, ms, fs, eqs, sol},
        es = labEnum[blk[[1]], 1];
        ms = labMonomialValue /@ es;
        fs = Array[ff, Length[ms]];
        eqs = Values[CoefficientRules[Expand[toBkd[(blk[[2]] - fs . ms) /. labSpecial]],
            labCompVars]];
        sol = Solve[Thread[eqs == 0], fs];
        If[Length[sol] != 1, Print["FAILED: no unique decomposition of block ", blk[[1]]];
            Exit[1]];
        If[!FreeQ[fs /. First[sol], X], Print["FAILED: a radial factor depends on X"]; Exit[1]];
        res = Join[res, Transpose[{es, fromBkd[Expand[fs /. First[sol]]]}]]],
        {blk, blocks}];
    Select[res, Expand[#[[2]]] =!= 0 &]];

(* ∇² of f(q) M, M a monomial in the invariants: (4q f″ + 6f′ + 4k f′) M
   with k the degree of M in y, plus f Σᵢⱼ ∂ᵢ∂ⱼM (uᵢ·uⱼ), the uᵢ the
   vectors v, α′, B, C of the y-linear invariants. *)
labDots = Module[{u = {"v", "a", "b", "c"}, tab},
    tab = <|{"v", "v"} -> invvv, {"v", "a"} -> invva, {"a", "a"} -> invaa,
        {"v", "b"} -> invbv, {"a", "b"} -> invab, {"v", "c"} -> invcv|>;
    Table[Lookup[tab, Key[Sort[{u[[i]], u[[j]]}, OrderedQ[{#1, #2} /. {"v" -> 1, "a" -> 2,
        "b" -> 3, "c" -> 4}] &]], forbidden], {i, 4}, {j, 4}]];
labLaplacian[terms_] := Module[{total},
    total = Total[Function[{e, f}, Module[{m = labMonomial[e], k = Total[e[[;; labYLinear]]]},
        (4 q D[f, {q, 2}] + 6 D[f, q] + 4 k D[f, q]) m +
        f Sum[D[m, labInvSyms[[i]], labInvSyms[[j]]] labDots[[i, j]],
            {i, labYLinear}, {j, labYLinear}]]] @@@ terms];
    If[!FreeQ[total, forbidden], Print["FAILED: a dot product of weight above 3"]; Exit[1]];
    Select[{#[[1]], Expand[#[[2]]]} & /@ CoefficientRules[Expand[total], labInvSyms],
        #[[2]] =!= 0 &]];

(* A term whose radial factor vanishes inside and outside once the B_k
   are their closed forms is dropped: one of the order-2 Laplacian's. *)
labDropZero[terms_] := Select[terms,
    !(Expand[#[[2]] /. labRules["in"]] === 0 && Expand[#[[2]] /. labRules["out"]] === 0) &];
Do[labTerms[n] = labDropZero[labDecompose[labN[n]]];
    labLapTerms[n] = labDropZero[labLaplacian[labTerms[n]]];
    Print["  order ", n, ": ", Length[labTerms[n]], " terms, Laplacian ",
        Length[labLapTerms[n]]], {n, 3}];

(* A radial factor inside, outside or as a point, in q; and a sum of
   terms at a point, with its invariants' values. *)
labRadial[f_, side_] := Expand[f /. labRules[side]];
labSum[terms_, side_, vals_] := Total[Function[{e, f},
    (labRadial[f, side] /. q -> (xv . xv /. vals) /. vals) (labMonomialValue[e] /. vals)] @@@ terms];

SeedRandom[5];
labRandom[n_, s_] := Thread[n -> RandomInteger[{-9, 9}, 3]/s];
(* A point inside the particle of size 3/2, or outside it. *)
labPoint["in"] := labRandom[xv, 20];
labPoint["out"] := Thread[xv -> RandomSample[{5/2 + RandomInteger[{0, 9}]/4,
    RandomInteger[{-9, 9}]/4, RandomInteger[{-9, 9}]/4}]];
Do[With[{vals = Join[labPoint[side], labRandom[vv, 25],
        labRandom[av, 7], labRandom[adv, 7], labRandom[addv, 7], {ell -> 3/2}]},
    If[side == "in" && (xv . xv /. vals) >= 9/4 || side == "out" && (xv . xv /. vals) <= 9/4,
        Print["FAILED: sample point on the wrong side"]; Exit[1]];
    Do[check["Λ" <> ToString[n] <> " is its decomposition, " <> side <> ", sample " <> ToString[i],
        zeroQ[RootReduce[labSum[labTerms[n], side, vals] - (labN[n] /. labRules[side] /. vals)]]],
        {n, 3}]],
    {side, {"in", "out"}}, {i, 2}];

(* The Laplacians against ∇² of the decomposed terms, with the vectors
   numbers and y symbolic. *)
Do[With[{vecs = Join[labRandom[vv, 25], labRandom[av, 7], labRandom[adv, 7],
        labRandom[addv, 7], {ell -> 3/2}], pt = labPoint[side]},
    Do[With[{field = Total[Function[{e, f}, (labRadial[f, side] /. q -> qq)
            (labMonomialValue[e] /. vecs)] @@@ labTerms[n]] /. vecs},
        check["∇²Λ" <> ToString[n] <> " in closed form, " <> side,
            zeroQ[RootReduce[(lap3[field] /. pt) -
                labSum[labLapTerms[n], side, Join[vecs, pt]]]]]],
        {n, 3}]],
    {side, {"in", "out"}}];

(* Uniform acceleration g along z with the particle at rest at the
   origin at t = 0: the Born field Φ = -1/4πD with
   D² = g²S²/4 - (X² - t²), S = X² - t² + ρ⊥² + g⁻², X = z + 1/g, which is
   time-symmetric. Its rest-frame worldline has α′ = g ẑ, B = 0 and
   C = -3g³ ẑ. On the tilted slice t = -v·y, at y → εy, its series in ε
   is orders 0-3 of the lab slice in the point limit, term by term. *)
bornEps[t_, y_] := Module[{X = y[[3]] + 1/g, S},
    S = X^2 - t^2 + y[[1]]^2 + y[[2]]^2 + 1/g^2;
    -1/(4 Pi eps Sqrt[Expand[(g^2 S^2/4 - (X^2 - t^2))/eps^2]])];
Module[{hyp = {a1 -> 0, a2 -> 0, a3 -> g, ad1 -> 0, ad2 -> 0, ad3 -> 0, add1 -> 0,
        add2 -> 0, add3 -> -3 g^3}, gn = 7/5},
    Do[Module[{vals = Join[labRandom[xv, 10], labRandom[vv, 25]], yn, vn, ser},
        {yn, vn} = {xv, vv} /. vals;
        ser = Series[bornEps[-eps vn . yn, eps yn] /. g -> gn, {eps, 0, 2}];
        check["the Born field on the slice t′ = -v·y, sample " <> ToString[i] <> ", order 0",
            zeroQ[RootReduce[SeriesCoefficient[ser, -1] + 1/(4 Pi Sqrt[yn . yn])]]];
        Do[check["the Born field on the slice t′ = -v·y, sample " <> ToString[i] <>
                ", order " <> ToString[n],
            zeroQ[RootReduce[SeriesCoefficient[ser, n - 1] -
                ((phiN[n] /. labRules["point"] /. hyp /. vals) +
                 labSum[labTerms[n], "point", Join[hyp, vals]]) /. g -> gn]]],
            {n, 3}]],
        {i, 3}]];

(* Λ₁'s force on the particle at rest, -∫ρ₀∇Λ₁, per unit charge squared,
   as for T₂: none. The field of the slicing density cancels T₂'s. *)
labForce1 = Module[{inside = Total[Function[{e, f},
        (labRadial[f, "in"] /. q -> qq) labMonomialValue[e]] @@@ labTerms[1]]},
    Simplify[-(ballIntegral[rhoX["in"] D[inside, #]] & /@ xv)]];
check["Λ₁ exerts no force on the particle at rest", labForce1 === {0, 0, 0}];

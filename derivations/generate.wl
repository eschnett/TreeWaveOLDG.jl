(* Writes src/particle_forms.jl from the derivations: the radial functions
   of the profile, the near-field terms of orders 1-3 and their
   Laplacians, and the self-force coefficients. CODE.md, "The particle"
   and "The near field".

       wolframscript -file derivations/generate.wl [output]

   writes `output`, by default src/particle_forms.jl. The test suite runs
   it with a temporary output and compares the result with the checked-in
   file byte for byte (CLAUDE.md, "Commands").

   Every radial function is emitted as a scale times a polynomial: inside
   the particle ℓ^p times a polynomial in σ² = q/ℓ², outside r^p times a
   polynomial in ℓ²/q, with q = |y|², r = √q, and the branch on σ < 1.
   Each near-field term is a sum over a basis of rotation invariants
   (a·y, a², ä·y) times such radial functions. Coefficients are exact
   rationals, written `_q(T, n, d)` and rounded once to the model's type
   `T`, and π is `T(π)`, so that the code is generic in its argument
   types and works on dual numbers. *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];
needs["selfforce"];

Print["generate.wl"];

output = If[Length[$ScriptCommandLine] >= 2, $ScriptCommandLine[[2]],
    FileNameJoin[{$derivationsDirectory, "..", "src", "particle_forms.jl"}]];

(* Numbers. *)
jlInt[n_Integer] := (
    If[Abs[n] >= 2^53, Print["FAILED: ", n, " is not exact in Float64"]; Exit[1]];
    ToString[n]);
jlNum[c_Integer] := "T(" <> jlInt[c] <> ")";
jlNum[c_Rational] := "_q(T, " <> jlInt[Numerator[c]] <> ", " <> jlInt[Denominator[c]] <> ")";
jlTuple[cs_List] := "(" <> StringRiffle[jlNum /@ cs, ", "] <> If[Length[cs] == 1, ",)", ")"];
jlPow[x_String, 0] := "";
jlPow[x_String, 1] := x <> " * ";
jlPow[x_String, p_Integer] := x <> "^" <> If[p < 0, "(" <> ToString[p] <> ")", ToString[p]] <> " * ";

(* f, an expression in q and ell, as (power of π, p, coefficients, basis)
   with f = π^(-k) ℓ^p Σ cᵢ xⁱ inside, x = σ² or 1 - σ², or
   f = π^(-k) r^p Σ cᵢ (ℓ²/q)^i outside. Inside, the basis is the one in
   which the polynomial is better conditioned on 0 ≤ σ ≤ 1: the largest
   ratio of Σ |cᵢ xⁱ| to |Σ cᵢ xⁱ|, which is what rounding the terms
   costs. For ρ₀ = c(1 - σ²)⁴ that is 1 against 10¹² near σ = 1. *)
powers[x_, n_] := Prepend[x^Range[1, n - 1], 1];
conditionNumber[cs_List, xs_List] := Max[Table[With[{p = Abs[cs . powers[x, Length[cs]]]},
    If[p == 0, 1, (Abs[cs] . powers[Abs[x], Length[cs]])/p]], {x, xs}]];
grid = Table[i/100, {i, 0, 99}];
piPower[f_] := Which[FreeQ[Expand[f], Pi], 0, FreeQ[Expand[Pi f], Pi], 1,
    True, Print["FAILED: not a rational multiple of 1/π: ", f]; Exit[1]];
interiorForm[f_] := Module[{k = piPower[f], e, p, poly},
    e = Expand[Pi^k f /. q -> s2 ell^2];
    p = Exponent[e, ell];
    poly = Expand[e/ell^p];
    If[!FreeQ[poly, ell] || !PolynomialQ[poly, s2],
        Print["FAILED: not ℓ^p times a polynomial in σ²: ", f]; Exit[1]];
    With[{cs = CoefficientList[poly, s2], cw = CoefficientList[Expand[poly /. s2 -> 1 - w], w]},
        If[conditionNumber[cw, (1 - #^2) & /@ grid] < conditionNumber[cs, grid^2],
            {k, p, cw, "1 - q / ℓ^2"}, {k, p, cs, "q / ℓ^2"}]]];
exteriorForm[f_] := Module[{k = piPower[f], e, p, poly},
    e = Expand[PowerExpand[Pi^k f /. q -> rr^2]];
    p = Exponent[e /. ell -> 0, rr];
    poly = Expand[PowerExpand[(e/rr^p) /. ell -> Sqrt[u] rr]];
    If[!FreeQ[poly, rr] || !PolynomialQ[poly, u],
        Print["FAILED: not r^p times a polynomial in ℓ²/q: ", f]; Exit[1]];
    {k, p, CoefficientList[poly, u]}];

piSuffix[0] = "";
piSuffix[1] = " / T(π)";

(* A radial function of q = |y|² and ℓ, from its interior and exterior
   expressions in q. *)
radialFunction[name_String, doc_String, fin_, fout_] := Module[{ki, pi, ci, xi, ke, pe, ce},
    {ki, pi, ci, xi} = interiorForm[fin];
    StringJoin[
        "# ", doc, "\n",
        "@inline function ", name, "(q, ℓ)\n",
        "    T = typeof(ℓ)\n",
        "    if q < ℓ^2\n",
        "        return ", jlPow["ℓ", pi], "evalpoly(", xi, ", ", jlTuple[ci], ")",
            piSuffix[ki], "\n",
        "    else\n",
        If[Expand[fout] === 0,
            "        return zero(q / ℓ^2)\n",
            {ke, pe, ce} = exteriorForm[fout];
            StringJoin[
                If[pe == 0, "", "        r = sqrt(q)\n"],
                "        return ", jlPow["r", pe], "evalpoly(ℓ^2 / q, ", jlTuple[ce], ")",
                    piSuffix[ke], "\n"]],
        "    end\n",
        "end\n"]];

(* The profile, per unit charge. *)
qIn[f_] := Expand[f /. r -> Sqrt[q]];
profileCode = StringJoin[
    radialFunction["_rho0", "ρ₀/Q: the profile, zero outside the particle.",
        qIn[rho0in[r]], 0], "\n",
    radialFunction["_phi0", "Φ₀/Q: its static field, ∇²Φ₀ = ρ₀.",
        qIn[phi0in[r]], qIn[phi0out[r]]], "\n",
    Table[radialFunction["_B" <> If[k < 0, "m" <> ToString[-k], ToString[k]],
        "B_" <> ToString[k] <> "/Q = ∫ ρ₀(x′) |y - x′|^" <> ToString[k] <> " d³x′ / Q.",
        qIn[bin[k][r]], qIn[bout[k][r]]] <> "\n", {k, ks}]];

(* The near field. Each order is a sum over a basis of invariants times
   radial functions, found by matching at y = (0, 0, X), a = (A1, 0, A3),
   ä = (D1, 0, D3), and then checked against the full expression. *)
invariants = <|"ay" -> av . xv, "aa" -> av . av, "äy" -> addv . xv|>;
basis[1] = {{"ay"}};
basis[2] = {{"aa"}, {"ay", "ay"}};
basis[3] = {{"äy"}, {"aa", "ay"}, {"ay", "ay", "ay"}};
basisValue[b_] := Times @@ (invariants /@ b);
special = {x1 -> 0, x2 -> 0, x3 -> X, a1 -> A1, a2 -> 0, a3 -> A3,
    add1 -> D1, add2 -> 0, add3 -> D3};

decompose[expr_, nOrder_] := Module[{fs, ansatz, sol, rad},
    fs = Array[ff, Length[basis[nOrder]]];
    ansatz = fs . (basisValue /@ basis[nOrder]);
    sol = Solve[Thread[Values[CoefficientRules[
        Expand[(expr - ansatz) /. special], {A1, A3, D1, D3}]] == 0], fs];
    If[Length[sol] != 1, Print["FAILED: no unique decomposition"]; Exit[1]];
    rad = Simplify[PowerExpand[(fs /. First[sol]) /. X -> Sqrt[q]], q > 0];
    check["order " <> ToString[nOrder] <> " is its basis decomposition",
        zeroQ[(rad /. q -> qq) . (basisValue /@ basis[nOrder]) - expr]];
    rad];

nearFieldCode[nOrder_] := Module[{fin, fout, lin, lout, nb = Length[basis[nOrder]],
        fname, lname, term},
    fin = decompose[Expand[phiN[nOrder] /. inRules], nOrder];
    fout = decompose[phiN[nOrder] /. outRules, nOrder];
    lin = decompose[Expand[lapN[nOrder] /. inRules], nOrder];
    lout = decompose[lapN[nOrder] /. outRules, nOrder];
    fname[j_] := "_F" <> ToString[nOrder] <> "_" <> ToString[j];
    lname[j_] := "_L" <> ToString[nOrder] <> "_" <> ToString[j];
    term[name_, j_] := StringRiffle[basis[nOrder][[j]], " * "] <> " * " <> name[j] <> "(q, ℓ)";
    StringJoin[
        Table[radialFunction[fname[j], "Order " <> ToString[nOrder] <>
            ", radial factor of " <> StringRiffle[basis[nOrder][[j]], "·"] <> ".",
            fin[[j]], fout[[j]]] <> "\n", {j, nb}],
        Table[radialFunction[lname[j], "Its Laplacian, radial factor of " <>
            StringRiffle[basis[nOrder][[j]], "·"] <> ".",
            lin[[j]], lout[[j]]] <> "\n", {j, nb}],
        "@inline function near_field_term(::Val{", ToString[nOrder], "}, Q, ℓ, y, a, ä)\n",
        "    q, ay, aa, äy = _invariants(y, a, ä)\n",
        "    return Q * (", StringRiffle[Table[term[fname, j], {j, nb}], " +\n" <>
            StringRepeat[" ", 16]], ")\n",
        "end\n\n",
        "@inline function near_field_laplacian(::Val{", ToString[nOrder], "}, Q, ℓ, y, a, ä)\n",
        "    q, ay, aa, äy = _invariants(y, a, ä)\n",
        "    return Q * (", StringRiffle[Table[term[lname, j], {j, nb}], " +\n" <>
            StringRepeat[" ", 16]], ")\n",
        "end\n"]];

(* A constant times Q^qp ℓ^lp / π. *)
constantCode[name_String, doc_String, c_, qp_Integer] := Module[{k = piPower[c], e, p},
    e = Expand[Pi^k c];
    p = Exponent[e, ell];
    StringJoin[
        "# ", doc, "\n",
        "function ", name, "(Q, ℓ)\n",
        "    T = typeof(ℓ)\n",
        "    return ", jlNum[e/ell^p], " * Q^", ToString[qp],
            If[p == 0, "", " * ℓ^" <> If[p < 0, "(" <> ToString[p] <> ")", ToString[p]]],
            piSuffix[k], "\n",
        "end\n"]];

code = StringJoin[
    "# Generated by derivations/generate.wl; do not edit. Regenerate with\n",
    "#\n",
    "#     $TREEWAVEOLDG_WOLFRAMSCRIPT -file derivations/generate.wl\n",
    "#\n",
    "# (CLAUDE.md, \"Commands\"). The derivations are derivations/profile.wl,\n",
    "# nearfield.wl and selfforce.wl; CODE.md, \"The particle\" and \"The near\n",
    "# field\", say what these functions are. Radial functions take q = |y|²\n",
    "# and the particle size ℓ, and are per unit charge.\n",
    "\n",
    "# An exact rational, rounded once to the model's type.\n",
    "@inline _q(::Type{T}, n, d) where {T} = T(n) / T(d)\n",
    "\n",
    "@inline function _invariants(y, a, ä)\n",
    "    q = y[1]^2 + y[2]^2 + y[3]^2\n",
    "    ay = a[1] * y[1] + a[2] * y[2] + a[3] * y[3]\n",
    "    aa = a[1]^2 + a[2]^2 + a[3]^2\n",
    "    äy = ä[1] * y[1] + ä[2] * y[2] + ä[3] * y[3]\n",
    "    return q, ay, aa, äy\n",
    "end\n",
    "\n",
    profileCode,
    Table[nearFieldCode[n] <> "\n", {n, 3}],
    constantCode["self_inertia", "δm: the self-inertia, the order-1 force -δm a.",
        deltaM, 2], "\n",
    constantCode["rho_rho_r", "∬ ρ₀ρ₀′ |x - x′|.", rhoRhoR, 2], "\n",
    constantCode["self_force_addot",
        "The order-3 force is self_force_addot · ä + self_force_a2a · a² a.",
        cAddot, 2], "\n",
    constantCode["self_force_a2a", "See `self_force_addot`.", cA2A, 2], "\n",
    constantCode["self_potential_a2",
        "Order 2's potential average: Φ̄⁽²⁾ = self_potential_a2 · a².",
        cPotential2, 1]];

If[!StringQ[code], Print["FAILED: the code did not evaluate to a string"]; Exit[1]];

(* Strings were read as Latin-1; write the bytes back as they came. *)
Export[output, utf8[code], "Text", CharacterEncoding -> "UTF-8"];
Print["  wrote ", output];

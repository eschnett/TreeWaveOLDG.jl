(* Writes src/particle_forms.jl from the derivations: the radial functions
   of the profile, the near-field terms of orders 1-3 and their
   Laplacians, the lab slice of orders 1-3 and its Laplacians, and the
   self-force coefficients. CODE.md, "The particle",
   "The near field" and "Prescribed paths".

       wolframscript -file derivations/generate.wl [output]

   writes `output`, by default src/particle_forms.jl. The test suite runs
   it with a temporary output and compares the result with the checked-in
   file byte for byte (CLAUDE.md, "Commands").

   Every radial function is emitted as a scale times a polynomial: inside
   the particle ℓ^p times a polynomial in σ² = q/ℓ², outside r^p times a
   polynomial in ℓ²/q, with q = |y|², r = √q, and the branch on σ < 1.
   Each near-field term is a sum over a basis of rotation invariants
   (a·y, a², ä·y) times such radial functions, emitted as a function of
   q and those invariants, which the source evaluates on dual numbers. Coefficients are exact
   rationals, written `_q(T, n, d)` and rounded once to the model's type
   `T`, and π is `T(π)`, so that the code is generic in its argument
   types and works on dual numbers. *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];
needs["labslice"];

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
(* p is the degree of homogeneity in ℓ and r, which is the point limit's
   power of r where there is one, and is still defined for a finite-size
   term that vanishes in the point limit. *)
exteriorForm[f_] := Module[{k = piPower[f], e, p, poly, degs},
    e = Expand[PowerExpand[Pi^k f /. q -> rr^2]];
    degs = Union[Exponent[#, rr] + Exponent[#, ell] & /@ If[Head[e] === Plus, List @@ e, {e}]];
    If[Length[degs] != 1, Print["FAILED: not homogeneous in ℓ and r: ", f]; Exit[1]];
    p = First[degs];
    poly = Expand[PowerExpand[(e/rr^p) /. ell -> Sqrt[u] rr]];
    If[!FreeQ[poly, rr] || !PolynomialQ[poly, u],
        Print["FAILED: not r^p times a polynomial in ℓ²/q: ", f]; Exit[1]];
    {k, p, CoefficientList[poly, u]}];

piSuffix[0] = "";
piSuffix[1] = " / T(π)";

(* A radial function of q = |y|² and ℓ, from its interior and exterior
   expressions in q. `inline` is the prefix of its definition: the lab
   slice's are not inlined, since a BigFloat Hessian of order 3 inlined
   takes minutes to compile (CODE.md, "The near field"). *)
radialFunction[name_String, doc_String, fin_, fout_, inline_String: "@inline "] :=
    Module[{ki, pi, ci, xi, ke, pe, ce},
    StringJoin[
        "# ", doc, "\n",
        inline, "function ", name, "(q, ℓ)\n",
        "    T = typeof(ℓ)\n",
        "    if q < ℓ^2\n",
        If[Expand[fin] === 0,
            "        return zero(q / ℓ^2)\n",
            {ki, pi, ci, xi} = interiorForm[fin];
            StringJoin["        return ", jlPow["ℓ", pi], "evalpoly(", xi, ", ", jlTuple[ci], ")",
                piSuffix[ki], "\n"]],
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
        "@inline function _near_field_term(::Val{", ToString[nOrder], "}, Q, ℓ, q, ay, aa, äy)\n",
        "    return Q * (", StringRiffle[Table[term[fname, j], {j, nb}], " +\n" <>
            StringRepeat[" ", 16]], ")\n",
        "end\n\n",
        "@inline function _near_field_laplacian(::Val{", ToString[nOrder], "}, Q, ℓ, q, ay, aa, äy)\n",
        "    return Q * (", StringRiffle[Table[term[lname, j], {j, nb}], " +\n" <>
            StringRepeat[" ", 16]], ")\n",
        "end\n"]];

(* The lab slice of order N (labslice.wl): a sum over monomials in the
   invariants vy = v·y, ay = α′·y, by = B·y, cy = C·y, vv = v², va = v·α′,
   aa = |α′|², bv = v·B, ab = α′·B and cv = v·C times radial functions,
   and its Laplacian in y likewise. *)
labArgs = "Q, ℓ, q, vy, ay, by, cy, vv, va, aa, bv, ab, cv";
labMonomialCode[e_List] := StringJoin[MapThread[
    Which[#2 == 0, "", #2 == 1, #1 <> " * ", True, #1 <> "^" <> ToString[#2] <> " * "] &,
    {labInvNames, e}]];
labMonomialDoc[e_List] := StringRiffle[DeleteCases[MapThread[
    Which[#2 == 0, Nothing, #2 == 1, #1, True, #1 <> "^" <> ToString[#2]] &,
    {labInvNames, e}], Nothing], "·"];
(* Each order and kind (the term, its Laplacian) is evaluated in two
   steps, since the source evaluates it some 10⁷ times per step for the
   same particle. Its terms are gathered by their monomial in the four
   invariants linear in y, their power of ℓ or r and of π; the six others
   do not depend on the point, so each group's radial polynomial with
   them folded into its coefficients is one polynomial, `_lab_coefficients`
   of them, computed once per particle and step (with its time
   derivatives, for the bracket), and `_lab_evaluate` sums one polynomial
   per group: 36 of them where there are 98 terms. It reads coefficient i
   as `_coefficient(c, i, d)` (src/particle.jl), which for the bracket
   lifts it from its rates in place, with the directions d of the further
   partials, so that the table is not copied. The branch on σ < 1 is
   taken once, with σ² and 1 - σ² or r and ℓ²/q formed once. Inside, a
   group's polynomial is in σ² or in 1 - σ², whichever is better
   conditioned for the worst of its terms. The sum is a running one: Julia
   parses a + b + … as one call of + with as many arguments, which
   allocates beyond about 30. Neither is inlined, since a BigFloat Hessian
   of order 3 inlined takes minutes to compile (CODE.md, "The near
   field"). *)
interiorBoth[f_] := Module[{k = piPower[f], e, p, poly},
    e = Expand[Pi^k f /. q -> s2 ell^2];
    p = Exponent[e, ell];
    poly = Expand[e/ell^p];
    If[!FreeQ[poly, ell] || !PolynomialQ[poly, s2],
        Print["FAILED: not ℓ^p times a polynomial in σ²: ", f]; Exit[1]];
    {k, p, CoefficientList[poly, s2], CoefficientList[Expand[poly /. s2 -> 1 - w], w]}];
labGroups[terms_, side_] := Module[{items},
    items = Flatten[Table[With[{f = labRadial[t[[2]], side]},
        If[Expand[f] === 0, {},
            If[side === "in",
                Module[{k, p, cs, cw}, {k, p, cs, cw} = interiorBoth[f];
                    {<|"lin" -> t[[1, ;; 4]], "con" -> t[[1, 5 ;;]], "k" -> k, "p" -> p,
                       "cs" -> cs, "cw" -> cw|>}],
                Module[{k, p, cs}, {k, p, cs} = exteriorForm[f];
                    {<|"lin" -> t[[1, ;; 4]], "con" -> t[[1, 5 ;;]], "k" -> k, "p" -> p,
                       "cs" -> cs|>}]]]], {t, terms}], 1];
    GatherBy[items, {#lin, #p, #k} &]];
labConMonomial[con_List] := StringTrim[labMonomialCode[Join[{0, 0, 0, 0}, con]], " * "];
pad[cs_List, n_] := PadRight[cs, n];
(* A group's merged coefficients, as Julia expressions in the constants,
   and the variable its polynomial is in. *)
labMerged[group_, side_] := Module[{basis, key, n, coeff},
    basis = If[side === "out", "u",
        If[Max[conditionNumber[#cw, (1 - #^2) & /@ grid] & /@ group] <
           Max[conditionNumber[#cs, grid^2] & /@ group], "w", "σ²"]];
    key = If[basis === "w", "cw", "cs"];
    n = Max[Length[#[key]] & /@ group];
    coeff[i_] := With[{parts = Select[{pad[#[key], n][[i]], labConMonomial[#con]} & /@ group,
            #[[1]] =!= 0 &]},
        If[parts === {}, "zero(T)",
            StringRiffle[If[#[[2]] === "", jlNum[#[[1]]], jlNum[#[[1]]] <> " * " <> #[[2]]] & /@
                parts, " + "]]];
    {basis, Table[coeff[i], {i, n}]}];
(* The powers a branch's groups multiply, each formed once per point and
   named by its exponent (vy³, r⁻⁵), rather than once per group: on the
   bracket's nested dual numbers the order-3 term's monomials cost more
   than its polynomials (CODE.md, "The near field"). x^k is x^(k/2)
   squared for even k and x^(k-1)·x for odd k. The superscripts are
   separate strings, since the script's strings are read as Latin-1 and
   each is several of its characters. *)
superscript[k_Integer] := StringJoin[If[k < 0, "⁻", ""],
    {"⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"}[[
        ToExpression /@ Characters[ToString[Abs[k]]] + 1]]];
powerName[x_String, 1] := x;
powerName[x_String, k_Integer] := x <> superscript[k];
powerLines[x_String, base_String, kmax_Integer, sign_Integer] := StringJoin[Table[
    "        " <> powerName[x, sign k] <> " = " <>
        If[EvenQ[k], powerName[x, sign k/2] <> " * " <> powerName[x, sign k/2],
            powerName[x, sign (k - 1)] <> " * " <> powerName[x, sign]] <> "\n",
    {k, 2, kmax}]];
labPowerLines[groups_List, side_] := Module[{lin, ps, x = If[side === "in", "ℓ", "r"]},
    lin = Max /@ Transpose[#[[1]]["lin"] & /@ groups];
    ps = #[[1]]["p"] & /@ groups;
    StringJoin[
        MapThread[powerLines[#1, #1, #2, 1] &, {labInvNames[[;; 4]], lin}],
        powerLines[x, x, Max[ps, 1], 1],
        If[Min[ps] < 0,
            "        " <> powerName[x, -1] <> " = 1 / " <> x <> "\n" <>
                powerLines[x, x, -Min[ps], -1], ""]]];
labMonomialPowers[lin_List, p_Integer, x_String] := StringJoin[
    MapThread[If[#2 == 0, "", powerName[#1, #2] <> " * "] &, {labInvNames[[;; 4]], lin}],
    If[p == 0, "", powerName[x, p] <> " * "]];
labKindCode[kind_String, nOrder_, terms_] := Module[{groups, merged, offset = 0, exprs = {},
        lines, branch},
    branch[side_] := StringJoin[labPowerLines[groups[side], side], Table[
        Module[{g = groups[side][[j]], b, cs, idx, first = groups[side][[j, 1]]},
            {b, cs} = labMerged[g, side];
            idx = offset + Range[Length[cs]];
            offset += Length[cs];
            exprs = Join[exprs, cs];
            "        " <> If[first["k"] == 0, "s₀", "s₁"] <> " += " <>
                labMonomialPowers[first["lin"], first["p"], If[side === "in", "ℓ", "r"]] <>
                "evalpoly(" <> b <> ", (" <>
                StringRiffle["_coefficient(c, " <> ToString[#] <> ", d)" & /@ idx, ", "] <>
                If[Length[idx] == 1, ",", ""] <> "))\n"],
        {j, Length[groups[side]]}]];
    groups["in"] = labGroups[terms, "in"];
    groups["out"] = labGroups[terms, "out"];
    lines = StringJoin[
        "function _lab_evaluate(::Val{", ToString[nOrder], "}, ::Val{:", kind,
            "}, Q, ℓ, q, vy, ay, by, cy, c, d=())\n",
        "    T = typeof(ℓ)\n",
        "    s₀ = s₁ = zero(q * _coefficient(c, 1, d))\n",
        "    if q < ℓ^2\n",
        "        σ² = q / ℓ^2\n",
        "        w = 1 - σ²\n",
        branch["in"],
        "    else\n",
        "        r = sqrt(q)\n",
        "        u = ℓ^2 / q\n",
        branch["out"],
        "    end\n",
        "    return Q * (s₀ + s₁ / T(π))\n",
        "end\n"];
    {Length[groups["in"]] + Length[groups["out"]], Length[exprs], StringJoin[
        "# The lab slice of order ", ToString[nOrder], ", ", kind, ": ", ToString[Length[terms]],
            " terms in ", ToString[Length[groups["in"]]], " groups inside and ",
            ToString[Length[groups["out"]]], " outside, ", ToString[Length[exprs]],
            " coefficients.\n",
        "function _lab_coefficients(::Val{", ToString[nOrder], "}, ::Val{:", kind,
            "}, ℓ, vv, va, aa, bv, ab, cv)\n",
        "    T = typeof(ℓ)\n",
        "    return SVector(", StringRiffle[exprs, ",\n" <> StringRepeat[" ", 19]], ")\n",
        "end\n\n",
        lines]}];
labOrderCode[nOrder_] := Module[{t, l},
    t = labKindCode["term", nOrder, labTerms[nOrder]];
    l = labKindCode["laplacian", nOrder, labLapTerms[nOrder]];
    labLengths[nOrder] = {t[[2]], l[[2]]};
    StringJoin[t[[3]], "\n", l[[3]]]];
labWrappers = StringJoin[
    "# The lab slice of order N and its Laplacian as functions of all the\n",
    "# invariants, in one step.\n",
    "@inline _lab_slice_term(::Val{N}, Q, ℓ, q, vy, ay, by, cy, vv, va, aa, bv, ab, cv) where {N} =\n",
    "    _lab_evaluate(Val(N), Val(:term), Q, ℓ, q, vy, ay, by, cy,\n",
    "                  _lab_coefficients(Val(N), Val(:term), ℓ, vv, va, aa, bv, ab, cv))\n",
    "@inline _lab_slice_laplacian(::Val{N}, Q, ℓ, q, vy, ay, by, cy, vv, va, aa, bv, ab, cv) where {N} =\n",
    "    _lab_evaluate(Val(N), Val(:laplacian), Q, ℓ, q, vy, ay, by, cy,\n",
    "                  _lab_coefficients(Val(N), Val(:laplacian), ℓ, vv, va, aa, bv, ab, cv))\n"];
labTypesCode := StringJoin[
    "# The types of the coefficients per order: the term's with their first\n",
    "# two time derivatives, the Laplacian's alone.\n",
    "const LabTermJets{T} = Tuple{", StringRiffle[Table["NTuple{3,SVector{" <>
        ToString[labLengths[n][[1]]] <> ",T}}", {n, 3}], ", "], "}\n",
    "const LabLaplacianCoefficients{T} = Tuple{", StringRiffle[Table["SVector{" <>
        ToString[labLengths[n][[2]]] <> ",T}", {n, 3}], ", "], "}\n"];

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
    "# nearfield.wl, selfforce.wl and labslice.wl; CODE.md, \"The particle\",\n",
    "# \"The near field\" and \"Prescribed paths\" say what these functions are.\n",
    "# Radial functions take q = |y|² and the particle size ℓ, and are per\n",
    "# unit charge.\n",
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
    "# A near-field term and its Laplacian at y, through the invariants, on\n",
    "# which the generated forms below depend.\n",
    "@inline near_field_term(::Val{N}, Q, ℓ, y, a, ä) where {N} =\n",
    "    _near_field_term(Val(N), Q, ℓ, _invariants(y, a, ä)...)\n",
    "@inline near_field_laplacian(::Val{N}, Q, ℓ, y, a, ä) where {N} =\n",
    "    _near_field_laplacian(Val(N), Q, ℓ, _invariants(y, a, ä)...)\n",
    "\n",
    profileCode,
    Table[nearFieldCode[n] <> "\n", {n, 3}],
    "@inline function _lab_invariants(y, v, a, b, c)\n",
    "    return (y[1]^2 + y[2]^2 + y[3]^2, dot(v, y), dot(a, y), dot(b, y), dot(c, y),\n",
    "            dot(v, v), dot(v, a), dot(a, a), dot(v, b), dot(a, b), dot(v, c))\n",
    "end\n",
    "\n",
    "# The lab slice of order N and its Laplacian at the rest-frame position y,\n",
    "# for lab velocity v and the rest-frame acceleration α′ and its derivatives\n",
    "# B and C, through the invariants.\n",
    "@inline lab_slice_term(::Val{N}, Q, ℓ, y, v, α′, B, C) where {N} =\n",
    "    _lab_slice_term(Val(N), Q, ℓ, _lab_invariants(y, v, α′, B, C)...)\n",
    "@inline lab_slice_laplacian(::Val{N}, Q, ℓ, y, v, α′, B, C) where {N} =\n",
    "    _lab_slice_laplacian(Val(N), Q, ℓ, _lab_invariants(y, v, α′, B, C)...)\n",
    "\n",
    Table[labOrderCode[n] <> "\n", {n, 3}],
    labWrappers, "\n",
    labTypesCode, "\n",
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

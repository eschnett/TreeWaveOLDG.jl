(* The particle profile: the density ρ₀, its static field Φ₀, its moments,
   the radial potentials B_k = ∫ ρ₀(x′) |x - x′|^k d³x′ inside and outside
   the particle, and the two double integrals that set its self-force.
   INSPIRAL.md, "The particle profile"; CODE.md, "The particle".

   Run alone with `wolframscript -file derivations/profile.wl`. Every
   closed form is derived here and then asserted against its definition
   and against the values INSPIRAL.md gives.

   Defines, all per unit charge Q (so that ∫ρ₀ = 1):
     rho0in[r]        ρ₀ inside, r < ℓ (zero outside)
     phi0in[r], phi0out[r]
     bin[k][r], bout[k][r]   B_k for k = -1, 1, 3, 5, with B₋₁ = -4πΦ₀
     moment[j]        ⟨s^{2j}⟩
     deltaM, rhoRhoR  δm and ∬ρρ′|x - x′|
   with ℓ the symbol `ell`. *)

If[!ValueQ[$derivationsDirectory],
    $derivationsDirectory = DirectoryName[$InputFileName]];
Get[FileNameJoin[{$derivationsDirectory, "common.wl"}]];

Print["profile.wl"];

$Assumptions = ell > 0 && r > 0 && s > 0;

rho0in[r_] := 3465/(512 Pi ell^3) (1 - r^2/ell^2)^4;

(* The charge is Q, here 1. *)
check["∫ρ₀ = 1", Integrate[4 Pi r^2 rho0in[r], {r, 0, ell}] == 1];

(* Φ₀ from Gauss's law, ∇²Φ₀ = ρ₀: Φ₀′ = (enclosed charge)/4πr², and
   Φ₀ = -1/4πr outside. *)
phi0out[r_] := -1/(4 Pi r);
enclosed[r_] = Integrate[4 Pi s^2 rho0in[s], {s, 0, r}];
phi0in[r_] = Expand[phi0out[ell] - Integrate[enclosed[s]/(4 Pi s^2), {s, r, ell}]];

radialLaplacian[f_, r_] := D[r^2 D[f, r], r]/r^2;

check["∇²Φ₀ = ρ₀ inside", zeroQ[radialLaplacian[phi0in[r], r] - rho0in[r]]];
check["Φ₀ matches INSPIRAL.md inside",
    zeroQ[phi0in[r] - 1/(1024 Pi ell) (-693 + 1155 sig^2 - 1386 sig^4 + 990 sig^6
        - 385 sig^8 + 63 sig^10) /. sig -> r/ell]];

(* The moments ⟨s^{2j}⟩ = ∫ ρ₀ s^{2j}. *)
moment[j_] := moment[j] = Integrate[4 Pi r^(2 + 2 j) rho0in[r], {r, 0, ell}];
check["⟨s²⟩ = 3ℓ²/13", moment[1] == 3 ell^2/13];
check["⟨s⁴⟩ = ℓ⁴/13", moment[2] == ell^4/13];
check["⟨s⁶⟩ = 7ℓ⁶/221", moment[3] == 7 ell^6/221];

(* B_k from the shell average of |x - x′|^k over a sphere of radius s,
       2π/(r s (k+2)) [(r + s)^{k+2} - |r - s|^{k+2}],
   split at s = r inside the particle. *)
shell[k_, r_, s_, rs_] := 2 Pi/(r s (k + 2)) ((r + s)^(k + 2) - rs^(k + 2));
bin[k_] := bin[k] = Function[{rr}, Evaluate[Expand[
    Integrate[rho0in[s] s^2 shell[k, r, s, r - s], {s, 0, r}] +
    Integrate[rho0in[s] s^2 shell[k, r, s, s - r], {s, r, ell}] /. r -> rr]]];
bout[k_] := bout[k] = Function[{rr}, Evaluate[Expand[
    Integrate[rho0in[s] s^2 shell[k, r, s, r - s], {s, 0, ell}] /. r -> rr]]];
ks = {-1, 1, 3, 5};

Do[
    check["B_" <> ToString[k] <> " is an even polynomial inside",
        PolynomialQ[bin[k][r], r] &&
        Expand[bin[k][r] - (bin[k][r] /. r -> -r)] === 0],
    {k, ks}];

check["B₋₁ = -4πΦ₀ inside", zeroQ[bin[-1][r] + 4 Pi phi0in[r]]];
check["B₋₁ = -4πΦ₀ outside", zeroQ[bout[-1][r] + 4 Pi phi0out[r]]];

(* ∇²B_k = k(k+1) B_{k-2}, and ∇²B₋₁ = -4πρ₀. *)
check["∇²B₋₁ = -4πρ₀ inside",
    zeroQ[radialLaplacian[bin[-1][r], r] + 4 Pi rho0in[r]]];
check["∇²B₋₁ = 0 outside", zeroQ[radialLaplacian[bout[-1][r], r]]];
Do[
    check["∇²B_" <> ToString[k] <> " = k(k+1) B_{k-2} inside",
        zeroQ[radialLaplacian[bin[k][r], r] - k (k + 1) bin[k - 2][r]]];
    check["∇²B_" <> ToString[k] <> " = k(k+1) B_{k-2} outside",
        zeroQ[radialLaplacian[bout[k][r], r] - k (k + 1) bout[k - 2][r]]],
    {k, {1, 3, 5}}];

(* The exterior forms are finite sums of the moments. *)
{m2, m4, m6} = moment /@ {1, 2, 3};
check["B₁ outside matches INSPIRAL.md", zeroQ[bout[1][r] - (r + m2/(3 r))]];
check["B₃ outside matches INSPIRAL.md",
    zeroQ[bout[3][r] - (r^3 + 2 r m2 + m4/(5 r))]];
check["B₅ outside matches INSPIRAL.md",
    zeroQ[bout[5][r] - (r^5 + 5 r^3 m2 + 3 r m4 + m6/(7 r))]];
check["B₁′ inside matches INSPIRAL.md",
    zeroQ[D[bin[1][r], r] - (231 sig/128 - 231 sig^3/128 + 99 sig^5/64 - 55 sig^7/64
        + 35 sig^9/128 - 63 sig^11/1664 /. sig -> r/ell)]];

(* Inside and outside agree at sig = 1 to as many derivatives as the
   profile's smoothness gives: ρ₀ is C³, and ∇²B₋₁ ∝ ρ₀, so B₋₁ is C⁵,
   and each B_k two orders smoother than B_{k-2}. *)
smoothness[k_] := Module[{j = 0},
    While[zeroQ[D[bin[k][r] - bout[k][r], {r, j}] /. r -> ell], j++];
    j - 1];
Do[
    check["B_" <> ToString[k] <> " is C^" <> ToString[5 + (k + 1)] <> " at sig = 1",
        smoothness[k] == 5 + (k + 1)],
    {k, ks}];

(* The two double integrals. *)
deltaM = Simplify[-1/3 Integrate[4 Pi r^2 rho0in[r] phi0in[r], {r, 0, ell}]];
check["δm = -⅓∫ρ₀Φ₀ = 693/4199πℓ", deltaM == 693/(4199 Pi ell)];
check["δm = (1/12π)∬ρρ′/|x - x′|",
    zeroQ[deltaM - 1/(12 Pi) Integrate[4 Pi r^2 rho0in[r] bin[-1][r], {r, 0, ell}]]];
rhoRhoR = Simplify[Integrate[4 Pi r^2 rho0in[r] bin[1][r], {r, 0, ell}]];
check["∬ρρ′|x - x′| = 60984ℓ/96577", rhoRhoR == 60984 ell/96577];

Print[utf8["  B₃ inside: "], Collect[bin[3][r] /. r -> sig ell, sig, Simplify]];
Print[utf8["  B₅ inside: "], Collect[bin[5][r] /. r -> sig ell, sig, Simplify]];

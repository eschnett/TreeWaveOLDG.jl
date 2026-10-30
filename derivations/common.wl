(* What every derivation script loads first: the assertion that makes a
   wrong derivation fail loudly, and the guard that loads a script once.

   A script asserts its identities with `check[name, condition]`, which
   prints the name and exits with status 1 if the condition is not
   `True`, so that `wolframscript -file` and the generator stop on the
   first wrong identity. *)

check[name_String, cond_] :=
    If[TrueQ[cond],
        Print["  ok: ", utf8[name]],
        Print["FAILED: ", utf8[name]]; Print["  condition: ", cond]; Exit[1]];

(* wolframscript reads a script's bytes as Latin-1, so a string literal
   holding UTF-8 comes out as one character per byte. `utf8` decodes it
   again, and leaves a string that was read correctly alone. *)
utf8[s_String] := With[{c = ToCharacterCode[s]},
    If[Max[c, 0] < 256, Quiet@Check[FromCharacterCode[c, "UTF-8"], s], s]];

(* `zeroQ[e]` asks whether an expression, or each entry of a list, is
   identically zero, through `Together` (for rational functions, exact)
   and then `FullSimplify`. *)
zeroQ[e_List] := AllTrue[e, zeroQ];
zeroQ[e_] := Module[{z = Together[Expand[e]]},
    TrueQ[z === 0] || TrueQ[FullSimplify[z] === 0]];

(* Load another script of this directory once. *)
needs[name_String] := If[!MemberQ[$loadedDerivations, name],
    AppendTo[$loadedDerivations, name];
    Get[FileNameJoin[{$derivationsDirectory, name <> ".wl"}]]];

If[!ValueQ[$loadedDerivations], $loadedDerivations = {}];

# Two inspiralling scalar particles on OLDG

A numerical method for two small scalar charges that start on a circular
orbit and spiral in by radiating, in three dimensions, built on this
package's OLDG scheme. It follows Vretinaris, Schnetter and Krishnan,
"Numerical self-force calculations for scalar particles, formulated in
the lab frame" ([arXiv:2606.06487](https://arxiv.org/abs/2606.06487);
"the particle paper" below): a particle of small but finite size `ℓ`, a
field split into an analytic particle field and a numerical remainder,
and an effective source through which an accelerating particle acts on
the remainder.

The particles have a known, physical size `ℓ`, fixed in advance and much
smaller than the mesh spacing: `0 < ℓ ≪ h`. Finite-size effects are
therefore physics, not discretization artefacts. A variant with `ℓ`
proportional to `h`, which avoids nearly singular integrals at the price
of a less clean convergence, is kept as [an
alternative](#alternative-particle-size-proportional-to-h).

**Status.** A design; nothing here is implemented. Everything is
**(proposed)** unless marked **(open)**. `CODE.md` describes the scheme
this builds on, and its section names are used below without links.

**Contents**

- [Summary](#summary)
- [Assumptions](#assumptions)
- [The physical model](#the-physical-model)
  - [Fields and particles](#fields-and-particles)
  - [The particle profile](#the-particle-profile)
  - [Equations of motion](#equations-of-motion)
  - [The near field of an accelerated particle](#the-near-field-of-an-accelerated-particle)
  - [Why the near field must be analytic](#why-the-near-field-must-be-analytic)
- [The split](#the-split)
  - [The particle field](#the-particle-field)
  - [The acceleration and its derivatives](#the-acceleration-and-its-derivatives)
  - [The remainder equations](#the-remainder-equations)
  - [The window](#the-window)
  - [Moving particles and the lab slice](#moving-particles-and-the-lab-slice)
- [The discretization](#the-discretization)
  - [The remainder on OLDG](#the-remainder-on-oldg)
  - [The transfer operator](#the-transfer-operator)
  - [Quadrature](#quadrature)
  - [The right-hand side](#the-right-hand-side)
  - [Time integration](#time-integration)
- [Initial data: a quasi-circular orbit](#initial-data-a-quasi-circular-orbit)
- [Diagnostics](#diagnostics)
- [Expected accuracy](#expected-accuracy)
- [Validation](#validation)
- [Parameters for a first run](#parameters-for-a-first-run)
- [Mapping onto the code](#mapping-onto-the-code)
- [Alternative: particle size proportional to h](#alternative-particle-size-proportional-to-h)
  - [What changes](#what-changes)
  - [The acceleration predictor](#the-acceleration-predictor)
  - [Sources and truncation](#sources-and-truncation)
  - [Accuracy along a series](#accuracy-along-a-series)
  - [A binary bound by its own field](#a-binary-bound-by-its-own-field)
- [Open questions](#open-questions)

## Summary

The method in one screen:

1. **State.** The remainder field `(Φ_r, Π_r)` on the OLDG mesh as the
   package's `(u, v)`, and each particle's position `z_a` and velocity
   `v_a`, `a = 1, 2`. Nothing else: no acceleration, no history.
2. **Particle size.** `ℓ` is the particle's true size, fixed, with
   `0 < ℓ ≪ h`. The grid never resolves the particle, so everything on
   the scale `ℓ` is analytic.
3. **Particle field.** Each particle's field is the time-symmetric field
   of the extended particle (half retarded plus half advanced), expanded
   in the distance from the particle to third order. Its terms are the
   boosted static field and terms in the acceleration `a` and its second
   derivative `ä`. In flat spacetime this is Detweiler and Whiting's
   singular field, smooth inside the particle. A fixed window cuts it off
   far outside the binary.
4. **Acceleration.** The particle field carries the particle's *actual*
   acceleration, which the state determines. Its first and second time
   derivatives, which the field's time derivative and the effective
   source need, come from the time derivatives of the equation of
   motion, each at most a 3×3 linear solve per particle. Only the
   third-order term's `ä`, and weak couplings to still higher
   derivatives, come from an analytic two-body predictor.
5. **Remainder equations.** `∂ₜΦ_r = Π_r` and `∂ₜΠ_r = ∇²Φ_r - S`, where
   `Π_p` is the particle field's exact time derivative. The effective
   source `S` is the residual of the particle field's wave equation along
   the actual motion. The split is exact.
6. **One transfer operator.** Particle to grid is the element-wise `L²`
   load vector divided by the Gauss–Lobatto weights. Grid to particle is
   its adjoint, the exact integral of a particle kernel against the
   reconstructed discontinuous field. Near the particle the integrals use
   a sinh-transformed Gauss rule at the scale `ℓ`.
7. **Particles.** Special-relativistic equations of motion with
   charge-weighted field averages. The bare mass is the physical
   (dressed) mass minus the analytic self-inertia `δm(ℓ)`. The particle's
   own acceleration terms push on it analytically.
8. **Binding.** An analytic, instantaneous Newtonian force holds the
   binary together, and small scalar charges make it radiate. A binary
   bound only by its own scalar field cannot have `ℓ ≪ h` at useful
   resolutions.
9. **Time.** RK4 with the package's limiter, the package's time step,
   and a right-hand side that is a pure function of the state. A warm-up
   on a prescribed circular orbit lets the remainder settle before the
   particles are released.

Two things depart from the particle paper. Both are three-dimensional
effects that its one-dimensional tests do not meet. In three dimensions
the near field of an accelerated particle holds a dipole `≈ Q a·n̂/8π`
that varies on the scale `ℓ` and pushes back on the particle with a
self-inertia `-δm a`, `δm ∝ Q²/ℓ`. At `ℓ ≪ h` the grid can produce only
a fraction of order `ℓ/hₑ` of that force. So the acceleration terms must
be in the analytic particle field, with the actual acceleration (see [Why
the near field must be analytic](#why-the-near-field-must-be-analytic)).
In one dimension the same term is `O(Q²ℓ)` and harmless.

## Assumptions

These fill in what the task leaves open. Each can be changed without
touching the rest of the method.

- **The particle size is physical.** `ℓ` is fixed in advance, with
  `0 < ℓ ≪ h`, and larger than the model's classical radius, so that the
  bare mass is positive (see [Why the near field must be
  analytic](#why-the-near-field-must-be-analytic)).
- **An analytic force binds the orbit.** The particles attract by an
  instantaneous Newtonian force `G M₁M₂/d²` in addition to their scalar
  interaction. Like scalar charges attract and unlike ones repel. The
  Newtonian force does not radiate, so the inspiral is driven by scalar
  radiation alone. This plays the role of the large black hole's gravity
  in the particle paper's motivation, and it lets the charges be small.
- **Flat spacetime, `c = 1`, and the particle paper's conventions:**
  `∂ₜΠ = ∇²Φ - ρ`, `∇²Φ₀ = ρ₀`, a force `-Q∇Φ` on a particle at rest.
  A static particle's field is then `-Q/4πr` outside it. The coupling is
  the covariant one of Poisson, Pound and Vega (see [Fields and
  particles](#fields-and-particles)).
- **The outer boundary is causally disconnected** from the binary and
  from wherever fluxes are measured, as the task says. The box is
  periodic, and no boundary treatment is designed.
- **Uniform forest**, as the package requires; see "Scope and
  non-goals".
- **`h` is the package's mesh spacing**, block extent over `N`, and
  `hₑ = (P+1)h` the element width.
- **The particles never overlap**, `|z₁ - z₂| > 2ℓ` with margin. The run
  stops well before that; a merger is out of scope.

## The physical model

### Fields and particles

The physical field obeys the equations of the particle paper's §III A:

    ∂ₜΦ = Π
    ∂ₜΠ = ∇²Φ - Σₐ ρₐ

A particle at rest is a spherical charge `ρ₀(r)` of radius `ℓ` and
charge `Q` with its static field, `∇²Φ₀ = ρ₀`. A particle at `z` moving
with velocity `v` carries these, Lorentz-contracted:

    r = x - z,   y = Λ(v) r,   Λ(v) = I + (γ - 1) v̂ v̂ᵀ
    |y|² = r² + γ² (v·r)²
    ρ(x) = ρ₀(|y|),   Φ_C(x) = Φ₀(|y|)

`Φ_C` is the field of a uniformly moving particle (the particle paper's
`Φ_lab`). Because the boost turns `∇² - (v·∇)²` into the rest-frame
Laplacian,

    □_v Φ_C := ∇²Φ_C - (v·∇)²Φ_C = ρ₀(|y|)

exactly.

The coupling is the covariant one of Poisson, Pound and Vega ("The
motion of point particles in curved spacetime", Living Rev. Relativ.
14, 7, 2011). The source is a Lorentz scalar, the rest-frame density
carried along with the particle, and `Q` is the coupling constant of the
action term `-Q∫Φ dτ`. The lab-frame charge `∫ρ d³x` is therefore
`Q/γ`. Unlike an electric charge, it is not conserved, because a scalar
source has no conserved current. The particle paper writes `γρ₀`
instead, to keep the lab-frame charge at `Q`. That choice would need
`γΦ₀(Λ(x - z))` as the particle field, and equations of motion of its
own, and it singles out the lab frame. It is not used here.

The normalization stays the particle paper's. Poisson, Pound and Vega's
field, written `Φ_PPV` here, obeys `□Φ_PPV = -4πμ` with
`μ = q∫δ⁴(x - z(τ)) dτ`. Their static field is `q/r` and their mass
`m₀ - qΦ_PPV`. The dictionary is

    Q = √(4π) q,   Φ = -Φ_PPV / √(4π),   ρ = √(4π) μ

so that `QΦ = -qΦ_PPV` and the two actions agree term by term.

Both the field and the particle equations follow from

    L = Σₐ [ -M₀ₐ/γₐ - ∫ ρₐ Φ d³x ] + ½ ∫ (Π² - |∇Φ|²) d³x

with `M₀ₐ` the bare mass, plus the Newtonian potential energy between
the particles.

### The particle profile

    ρ₀(r) = (3465 Q / 512π ℓ³) (1 - r²/ℓ²)⁴        r < ℓ

The profile is polynomial and `C³` at `r = ℓ`. Polynomial makes every
particle quantity closed form; `C³` is enough smoothness for the
quadratures below. With `σ = r/ℓ`:

    Φ₀(r) = (Q / 1024π ℓ) (-693 + 1155σ² - 1386σ⁴ + 990σ⁶ - 385σ⁸ + 63σ¹⁰)
                                                                 σ < 1
    Φ₀(r) = -Q / 4πr                                             σ ≥ 1

The near field below is built from the radial potentials

    B_k(r) = ∫ ρ₀(x′) |x - x′|^k d³x′,     ∇²B_k = k(k+1) B_{k-2}

with `B₋₁ = -4πΦ₀`. Outside the particle they are finite sums of the
profile's moments, `⟨s²⟩ = 3ℓ²/13`, `⟨s⁴⟩ = ℓ⁴/13` and
`⟨s⁶⟩ = 7ℓ⁶/221`:

    B₁ = Q (r + ⟨s²⟩/3r)
    B₃ = Q (r³ + 2r⟨s²⟩ + ⟨s⁴⟩/5r)
    B₅ = Q (r⁵ + 5r³⟨s²⟩ + 3r⟨s⁴⟩ + ⟨s⁶⟩/7r)

Inside they are polynomials. For `B₁`:

    B₁′(r) = Q (231σ/128 - 231σ³/128 + 99σ⁵/64 - 55σ⁷/64
                + 35σ⁹/128 - 63σ¹¹/1664)                         σ < 1

`B₃` and `B₅` follow the same way **(open: not yet derived)**. Two
double integrals of the profile set the particle's self-force:

    δm = (1/12π) ∬ ρ₀ρ₀′ / |x - x′| = -⅓ ∫ ρ₀Φ₀
       = 693 Q² / 4199π ℓ ≈ 0.0525 Q²/ℓ
    ∬ ρ₀ρ₀′ |x - x′| = ∫ ρ₀ B₁ = 60984 Q²ℓ / 96577 ≈ 0.631 Q²ℓ

These were derived symbolically, and `B₁′` matches its exterior form to
first derivative at `σ = 1`. Any other compact profile works the same
way.

### Equations of motion

For a field `f`, its charge-weighted average over particle `a` is

    f̄ₐ := (γₐ/Q) ∫ ρₐ f d³x

normalized so that `f̄ = f` for a constant `f`. The gradient average is
written `∇̄f = -(γ/Q) ∫ (∇ρ) f`, so that the force is `∂L/∂z`. The
particle equations are the Euler–Lagrange equations of `L` with `Φ(z)`
replaced by `Φ̄`, with the Newtonian force `F_N` added to `d(mγv)/dt`:

    ż = v
    a = [ -(Q/γ)(∇̄Φₑ + v Π̄ₑ) + (I - vvᵀ) F_N ] / (m γ),   m = M₀ + Q Φ̄ₑ

In a static field the particle's kinetic plus interaction energy is
conserved, and a field that jumps across an element face is
differentiated in the sense of distributions. In the point limit and
without `F_N` these are the covariant
`m du^μ/dτ = -Q (η^{μν} + u^μu^ν) ∂_νΦ`. The particle paper's 3+1 form
is their slow-motion limit. Its covariant form has `M₀ - QΦ`, which is
Poisson, Pound and Vega's `m₀ - qΦ_PPV`. In the particle paper's
normalization the same mass is `M₀ + QΦ`, as here.

`Φₑ` is the total field minus the particle's own boosted static field
`Φ_C`. That part exerts no net force, since `ρₐ` and `Φ_C,a` are both
even about `zₐ`. Its average is a boost-invariant constant, which is
absorbed into `M₀`. Everything else counts: the remainder, the other
particle's field, and the particle's own acceleration terms of the next
section. The equations leave out terms of relative order `ℓ²∇²Φ` from
`∂ρ/∂v` in `∂L/∂v` and from the rigidity of the particle. At `ℓ ≪ h`
these are far below everything else.

### The near field of an accelerated particle

The retarded field is a sum over powers of the retardation
`R = |x - x′|`:

    Φ = -(1/4π) Σₙ ((-1)ⁿ/n!) ∫ Rⁿ⁻¹ ∂ₜⁿρ(t, x′) d³x′

- **Odd `n`** have polynomial kernels `R^{2j}`. Their fields are
  polynomials in `x - z`, smooth at the particle. They are the
  radiative part, starting with the radiation-reaction field
  `-(Q/12π) ȧ·(x - z)` at `n = 3`.
- **Even `n`** have kernels `R^{2j-1}`, which are not smooth at the
  particle. They are the time-symmetric part, half retarded plus half
  advanced. In flat spacetime that is Detweiler and Whiting's singular
  field, here for an extended source and therefore smooth inside the
  particle.

Count the order `N` of a term by derivatives of the velocity: `a` counts
1, `ȧ` and `a²` count 2, `ä`, `aȧ` and `a³` count 3. Outside the
particle an order-`N` term behaves like `r^{N-1}` times an angular
function, and it collects even `n` up to `2N`.

**How the terms are computed.** For a particle momentarily at rest at
`z`, write `r = x - z` and `s = x′ - z`.
1. **Trajectory and contraction.** The displacement is
   `Δz(t) = ½at² + ⅙ȧt³ + (1/24)ät⁴`. The profile is Lorentz-contracted
   along `v(t) = at + ½ȧt² + …`, with `Λ ≈ I + ½vvᵀ`. So
   `ρ(t, x′) = ρ₀(s + δ(t))` with `δ = (Λ - I)s - ΛΔz`.
2. **Expand.** Taylor-expand `ρ₀(s + δ)` in `δ`, and collect the
   even-`n` coefficients at each order `N`.
3. **Integrate.** Two identities reduce every integral to the radial
   potentials:

       ∫ R^k (a·∇′)^m ρ₀ = (a·∇)^m B_k
       ∫ R^k sᵢ ρ₀ = rᵢ B_k - ∂ᵢB_{k+2} / (k+2)

**Results**, for the extended particle, and in the point limit with
`n̂ = r/|r|`:

    order 1:  (1/8π) (a·∇)B₁
              → (Q/8π) a·n̂

    order 2:  -(1/8π) (a·r)(a·∇B₁) + (1/96π) (a·∇)²B₃
              → (Q r/32π) [a² - 3(a·n̂)²]

    order 3:  (1/96π)(ä·∇)B₃ + (1/16π) a²(a·∇)B₃
              + (1/16π)(a·r)(a·∇)²B₃ - (7/960π)(a·∇)³B₅
              → (Q r²/64π) [2(ä·n̂) + 3a²(a·n̂) + 5(a·n̂)³]

At order 2 the first piece comes from the Lorentz contraction of the
profile and the second from the curvature of the trajectory. There is no
`ȧ` term: `ȧ` appears only at `n = 3`, in the regular radiation-reaction
field. Order 3 has no `a ȧ` term, since that appears only at `n = 5`.

Two checks:
- **Wave equation.** Each term satisfies the near-zone hierarchy
  `∇²Φₙ = ∂ₜ²Φₙ₋₂` that follows from `∂ₜ²Φ = ∇²Φ - ρ`. This was checked
  for the `a²` terms and the `ä` term.
- **Uniform acceleration, point limit.** At `t = 0` the exact retarded
  field of a point charge in hyperbolic motion with proper acceleration
  `g` equals the time-symmetric field, by time symmetry. Numerically it
  is exactly

      Φ = -Q / (4π r √(1 + g r cosθ + g²r²/4))
        = -(Q/4πr) Σₖ (-g r/2)^k P_k(cosθ)

  inside the region the charge's light cone reaches. Its terms
  `k = 1, 2, 3` reproduce orders 1–3, with `ä = -3g²a` for that motion,
  and the residual after subtracting them falls as `r³`.

The interior pieces of the extended forms come only from the derivation
**(open: check them independently)**.

**The self-force of these terms.** Orders 1 and 3 are odd under
`n̂ → -n̂` and push on the extended particle; order 2 is even and does
not.
- **Order 1** gives the self-inertia `-δm a`: the momentum `δm v` of the
  particle's own field has to be accelerated with it.
- **Order 3** gives a finite-size force
  `-(ä/24π) ∬ρρ′|x - x′| ≈ -0.0084 Q²ℓ ä`, plus an `a² a` term of the
  same structure **(open: not yet computed)**. Relative to radiation
  reaction it is of order `ℓΩ`. On a circular orbit, where `ä = -Ω²a`,
  it slightly reduces the effective inertia.
- **In the point limit** the order-3 force vanishes in proportion to
  `ℓ`, and the singular field exerts no force beyond the self-inertia,
  which renormalizes the mass.

### Why the near field must be analytic

The self-inertia `δm ∝ Q²/ℓ` comes from the order-1 term's change, from
linear in `r` to the direction-dependent constant `(Q/8π) a·n̂`, within a
distance `ℓ` of the particle. Polynomials on an element of width
`hₑ ≫ ℓ` cannot follow that. If the order-1 term is left to the grid,
the grid supplies a force of order `Q²a/hₑ` instead of `δm a`: almost
the whole self-inertia is missing.

So the particle field must carry the near field, and with the particle's
actual acceleration. A predicted acceleration `α` would leave
`(Q/8π)(a - α)·n̂` on the grid, and with it an error of about
`δm (a - α)` that no resolution removes. The order of the near field
carried then sets the convergence of the force:

| particle field carries | first non-smooth structure left in the remainder | grid's force error |
|---|---|---|
| boosted static field only (the particle paper) | `(Q/8π) a·n̂` (`r⁰`) | the whole `δm a` |
| plus order 1 with a predicted `α` | `∝ (a - α)·n̂` (`r⁰`) | `≈ δm (a - α)`; no convergence |
| plus order 1 with the actual `a` | order 2 (`r¹`) | `O(1)` |
| plus order 2 | order 3 (`r²`) | `O(h)` |
| plus order 3 | order 4 (`r³`) | `O(h²)` |

A structure that behaves like `r^k` is approximated by polynomials with a
gradient error of order `h^{k-1}`, hence the last column. The order-2
terms exert no force on a symmetric particle in the continuum, but on
the grid they cause an error that does not shrink. Each row also needs
the lab-slice corrections of [Moving particles and the lab
slice](#moving-particles-and-the-lab-slice) at the same order.

The bare mass is `M₀ = M - δm(ℓ)`, with `M` the physical mass. It must be
positive, `ℓ > 0.0525 Q²/M`: the particle must be larger than the
model's classical radius. Below it the coupled particle and field have
runaway solutions, as an extended classical charge does, which is a
property of the model rather than of the numerics. This bound is what
rules out a binary bound only by its own scalar field. There the
orbit ties the charge to the speed, `Q² ≈ 16πMv²d` for equal masses and
`Q₂ = Q₁/2`, so `δm/M ≈ 2.6 v²d/ℓ`. With `ℓ ≪ h` that would need elements
coarser than the orbit, or an orbit too slow to inspiral measurably.

## The split

As in the particle paper, `Φ = Φ_r + Φ_p` and `Π = Π_r + Π_p`. The
particle fields `Φ_p` and `Π_p` may be *any* functions of the state, and
the remainder equations follow from the physical ones. The choice
decides only how smooth the remainder is near the particles.

### The particle field

Write the particle state as `X = (z₁, v₁, z₂, v₂)`. Each particle's field
is its boosted static field plus the near-field terms of orders 1–3:

    Ψₐ = Φ_C(zₐ, vₐ) + Φ⁽¹⁾[aₐ] + Φ⁽²⁾[aₐ] + Φ⁽³⁾[aₐ, ä_pred,a(X)]

The terms `Φ⁽ᴺ⁾` are those of [The near field of an accelerated
particle](#the-near-field-of-an-accelerated-particle), evaluated on the
lab's time slice (see [Moving particles and the lab
slice](#moving-particles-and-the-lab-slice)). Orders 1 and 2 use the
particle's actual acceleration `aₐ`. The order-3 term's `ä` comes from
the analytic predictor of the next section, for a reason given there.
The particle fields are then

    Φ_p = W(x) Ψ̂,      Ψ̂ := Σₐ Ψₐ
    Π_p = ∂ₜΦ_p = W(x) 𝒟Ψ̂,
    𝒟 := Σₐ ( vₐ·∂_{zₐ} + aₐ·∂_{vₐ} + ȧₐ·∂_{aₐ} ) + (d/dt ä_pred)·∂_ä

`𝒟` is the time derivative along the actual motion. `W` is the window of
[The window](#the-window), equal to 1 around both particles. `Π_p` is
the exact time derivative of `Φ_p`, so the remainder's first equation
gets no source. The particle paper takes `Π_p = Σ vₐ·∂_{zₐ}Φ_C` instead,
the time derivative at fixed velocity, which puts a source proportional
to `a` into that equation. With the near field in `Φ_p`, the exact
derivative is both simpler and smoother.

### The acceleration and its derivatives

The particle field needs `a`, and its time derivative needs `ȧ`. The
effective source below needs one derivative more, `ä`. All three come
from the state:

- **`a`** comes from the equation of motion. The particle's own order-1
  and order-3 terms push on it in proportion to `a`, so this is a 3×3
  linear solve per particle. At rest it amounts to replacing the bare
  mass `M₀` by the physical mass `M = M₀ + δm`.
- **`ȧ`** comes from the time derivative of the equation of motion. The
  forces read `Φ_r` and `Π_r` near the particle, and `∂ₜΦ_r = Π_r`
  exactly. `∂ₜΠ_r` enters only through the relativistic `v Π̄` term,
  where it brings in the source `S`, which depends on `ä`. That coupling
  is weak: in the continuum it vanishes by symmetry, and it carries a
  factor `v`. It is cut by taking that one `ä` from the predictor.
- **`ä`** comes from the second time derivative. It now reads `∂ₜΠ_r`
  through the gradient average, and `S` depends on `ä` linearly through
  the term `ä·∂_aΨ̂`. This is a 3×3 linear solve per particle. Its matrix
  is `M·I` plus a correction of the size of the grid's self-inertia
  `δm_h ~ Q²/hₑ`, so it is well-conditioned while `δm_h ≪ M`. The
  relativistic term would need `d³a/dt³`, which again comes from the
  predictor.

At finite `ℓ` these loops are regular linear systems. In the point limit
the coefficient of the `ä` solve would be `δm → ∞`. That is where the
self-referential dependence on derivatives of the self-acceleration,
which the particle paper warns about, comes from, and why the particle
here must have a size.

**The predictor.** `ä_pred,a(X)` is an analytic second derivative of the
acceleration. It is a smooth function of the particle positions and
velocities alone. It is obtained by differentiating, along the motion it
predicts, the two-body acceleration that the instantaneous forces give:
the Newtonian force and each particle's boosted static field acting on
the other, with the physical masses. Derivatives come from dual numbers
through these closed forms.

The predictor supplies the order-3 term's `ä` and closes the relativistic
couplings above. The order-3 term cannot use the solved `ä`: `Π_p` would
then need `d³a/dt³`. A function of `X` alone has a time derivative the
state determines. The split stays exact whatever the predictor. Its error
leaves `Φ⁽³⁾[ä - ä_pred]` in the remainder, a structure that behaves
like `r²`, which costs one order of convergence at a small coefficient.
For the binary, the predictor's error is of post-Newtonian size, `O(v²)`.

### The remainder equations

    ∂ₜΦ_r = Π_r
    ∂ₜΠ_r = ∇²Φ_r - S

    S = W [ 𝒟²Ψ̂ - ∇²Ψ̂ + Σₐ ρₐ ] - 2∇W·∇Ψ̂ - Ψ̂ ∇²W

They follow from `∂ₜΠ = ∇²Φ - Σρ` with `W = 1` wherever `ρₐ ≠ 0`. The
bracket is the residual of the particle field's wave equation along the
actual motion. Its behaviour near the particle is set by the order the
field is carried to. The hierarchy `∇²Φₙ = ∂ₜ²Φₙ₋₂` cancels everything
below the highest order kept, so with orders up to `N` the residual
behaves like `r^{N-2}`. The response of the remainder to it then behaves
like `r^N`: one order beyond those kept, as the table in [Why the near
field must be analytic](#why-the-near-field-must-be-analytic) says.

`S` is not computed by subtracting large numbers. Near the particle,
`∇²Ψ̂`, `(v·∇)²Ψ̂` and `ρ` are each of size `Q/ℓ³`. The identity
`□_vΦ_C = ρ` is used analytically, and each order's Laplacian is taken
from the hierarchy in closed form. What is left is assembled from terms
of size `Qa/ℓ²` and smaller.

With the particle paper's choice (`Φ_p = Φ_C` and
`Π_p = Σ vₐ·∂_{zₐ}Φ_C`), the same derivation gives its remainder
equations (§III C 2): `∂ₜΦ_r = Π_r - a·∂_vΦ_lab` and
`∂ₜΠ_r = ∇²Φ_r - a·∂_vΠ_lab`.

### The window

`W(x) = w(|x - x_c|)`, centred at the binary's centre of mass `x_c` and
fixed in time. `w = 1` for `r ≤ R₁`, and it falls smoothly to 0 at `R₂`:
a polynomial step, `C^{P+1}` or smoother. `R₁` exceeds every particle's
distance from `x_c` by more than `ℓ` with margin, which keeps holding as
the orbit shrinks. The transition `R₂ - R₁` spans about `d` and several
elements, and is fixed in physical units, so each refinement resolves it
better.

The window does three things:

- **Cut off the near-field terms.** They do not decay: the order-1 term
  tends to `(Q/8π) a·n̂`, and orders 2 and 3 grow like `r` and `r²`.
  Without a window the remainder would have to cancel them everywhere.
- **Confine the source.** `S` vanishes outside `R₂`, so the projection
  below runs over the elements of one ball.
- **Make the outer remainder physical.** Outside `R₂`, `Φ_r = Φ`, so
  fluxes and waveforms come from the state directly.

The window costs only discretization error. The split is exact for any
`W`, so inside `R₁` the continuum remainder does not depend on it, and
the window's terms `-2∇W·∇Ψ̂ - Ψ̂∇²W` are smooth on the scale of the
transition and resolved. Because `W = 1` at both particles, each
particle's force from the other particle's `Ψ_b` is analytic and exact.
In particular the bulk of the scalar part of the binding force never
touches the grid.

### Moving particles and the lab slice

The near-field terms above hold in the particle's instantaneous rest
frame. There the acceleration is the proper acceleration
`α′ = γ²Λ(v) a`, and the higher orders use proper-time derivatives. The
particle field is needed on the lab's time slice, however. A lab point
at `r = x - z` sits at rest-frame position `y = Λ(v) r` and rest-frame
time `t′ = -γ v·r`, so

    Ψ_lab(r) = Ψ′(t′, y) = Σⱼ (t′ʲ/j!) ∂ⱼ_{t′} Ψ′(0, y)

where `Ψ′(t′, ·)` is the rest-frame near field at rest-frame time `t′`,
with the particle at its rest-frame position and velocity at that time.
Every `t′` carries one power of `r`, and every `∂_{t′}` raises the order
by one. So a lab-frame term of order `N` collects rest-frame terms up to
order `N`, and order `N` still behaves like `r^{N-1}`. The identities of
the previous section apply unchanged.

Boosting the rest-frame terms without this re-expansion is not enough. A
symbolic check of the order-1 term shows the boosted form missing a part
proportional to `v²a` that behaves like `r⁰`. By the table above, that
leaves a force error of about `δm v² a` that does not converge. At
`v = 0.2` and `δm/M ≈ 0.26` that is about 1% of the binding force, a
hundred times radiation reaction. The re-expansion is mechanical
computer-algebra work with the same identities **(open: not yet
derived)**. Until it is, tests run at small `v`, where the missing terms
are suppressed by powers of `v`.

## The discretization

### The remainder on OLDG

The remainder system is the package's equation with a source, in the
same second-order form: `u = Φ_r`, `v = Π_r`. The particle paper's
finite-volume scheme needed `Ψ = ∇Φ` as a variable. OLDG takes the
remainder equations as they stand, and its auxiliary gradient stays
internal to the right-hand side. The semi-discrete system is

    du = v
    dv = Σ_d D_d(D_d u) - P S

`wave_rhs!` is unchanged and the source is subtracted after it. `P` is
the transfer operator below. `v` is truncated in the state through the
limiters, as in "Truncation", and that also removes the top mode of
`P S`. Because `Π_p` is the exact time derivative of `Φ_p`, `u`'s
equation has no source. The alternative with a source there needs that
source truncated too (see [Sources and
truncation](#sources-and-truncation)).

Ghost exchange, layout and time step are untouched. The source lives on
owned nodes and is element-local.

### The transfer operator

Within an element, let `φₙ` be the tensor-product Lagrange polynomial of
Gauss–Lobatto node `n`, and `wₙ = (hₑ/2)³ ŵᵢŵⱼŵₖ` its weight. Write
`⟨f, g⟩_w = Σₙ wₙ fₙ gₙ` for the quadrature inner product, the one
summation by parts holds in, and `g_h(x) = Σₙ gₙ φₙ(x)` for the
reconstruction of a grid function, which is discontinuous across element
faces. The transfer from the particles to the grid is

    (P S)ₙ = (1/wₙ) ∫_{element(n)} S(x) φₙ(x) d³x

equivalently `⟨P S, g⟩_w = ∫ S g_h d³x` for every grid function `g`.

- **It is element-local.** Each element owns both face nodes, so the
  transfer needs no assembly and no exchange.
- **Pairings are exact.** In the discrete energy, the work the source
  does on the remainder is the continuum integral against the
  reconstructed field, `⟨Π_r, P S⟩_w = ∫ S Π_r,h`.
- **It is consistent at the scheme's order.** For smooth `S`,
  `(P S)ₙ - S(xₙ) = O(hₑ^P)`, the same order as the lumped Gauss–Lobatto
  mass that OLDG already uses.
- **It filters.** `S` is never sampled at a point. The particle paper's
  remark that point evaluation of these sources has unbounded error
  applies to nodal collocation too.

From the grid to a particle the transfer is the adjoint. For a particle
kernel `K` (here `ρₐ` and `∇ρₐ`),

    ∫ K Φ_r,h d³x = ⟨P K, Φ_r⟩_w

which gives the remainder's contributions to the averages:

    Φ̄_r = (γ/Q) ⟨P ρₐ, Φ_r⟩_w
    Π̄_r = (γ/Q) ⟨P ρₐ, Π_r⟩_w
    ∇̄Φ_r = -(γ/Q) ⟨P ∇ρₐ, Φ_r⟩_w

The last one is `∫ρ∇Φ_r,h` with `∇` the distributional gradient of the
discontinuous field, so it counts `ρ·[Φ_r,h]` at element faces. It is
`∂/∂z` of the discrete interaction energy `⟨P ρₐ(z), Φ_r⟩_w`. In a
static remainder the particle's energy is therefore conserved exactly in
the semi-discrete sense, and crossing elements does no spurious work.
Averaging the gradient pass's `D u` against `P ρ` instead would not be
the `z`-gradient of anything, so it is not used.

Because `ρ` is much narrower than an element, the particle crosses an
element face within a time `ℓ/v`, which can be shorter than a time step.
It then feels the jump `[Φ_r,h]` across the face as a brief impulse
`Q[Φ_r,h]`. The jumps are small where the remainder is smooth, which the
near-field terms ensure. If tests show the crossings, the remainder can
instead be read through a smooth symmetric kernel of width about `hₑ`,
with its moments corrected. That force is still the gradient of a
potential, and the physical form factor `(⟨s²⟩/6) ∇□Φₑ` then becomes a
small analytic correction **(open)**.

### Quadrature

The transfer integrals have integrands that are smooth but sharply
peaked near the particles, on the scale `ℓ ≪ hₑ`. Two rules handle them.

- **Far from the particles** (beyond `r_near ≈ 2hₑ`): tensor-product
  Gauss–Legendre with `q = P + 3` points per dimension. `S` is evaluated
  at the `q³` points, and the 1D `q × (P+1)` table of Lagrange values
  contracts it along one dimension at a time (sum factorization).
- **Within `r_near` of a particle**: per dimension, a sinh-transformed
  Gauss–Legendre rule centred on the particle's coordinate with scale
  `ℓ` (Johnston and Elliott, "A sinh transformation for evaluating
  nearly singular boundary element integrals", 2005). It clusters points
  near the particle, and the number of points it needs grows only like
  `log(hₑ/ℓ)`. The transformation depends smoothly on `z`, so the
  transfer stays a smooth function of the particle position, and RK4
  keeps its order. Adaptive subdivision would change discretely as the
  particle moves, and its error with it. `q` is chosen by a convergence
  test that puts the quadrature error below `10⁻¹²` relative.

At `r_near` both rules agree to that tolerance, so switching between
them is invisible. The force kernels `P ρₐ` and `P ∇ρₐ` use the near
rule on the elements that meet the particle's support, at most eight.
The analytic parts of a particle's averages (the other particle's `Ψ_b`,
and the particle's own near-field terms) need no grid. They are
integrals over the particle's ball in its rest frame: Gauss in radius,
a product Gauss rule on the sphere.

As with the element tables (see "Precision"), the Gauss nodes and
weights, the sinh-rule's reference nodes and the Lagrange tables are
built in `BigFloat` and rounded once.

**A faster alternative** (optional). Since the test functions vary on
the scale `hₑ ≫ ℓ`, the source can be split into two parts:
- an exterior form, a finite sum of homogeneous functions of `x - z`
  including the `ℓ²/r²` tails of the `B_k`, integrated over the element
  by Duffy-type quadrature or analytically;
- a core supported in `r < ℓ`, which enters only through a few analytic
  moments times derivatives of `φₙ` at `z`, with an error of order
  `(ℓ/hₑ)^m`.

That removes all quadrature at the scale `ℓ`. It is worth doing only if
profiling shows the sinh rule to be the bottleneck.

For cost, take a window ball of radius `2.5d` with `hₑ = d/8`. It holds
about 30 000 elements, and the far rule evaluates `S` (with its
dual-number derivatives) about `7·10⁶` times per particle and stage.
That is roughly ten times the wave equation's work on the same elements.
It becomes a modest fraction of the total once the box is much larger
than the window **(open: measure)**. If needed, `q` can drop to `P + 1`
far from the particles, where `S` is smooth on the element scale.

### The right-hand side

One evaluation, for the state `(u, v, X)`:

1. **Averages and `a`.** For each particle, the averages `Φ̄ₑ`, `∇̄Φₑ`
   and `Π̄ₑ`: the remainder's part by the adjoint transfer, the other
   particle's `Ψ_b` and the particle's own near-field terms analytically.
   Then `a` from the equation of motion, a 3×3 solve.
2. **`ȧ`.** From the time derivative of the forces, using `∂ₜΦ_r = v`.
3. **Field equation without source.** `wave_rhs!`, which gives
   `∂ₜΠ_r` up to the source.
4. **`ä`.** From the second time derivative of the forces, using step
   3's `∂ₜΠ_r` near the particle and the part of `S` that is linear in
   `ä`, projected on the elements that meet the particle's support. A
   3×3 solve.
5. **Source.** `P S` over the window's elements, subtracted from `dv`.
6. **Particles.** `żₐ = vₐ`, `v̇ₐ = aₐ`.

The result is a pure function of the state, as in `CODE.md`. The
particle state enters the kernels as an `isbits` tuple. The sums over
the elements that meet a particle's support are a `mesh_fold` over
blocks, like `quadrature_norm`.

### Time integration

The state vector joins the field state and the 12 particle numbers, for
example as a `RecursiveArrayTools.ArrayPartition`, so that OrdinaryDiffEq
sees one vector. The field part keeps `statearray`'s interleaved layout.
`truncate_stage!` acts on the field part only. The integrator is RK4
with `truncate_stage!` as stage and step limiter, and the time step is
`oldg_timestep`'s.

The particle's own light-crossing time `ℓ` is shorter than a time step.
That is consistent with the method: the near field is analytic and
quasi-static, which is exactly what a time step longer than `ℓ` cannot
see. The slow-motion expansion behind it is valid while `ℓΩ ≪ 1`. The
transfer varies on the time scale `hₑ/v` a particle takes to cross an
element, which is many steps. At element faces it is less smooth in `z`,
and RK4 may lose some order at each crossing. Carrying the near field to
a higher order makes the source more regular and that loss smaller.

## Initial data: a quasi-circular orbit

1. **Orbit.** Choose `d₀`, `Mₐ`, `Qₐ` and `G`. Take the angular
   frequency `Ω` from the predictor's two-body force, which must be
   centripetal on a circle. Place the particles about the centre of mass
   with velocities perpendicular to the separation, and set the total
   mechanical momentum `Σ γₐ Mₐ vₐ` to zero.
2. **Fields.** `Φ_r(0) = (1 - W) Ψ̂` and `Π_r(0) = (1 - W) 𝒟Ψ̂`, with the
   accelerations and their derivatives of the circular orbit. Both are
   sampled at the nodes with `fill_at_nodes!`, and are smooth because
   `1 - W` vanishes near the particles. Then truncate `Π_r` as
   `initial_state` does. The total field is then the particle field
   everywhere: the instantaneous field of particles on the circle,
   without retardation. It is not the field of a binary that has been
   orbiting. The difference radiates away as junk of relative size
   `O(v²)` and leaves an eccentricity of that order.
3. **Warm-up.** Evolve with the particles held on the circle, the
   particle paper's forced mode. On a prescribed path `a`, `ȧ` and `ä`
   are known exactly, so no solve is needed. Run for
   `t_w ≈ 2(R₂ + d)`, so that the junk leaves the window and the
   remainder settles into the co-rotating state. During the warm-up,
   update `Ω` a few times from the measured radial force, the condition
   for a circular orbit with the full force.
4. **Release.** Switch to `a`, `ȧ` and `ä` from the state. Give each
   particle the radial velocity that the measured tangential force
   implies through energy balance (`ḋ = Ė / (dE_orb/dd)`). This removes
   most of the eccentricity that releasing from a circle would cause, as
   eccentricity reduction does in numerical relativity. Measure the
   remaining eccentricity from `d(t)`.

## Diagnostics

- **Orbit:** the separation `d(t)`, the orbital phase and frequency, and
  the eccentricity from the oscillation of `d(t)` about its secular
  trend.
- **Flux:** the energy flux through spheres of radius `R_ext > R₂`,
  `-∮ Π_r ∂_rΦ_r dA`. Outside the window `Φ_r` is the physical field.
  The flux is a quadrature on the sphere of the reconstructed fields. A
  rotating source's flux averaged over an orbit equals its power through
  any enclosing sphere, so `R_ext` need not reach the wave zone.
- **Balance:** the two-body orbital energy, with physical masses and the
  analytic interaction, against the time integral of the flux, shifted
  by the light travel time `R_ext`.
- **Momentum:** the drift of the centre of mass, which should stay small.
- **Leading-order prediction:** a scalar dipole radiates `|d̈|²/12π`,
  with `d = Σ Qₐ zₐ`. For a Newtonian circular orbit with total
  attraction `K/d²` this gives

      K = G M₁M₂ + Q₁Q₂/4π,   Δq = Q₁/M₁ - Q₂/M₂
      E_orb = -K/2d,   ḋ = -Δq² K / (6π d²)
      d(t)³ = d₀³ - (Δq² K / 2π) t

  correct to `O(v²)`. Equal charge-to-mass ratios give no dipole
  radiation and a much slower, quadrupolar inspiral; the parameters below
  avoid them.

## Expected accuracy

The error budget, and how each term behaves as `h` shrinks at fixed `ℓ`,
with the near field carried through order 3:

| source | size | at fixed ℓ |
|---|---|---|
| remainder discretization, window transition included | `O(hₑ^P)` of the remainder | falls at order `P` |
| first near-field order not carried (order 4, `r³`) | grid error in the force | falls at order 2 |
| predictor's error in the order-3 term | `Φ⁽³⁾[ä - ä_pred]`, `ä - ä_pred = O(v² ä)` | falls at order 1, small coefficient |
| lab-slice corrections, until derived | `≈ δm v² a` from order 1 | does not fall |
| quadrature | below `10⁻¹²` | none |
| RK4 | `O(dt⁴)`, less smooth at face crossings | order 4 or somewhat less |
| finite size | physical: `δm` and the order-3 force are exact constants of the particle; `O(ℓ²∇²Φ)` rigidity terms dropped from the equations of motion | none; a property of the model |

There are no artificial finite-size effects: `ℓ` is the particle's size,
and the grid adds no dependence on it beyond the transfer's quadrature.
With orders 1–3 carried, the force converges at second order, up to a
first-order term proportional to the predictor's error in `ä`. A first
implementation carrying orders 1–2 converges at first order. Both
assume the lab-slice corrections, without which a term proportional to
`v²` does not converge at all.

## Validation

Each step has a criterion that says when it is done. The order follows
how the implementation would grow.

1. **Profile and near field.** The closed forms agree with numerical
   quadrature of their definitions (`Φ₀`, `B_k`, `δm`) to `10⁻¹⁴`. Each
   order satisfies `∇²Φₙ = ∂ₜ²Φₙ₋₂`. The point limits reproduce the
   field of hyperbolic motion. The dual-number derivatives agree with
   finite differences.
2. **Transfer.** `⟨P S, g⟩_w = ∫ S g_h` holds for random `g` to the
   quadrature tolerance. `P S` converges to `S` at the nodes at order
   `P` for smooth `S`. `P S` is smooth in `z`: second differences in `z`
   across element faces and across the switch between quadrature rules
   show no jump above the tolerance.
3. **A particle at rest, and in uniform motion,** with
   `Φ_r(0) = (1 - W) Ψ̂`. `Φ_r(t)` stays `(1 - W)Φ_C(z + vt)` and the
   force stays zero, both to the window's discretization error, which
   falls at order `P`. This checks the window, that the uniformly moving
   field is exact, and that the particle's own field exerts no force.
4. **One particle on a prescribed path**, where `a`, `ȧ` and `ä` are
   known exactly and nothing is solved.
   - *Hyperbolic motion.* Outside the particle, `Φ_r + Φ_p` matches the
     exact retarded field, computed by root-finding for the retarded
     time. At `t = 0` that field is time-symmetric, so `Φ_r` near the
     particle holds only the orders not carried. The radiation-reaction
     force of a point charge vanishes for this motion, so the remainder's
     force on the particle must converge to zero, up to the finite-size
     terms of the orders not carried.
   - *A circle at small `v`.* The remainder's tangential force matches
     `(Q²/12π) ȧ`. Its radial force converges to zero, because the
     self-inertia is analytic. Both converge at order 1 with orders 1–2
     carried, and at order 2 with orders 1–3.
5. **The lab slice.** With the corrections derived, step 4's circle at
   `v ≈ 0.2` converges as at small `v`.
6. **Dynamic particles.** `a`, `ȧ` and `ä` from the state. The `ä` solve
   stays well-conditioned, and a single particle on a prescribed external
   force reproduces step 4. Then the Newtonian binary: the flux balances
   the loss of orbital energy, `ḋ` matches the leading-order formula to
   `O(v²)`, and the orbital phase converges as in [Expected
   accuracy](#expected-accuracy).
7. As for the package: the suite passes with `--check-bounds=yes` and on
   four threads.

## Parameters for a first run

A Newtonian binary with opposite charges, which gives the largest dipole
for a given self-inertia: `M₁ = M₂ = M`, `Q₁ = -Q₂ = q`, initial
separation `d₀`, and units `M = d₀ = 1`. Each particle moves at
`v = 0.2`, so `Ω = 0.4` and `K = MΩ²d³/2 = 0.08`. With

    ℓ = 10⁻³,   q² = 0.005,   G = K + q²/4π ≈ 0.0804

the particle has `δm/M ≈ 0.26`, so `M₀ ≈ 0.74`, comfortably above the
classical-radius bound. The dipole formula gives:

| quantity | value |
|---|---|
| period | 15.7 |
| `ḋ` | `-8.5·10⁻⁵` |
| `Δd` per orbit | `1.3·10⁻³` |
| orbits to `d = 0` (leading order, past its validity) | about 250 |
| radiation reaction / binding force | `1.1·10⁻⁴` |
| `ℓΩ` (order-3 force / radiation reaction) | `4·10⁻⁴` |

Proposed resolutions: `P = 3`, `hₑ = d/8, d/12, d/16`, so
`h = d/32, d/48, d/64` and `ℓ/h = 0.03`–`0.06`. Window `R₁ = 1.25`,
`R₂ = 2.5`. A warm-up of `t_w ≈ 7`, then five orbits, over which `d`
falls by about 0.7%. At the finest resolution the time step is about
`0.002`, longer than `ℓ`, as [Time integration](#time-integration)
assumes.

Steps 4–5 of [Validation](#validation) run the same particle at
`v = 0.05` first, where the lab-slice corrections are small.

## Mapping onto the code

- `src/particle.jl`: the profile and the `B_k` in closed form; the near
  field of orders 1–3 and, once derived, its lab-slice re-expansion;
  `Ψ`, `𝒟Ψ` and `S`, differentiated with dual numbers; the predictor;
  the self-force of a particle's own near-field terms.
- `src/transfer.jl`: the Gauss and sinh-rule tables, kept in an `isbits`
  struct beside `Element` so that `Element` is unchanged; the projection
  kernel over the window's elements; the adjoint averages through
  `mesh_fold`. The projection is the package's first per-element kernel,
  the extension "Possible extensions" in `CODE.md` proposes, because sum
  factorization needs the whole element.
- `src/inspiral.jl`: an `InspiralProblem` holding a `WaveProblem`, the
  particles, the window and the tables; `inspiral_rhs!` with the six
  steps of [The right-hand side](#the-right-hand-side); the initial data,
  warm-up and release; and the flux spheres.
- New dependencies: ForwardDiff and RecursiveArrayTools. TreeAMR does
  not change, and the forest stays uniform.
- The sharp edges in `CLAUDE.md` all hold. Positions come from
  `node_coordinates`, tables are built in `BigFloat`, and `v` is
  truncated in the state through the limiters.

The implementation can follow [Validation](#validation) step by step.
Steps 1–4 need neither the solves nor the predictor, and already show
whether the method works at `ℓ ≪ h`.

## Alternative: particle size proportional to h

The particle paper's tests, and the first version of this design, take
the particle about as large as the mesh spacing: `ℓ = κh` with `κ`
between 1 and 2, held fixed along a convergence series. The integrals
are then never nearly singular, and the grid partly resolves the
particle's near field. The variant is documented as a comparison, and
because it is the setting in which a binary bound only by its own scalar
field is possible. The same code runs it with `ℓ` chosen comparable to
`h`.

### What changes

- **`ℓ` shrinks with `h`.** Finite-size effects become part of the error
  against the point-particle limit. `δm ∝ 1/ℓ` grows along a series,
  while the physical mass `M` is held fixed. `M₀ = M - δm(ℓ)` must stay
  positive, so the classical radius bounds the finest member of a
  series; a margin `M₀ ≥ M/2` is proposed.
- **The particle field carries order 1 only**, with a *predicted*
  acceleration `α(X)` in place of the actual one. Its time derivative is
  taken along the predicted motion.
- **Nothing is solved.** `α` depends on `X` alone, so every source is
  explicit.
- **Quadrature** needs the sinh rule only at the scale `ℓ ≈ h`, with few
  points.

### The acceleration predictor

Each particle's field is

    ψ(y; α′) = Φ₀(|y|) + (1/8π) B₁′(|y|) α′·ŷ
    Ψₐ(x; X) = ψ( Λ(vₐ)(x - zₐ);  γₐ² Λ(vₐ) αₐ(X) )

    Φ_p = W Ψ̂,   Π_p = W (𝒱·∂_X) Ψ̂,   𝒱(X) := (v₁, α₁(X), v₂, α₂(X))

`𝒱` is the predicted flow of the particle state, and `Π_p` the time
derivative of `Φ_p` along it. In increasing order of accuracy:

- `α = 0`: the particle paper, free flight.
- `α_C`: the equations of motion with `Φₑ` replaced by the other
  particle's `Φ_C`, evaluated at `zₐ` in point form, with the physical
  mass. This is the instantaneous, boosted two-body attraction.
- `α_C + α_RR`: plus the leading radiation reaction. For a system much
  smaller than its wavelength the radiation-reaction field near the
  source is the uniform gradient `-(1/12π) d⃛·x` of the dipole moment
  `d = Σ Q_b z_b`. This gives `α_RR,a = (Qₐ / 12π Mₐ) Σ_b Q_b α̇_C,b`,
  with `α̇_C` taken along the predicted flow.

The split is exact for every `α`. With `α ≈ a`, the particle's own
order-1 term exerts the self-inertia `-δm α` analytically, and the grid
supplies only `-δm_h (a - α)`, where `δm_h` is the grid's version of
the self-inertia. The error `(δm - δm_h)(a - α)` is smaller than in the
particle paper's split by the predictor's relative error, which for
`α_C + α_RR` is `O(v²)`. The radiation-reaction part `α_RR` matters
most, because an error in the inertia for the tangential acceleration is
an error in the inspiral rate.

### Sources and truncation

    ∂ₜΦ_r = Π_r - S_Φ
    ∂ₜΠ_r = ∇²Φ_r - S_Π

    S_Φ = W Σₐ (aₐ - αₐ)·∂_{vₐ} Ψ̂
    S_Π = W [ Σₐ (aₐ - αₐ)·∂_{vₐ} (𝒱·∂_X Ψ̂) + R ] - 2∇W·∇Ψ̂ - Ψ̂ ∇²W
    R   = (𝒱·∂_X)² Ψ̂ - ∇²Ψ̂ + Σₐ ρₐ

`∂_{vₐ}` acts on `Ψ̂` as a function of `X`, so it reaches `α_b(X)` too.
With `α ≡ 0` and `W ≡ 1`, `R = 0` and the sources are the particle
paper's, `a·∂_vΦ_lab` and `a·∂_vΠ_lab`. Near a particle at rest, the
`1/r²` part of `S_Π` is `-(a - α)·∇Φ₀`; the particle paper's split has
`-a·∇Φ₀`. At `v ≠ 0`, `R` keeps the part proportional to `v²α` that the
boosted order-1 term misses on the lab slice.

**`S_Φ` must be truncated.** On OLDG,

    du = v - 𝒯 P S_Φ
    dv = Σ_d D_d(D_d u) - P S_Π

where `𝒯` is the truncation projector, called `Π` in `CODE.md`. In the
truncated scheme `v` has degree `P - 1`, so `∂ₜu` does too, and `u`'s top
mode never changes after the initial data. In the continuum, `Π_r` and
`S_Φ` both carry the near-particle structure of `∂ₜΦ_p` that `Π_p` does
not account for, and they cancel in `∂ₜΦ_r`. On the grid that
cancellation can happen only below degree `P`. Without `𝒯`, the top mode
of `P S_Φ` would integrate into `u` with nothing to cancel it. This does
not conflict with "Truncation" in `CODE.md`: `v` is still truncated in
the state, and what is projected is a source in `u`'s equation. With
`truncate=false`, `𝒯` is omitted.

### Accuracy along a series

How each error term behaves in a series at fixed `κ = ℓ/h`:

| source | size | with `ℓ/h` fixed |
|---|---|---|
| remainder discretization, window transition included | `O(hₑ^P)` of the remainder | falls at order `P` |
| grid self-inertia of the prediction error | `ε δm (a - α)`, `ε = O(1)` | grows as `1/h`; small through `a - α` |
| `R`'s `v²α` part | `ε δm v² α` | grows as `1/h` |
| order-2 near-field terms | no net force in the continuum; on the grid a fluctuating force of order `ε Q² a²/4π` | does not fall; fluctuates with the position in the element |
| finite size against a point | conservative `O(ℓ ä)` and `O(ℓ²∇²Φ)`; dissipative `O(ℓ²)` | orders 1 and 2 |
| RK4 | `O(dt⁴)` | order 4 |

The finite-size terms follow from the parity of the near-field
expansion. Odd `n` have polynomial kernels, so their finite-size
corrections are moments of the profile and start at `O(ℓ²)`: radiation
reaction itself is exact. Even `n` give the self-inertia and then the
conservative order-3 force `-(ä/24π) ∬ρρ′|x - x′|`, of order `ℓΩ`
relative to radiation reaction. So against the point-particle limit the
orbit should converge at first order with a small coefficient, until the
terms that grow as `1/h` take over; the predictor's quality decides
when. A second series at fixed `ℓ`, letting `κ` grow, separates
discretization from model.

The main design is what removes the terms that grow or do not fall:
actual acceleration, orders 2 and 3, `ℓ` fixed. A profile whose moments
vanish could also remove the finite-size terms of this variant. In the
family `(1 - σ²)⁴(1 + c₁σ² + c₂σ⁴)`, `⟨s²⟩ = 0` and `∬ρρ′|x - x′| = 0` can
both be imposed, but `δm` then rises from `0.0525` to `0.231 Q²/ℓ`, so
`ℓ` must be 4.4 times larger for the same margin.

### A binary bound by its own field

Without the Newtonian force, the classical-radius bound ties `ℓ` to the
orbit. Reference configuration: `M₁ = M₂ = M`, `Q₂ = Q₁/2`, initial
separation `d₀`, units `M = d₀ = 1`. Each particle moves at speed `v`,
`Ω = 2v`, `Q₁² = 16πv²`. The dipole formula, the self-inertia of
particle 1 and the ratio of radiation reaction to binding force are:

| `v` | `Q₁` | period | `Δd` per orbit | orbits to `d = 0` | `δm₁ ℓ` | `F_RR/F_bind` |
|---|---|---|---|---|---|---|
| 0.05 | 0.354 | 62.8 | 5.2e-4 | 637 | 0.0066 | 4.2e-5 |
| 0.08 | 0.567 | 39.3 | 2.1e-3 | 155 | 0.0169 | 1.7e-4 |
| 0.1 | 0.709 | 31.4 | 4.2e-3 | 80 | 0.0264 | 3.3e-4 |
| 0.15 | 1.063 | 20.9 | 1.4e-2 | 24 | 0.0594 | 1.1e-3 |
| 0.2 | 1.418 | 15.7 | 3.4e-2 | 10 | 0.106 | 2.7e-3 |

Here `ḋ = -(4/3) v⁴` and `F_RR/F_bind = v³/3`. "Orbits to `d = 0`" is
the leading-order formula run past its validity. The table shows the
regime's limit, `F_RR/F_bind ≈ 0.06 ℓΩ δm₁/M`: a fast inspiral needs a
large classical radius, and the particle must stay larger than its
classical radius.

A first run of this variant: `v = 0.08`, `P = 3`, `κ = 2`, and
`hₑ = d/5, d/7, d/10`, so `ℓ = d/10, d/14, d/20` and
`δm₁/M = 0.17, 0.24, 0.34`, inside the `M₀ ≥ M/2` margin. Window
`R₁ = 1.25`, `R₂ = 2.5`. A warm-up of `t_w ≈ 7`, then three orbits,
over which `d` falls by about 0.6%.

## Open questions

- **The lab-slice re-expansion** of orders 1–3, without which a term
  proportional to `v²` does not converge (see [Moving particles and the
  lab slice](#moving-particles-and-the-lab-slice)).
- **The interior of `B₃` and `B₅`**, an independent check of the
  extended near-field terms inside the particle, and the `a² a` part of
  the order-3 self-force.
- **Stability of the coupled particle and field**: where it fails as
  `M₀ → 0`, whether the margin `M₀ ≥ M/2` is right, and how the `ä`
  solve behaves as `δm_h` grows.
- **Element-face crossings**: whether the narrow kernel `ρ` gives
  visible impulses, and the smooth reading kernel is needed.
- **Cost of the near-particle quadrature**, and whether the split into
  exterior forms and core moments is worth doing.
- **A post-Newtonian predictor**, to shrink the predictor's error in the
  order-3 term (and, in the alternative, `a - α`).
- **Free amplitudes for the near-field terms**: a Galerkin enrichment
  whose amplitudes are unknowns instead of the actual acceleration. It
  needs no solves and no predictor, at the price of a local, non-diagonal
  mass block (a 3×3 Schur complement per particle), cross terms that
  keep OLDG's energy estimate, and an open question of how to truncate
  the `Π`-space enrichment.
- **Filtering the projected source** with the OLDG paper's modal filter
  (its eq. 76), if the remainder near the particles turns out noisy.
- **The outer boundary**, which the task sets aside. On a uniform
  periodic box, five orbits at `v = 0.2` need a box of side about
  `2(R_ext + 90)`. A real run needs a refined forest, which waits on
  TreeAMR (see "What a refined forest needs"), or an absorbing boundary.

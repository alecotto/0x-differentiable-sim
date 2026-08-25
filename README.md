# 0x-differentiable-sim

**A fully differentiable, GPU-accelerated articulated-body simulator for
humanoid policy learning and variational analysis of contact dynamics.**

Every quantity in the physics pipeline — forward kinematics, mass matrix,
Coriolis/gravity forces, collision detection, contact forces, integrator —
is implemented as batched PyTorch tensor operations. Gradients flow exactly
through arbitrary-length rollouts, and an arbitrary number of parallel
environments `E` rides the leading dimension of every tensor.

## Validation status

All results below were produced by code at the current commit, verified by
aggressive audit under perturbed initial conditions.

### Gradient behaviour across contact events: two regimes (in progress)

**Preliminary; sweep under way** (`scripts/q2a_dcos.py`,
`benchmarks/q2a_dcos_*.json`). Walker heelstrike windows, analytic BPTT
vs smoothed central-FD reference, split-sample floors logged:

1. **Soft contact (k=2.5e4)**: analytic == FD to reported precision
   (cos ≥ 0.999 across events, floors ≈ 1.0) at every dt tested
   (1e-4–4e-4). The original humanoid n=1 measurement (norm ratio 0.99,
   cosine rotation −0.07→−0.23 across events) lives in this regime:
   directional rotation without norm growth.
2. **Stiff contact (k≥2.5e5, few substeps per compression)**: the
   ANALYTIC gradient norm detonates (‖g‖ ∼ 10⁷–10⁸) crossing the event
   while FD sees O(0.1–7). ε-convergence diagnostic settles *what* it
   is: ‖g_fd‖·ε is constant (~3.1–3.3e-6) across ε∈[5e-6, 4e-5] with
   ‖g_fd⟩ doubling per halving — the exact signature of a **finite jump**
   in the objective (a step Δ≈6.4e-6 gives ‖g_fd‖=Δ/(2ε), diverging
   forever; a steep-but-finite derivative would instead converge). cos
   saturates at ~0.976 rather than →1. So: the objective has a jump at
   contact events in this regime; the analytic gradient retains ~97.6%
   correct direction but its magnitude describes a boundary layer of
   width Δ/‖g_ana‖ ≈ 6e-13 m — below any physical scale in the system.
   This is the softplus knee being resolved, not contact mechanics.
   Explicit vs implicit damping makes no difference (both explode
   identically): not an integrator artifact.
3. Transition location does **not** collapse on Π₁ = √(m_eff/k)/dt
   alone: matched-Π₁ configs differ (k=2.5e6/dt=3.16e-5 is clean at the
   same Π₁ where k=2.5e5/dt=1e-4 explodes), and narrowing the softplus
   ramp β_soft 1e4→1e5 at fixed (k,dt) flips clean→rotated (Δcos
   0→0.26). Strike-speed logging added; group search continues.

Working synthesis (revised after the jump finding): the stiff-regime
"explosion" is the analytic gradient differentiating a **numerical
smoothing artifact** — the objective contains a finite jump created by
the softplus knee, and BPTT honestly reports the knee's inverse width.

**Π_ramp collapse (preliminary, n=10 configs).** With strike speed vₙ now
measured per event, the dynamic group

    Π_ramp = ε_ramp / (vₙ · dt)     [substeps spent traversing the ramp]

orders every post-fix measurement with a transition at **Π_ramp ≈ 2.5 ± 0.4**:

| Π_ramp | cos_B | outcome |
|---|---|---|
| 8.44, 4.95, 2.61, 2.20 | 1.000 | analytic == FD |
| 2.50 | 0.705 | norm explosion |
| 1.28 | 0.657 | rotated |
| 0.88, 0.26 | 0.00–0.66 | destroyed |

(vₙ varies 0.12–1.21 across configurations — it is an emergent quantity of
each gait/timing, which is why it must be measured, not assumed.) One
coarse-dt row sits below threshold yet clean; its vₙ estimate spans a full
400 µs substep and is unreliable. Two length-scale constraints bracket
ε_ramp jointly:

    static:   δ_pen = mg/(n·k)  ≫  ε_ramp      (operate off the knee)
    dynamic:  ε_ramp / (vₙ·dt)  ≫  1            (several substeps across it)

yielding a closed-form **maximum usable contact stiffness** for a
differentiable simulator,

    k ≪ mg / (n · vₙ · dt),

with ε_ramp optimally near the geometric mean √(δ_pen · vₙ · dt).
Practitioner rule: if your task's impact speeds and timestep put you below
Π_ramp ≈ 3 at your chosen k, first-order gradients through events are
untrustworthy regardless of integrator.

### Physics correctness

| Property | Verified how | Result |
|---|---|---|
| Mass matrix positive definite | `eigvalsh(M)` checked every substep for 1000 steps | min eig > 0 always |
| Mass matrix exactness | vs independent FK-geometry kinetic energy reference | rel err 3.8e-9 |
| Coriolis exactness | Christoffel identity vs FD of M(q) at random states | max diff 1e-10 |
| Momentum conservation | P_lin / m == v_com to machine precision | err < 1e-16 |
| COM acceleration invariant | gravity-only free fall gives exactly [0, 0, g] | exact |
| Energy bounded | 2000-step dynamic rollout, no persistent injection | max transient 1.3 kW, decays |
| Gradient fidelity per-env | AD vs central-FD, E=16 envs individually, H=32 | 16/16 envs: cos=1.000000, worst rel_err=5e-5 |

### Stability analysis

| Measurement | Value | Interpretation |
|---|---|---|
| Benettin λ (standing, PD-controlled) | **−2.87 ± 0.03 /s** | asymptotically stable fixed point |
| δ₀-invariance | −2.870 and −2.876 for δ₀=1e-8 and 1e-11 | accumulator correct; not tracking log(δ₀) |
| Analytic unstable rate √(mgd/I) | **+3.36 /s** | bare inverted pendulum without control |
| Sign flip (uncontrolled→controlled) | +3.36 → −2.87 | PD stabilization works correctly |
| k_ground dependence | invariant across decade (2.5e3–2.5e5) | contact spring not dominant instability |
| Chaotic double pendulum λ₁ | **+0.09 ± 0.02 /s** (dt=2e-4, T=60 s, released-from-rest ICs) | estimator detects real chaos ✓ |
| Estimator δ₀-invariance | spread 0.000 across {1e-6, 1e-9} | ✓ |
| Integrator dissipation bias | λ₁ = +0.047 (dt=5e-3) → +0.075 (2e-3) → +0.116 (1e-3) → ~+0.09–0.10 (≤5e-4) | coarse-dt semi-implicit Euler suppresses chaos; exponents are properties of the simulated map |

**Correction history**: an earlier table entry claimed chaotic λ₁ = +0.80 /s.
That number is not reproducible under the post-`b0fb030` accumulator and is
attributed to the pre-fix accumulation bug. Any document citing +0.80/s
should be considered superseded by this table.

### Full Lyapunov spectra inside AD (`diffsim.lyapunov_spectrum`)

New capability: the complete spectrum λ₁…λₙ from forward-mode tangent
propagation (`torch.func.jvp`, exact Jacobian action — no finite
differences) with QR-reorthonormalized Benettin accumulation.

| Check | Result |
|---|---|
| One-step Jacobian, AD vs central-FD | rel err 1.6e-10 |
| Chaotic double pendulum spectrum | [+0.11, +0.08, −0.00, −0.21] (dt=5e-3); λ₁ matches independent FD-Benettin within finite-time error |
| Hamiltonian structure | symmetric spectrum, Σλᵢ = −0.015 ≈ 0 (O(dt) volume bias), seed-invariant |
| Damped equilibrium vs discrete-map eigenvalues | pair-means match log\|eig(J_step)\|/dt to <0.01 |
| Sum rule | Σλᵢ = −12.72 vs (⟨tr J⟩−n)/dt = −12.64 ✓ |
| **Chaotic impact oscillator** (bouncing ball on vibrating table, Γ≈2) | spectrum [+1.01, +0.05, −1.10]; λ₁ matches FD-Benettin; **tangents cross impacts with no saltation correction** |
| Differentiable spectrum dλ/dc | autodiff −0.174966 vs central-FD −0.174979 |

Implementation notes (negative results worth remembering):
* Cotangent propagation `W ← JᵀW` with LQ filtering computes *adjoint*
  exponents — NOT the Lyapunov spectrum for non-normal maps (measured
  σ_max(J)=1.11 → σ_max(J⁵⁰)=2.45 despite ρ(J)<1: transient growth).
* Reverse-mode through `torch.func.jvp`/`jacrev` produces NaN grads;
  `lyapunov_spectrum_diff` instead builds each J column-by-column with
  plain `autograd.grad(..., create_graph=True)` so dλ/dθ flows exactly.

### Independent compass-walker oracle (`scripts/walker_oracle.py`) [Q1c]

A third, fully independent implementation of passive-dynamic-walking
physics: sympy-derived Euler–Lagrange flow + Newtonian 8×8 impulse solve
for plastic heelstrike (old pin releases; rods transmit axial impulses
only).  Shares morphology parameters with the DiffSim twin via
`diffsim.walker.WALKER_P`; slope realized as tilted gravity (Galilean
equivalence — zero collision-code changes).

| Anchor | Result |
|---|---|
| Acceleration field vs published Garcia eqs (1)–(2), O(β) map | err 8.9e-4 @ β=1e-4 → 8.9e-6 @ β=1e-6 (**exact linear scaling**, ratio 0.010 vs expected 0.01) |
| Flow energy conservation | drift < 1.7e-11 over 3000 RK4 steps |
| Impact: angular momentum about new pivot | conserved to machine precision (−2.2827481800e+00 both sides) |
| Impact: kinetic energy | ratio post/pre = 0.834 (strictly dissipative, plastic) |
| Period-one FP @ γ=0.009 vs mapped Table-1 asymptotics | θ*, ω* match to 1.9e-4 / 4.3e-4; ρ=0.58 attracting |
| Stride closure from mapped FP | closes at τ=0.8935 s (=3.957 dimensionless vs paper 3.88) |
| **β-convergence of stride period** | β=0.05: 3.957 (2.0% off); **β=0.02: T=0.875 s vs 3.88·√(l/g)=0.876 s — 0.1%**. The residual at β=0.05 was pure finite-β: θ*, ω*, AND τ now converge together, anchoring the O(β) story across three independent quantities. |

Documented subtleties (each caught by a physical check):
1. Garcia's rates are DIMENSIONLESS (τ=t√(g/l)); seeding without
   rescaling velocity lands inside the inverted-pendulum potential well.
2. Coordinate map is θ_G = −θ_world (their stance angle decreases);
   verified numerically, not by convention.
3. β→0 is a singular perturbation: det M→0, swing-rate coordinate is
   slaved differently than at finite β (d₂* not comparable across the
   limit; θ*, ω*, τ, multipliers are).
4. Rigid rod constraint: endpoint relative velocity ⊥ rod — written
   along the tangent once, which silently turned legs into prismatic
   joints (energy-increasing impacts; caught by L-conservation check).

### Garcia bifurcation diagram reproduced end-to-end [Q1 anchor]

`benchmarks/garcia_oracle.json` + `garcia_cascade_refine.json`:

| γ | attractor | published anchor |
|---|---|---|
| ≤0.013 | stable period-1 (double multiplier 0.57→0.63) | stable for γ<0.0151 ✓ |
| 0.014 | multiplier split 0.773/0.406 | — |
| 0.0145 | largest \|μ\|=0.918 → flip imminent | limping near 0.017 ✓ |
| 0.015–0.017 | **period 2** (two-point Poincaré) | limping/cascade region ✓ |
| 0.0176 | **period 4** | cascade ✓ |
| 0.0178 | **period 8** | Feigenbaum accumulation ✓ |
| 0.018 | **chaotic** (spread 0.023) | cascade complete ~0.019 ✓ |
| ≥0.019 | falls | walker falls above cascade ✓ |

Matched-precision bisection of the onset slopes (criterion: Poincaré
point-count change, ≤1e-5 brackets, `benchmarks/garcia_feigenbaum.json`):
γ₂=0.014699, γ₄=0.017268, γ₈=0.017748 →

    δ₁ = (γ₄−γ₂)/(γ₈−γ₄) = 5.35 ± 0.16

Published first accumulated ratio: 5.9. **That is a 3.4σ gap and we do
not claim agreement.** Measured on oracle-A (β→0 paper-faithful), so
finite-β is not an available explanation; live hypotheses are (a) onset
criterion differs from the paper's, (b) the published ratios are
themselves approximate. Discriminator: δ₂ = (γ₈−γ₄)/(γ₁₆−γ₈), which must
approach 4.669 from above — if δ₂ ≈ 4.8–5.0 the sequence is converging
correctly and the δ₁ gap is definitional; if it is off-trend the onset
criterion has a real problem. γ₁₆ bisection running
(`benchmarks/garcia_feigenbaum_d2.json` when done).

### Walker hybrid-map Lyapunov spectrum [Q5 × Q1c]

FD-Benettin through the validated sympy flow + Newtonian impact map
(3-dim section, `scripts/walker_lyapunov.py`):

| Check | Result |
|---|---|
| Spectrum @ γ=0.009, β=0.02 | λ = [−1.216, −1.235, −21.82] /s, T_stride=0.875 s |
| λ₁ vs independent FD multipliers | ln(ρ)/T = **−1.2248** vs Benettin **−1.216** (0.7%) |
| Structure | two slow exponents ≈ equal (gait contraction), third fast/negative = light-foot slaving |

Tangents propagate through heelstrike events with no saltation
correction, extending the earlier bouncing-ball finding to
multi-DOF hybrid locomotion.

### Compliant twin (DiffSim): current empirical status

**Headline (bounded result).** Compliant point-foot passive walking
requires the static penetration δ_pen = mg/(n·k) to drop below the
swing-clearance budget — i.e. k ≳ mg/(n·h_clear) ≈ 1e5–1e6 N/m for this
machine. At k = 4e5–1e6 N/m (implicit damping, μ=0.9, true slope), the
twin achieves liftoff and completes **one stride** from an oracle-derived
mid-stance seed: +0.53 m travel, 19–20 mm swing clearance, dt-converged
to 4 decimals across dt ∈ {5e-5, 2e-5} (`scripts/twin_strike_mechanism.py`,
`benchmarks/twin_strike_mechanism.json`). The stride does not repeat:
after touchdown the machine stands in double support ~0.8 s, then slides
into a fall. **No self-sustaining orbit found at any stiffness tested**
(k ≤ 4e5: zero net advance at every dt; random ICs never enter the
walking basin even at k=1e6 — only the oracle-derived seed does).

Mechanism, quantified: across the compliant strike window the new swing
leg retains only ~50% of the rigid-orbit backswing rate (+0.386 vs
+0.71 rad/s) because the old foot scrubs angular momentum while it
unloads; what survives decays to zero within 135 ms and the backswing
stalls at ~1 mm tip rise vs the rigid orbit's 5 mm.

**Coverage disclosure.** An earlier degrees/radians bug (`slope_gravity`
took degrees, callers passed radians) silently ran every twin simulation
before commit `1983c71` at effective slope γ_eff ≈ 9×10⁻⁵ rad — flat
ground. All scans from before that commit are **invalidated as slope
measurements** and were discarded. What follows is the post-fix record
only.

**Post-fix scanned to date** (point feet r=1 mm, foot mass βM=0.02,
semi-implicit Euler):

| slopes | k [N/m] | b [Ns/m] | μ | ICs/config | result |
|---|---|---|---|---|---|
| 0.009–0.015 | 2.5e4 | 400 | **3** | 64 | falls + rocking-in-place |
| 0.012 | 2.5e4–1e5 | 60–200 | 3 | 48 | falls + rocking-in-place |
| 0.015, 0.022, 0.028 | 2.5e4–4e5 | 120–400 | 3 | 48 | falls + rocking-in-place |

No passive walking orbit found in that set. The attractors are (a)
falls within 1–2 steps and (b) a two-foot rocking-in-place equilibrium
(legs ±0.11 rad, both feet penetrating ~2 mm, alternating micro-contacts,
zero net advance).

**What is NOT yet covered post-fix** (and therefore not claimable):
any μ < 3 (physical rubber is 0.9–1.0); the steeper powered regime
predicted by the energy budget below; implicit-damping runs at foot
masses near Garcia's β→0 limit (now unblocked by the implicit damper);
k ≥ 1e6.

**Energy-budget hypothesis: REFUTED (2026-08-24, session 4).** The
falsifiable form predicted walking near γ ≈ 0.013 (= 0.009/(1−f), f≈0.29).
Post-fix scan at μ=0.9 with implicit damping (E=48 ICs × γ∈{0.009…0.018})
found no walking anywhere — `benchmarks/twin_prediction_scan.json`. The
dissipation-shift story is wrong. Combined evidence now points at a
structural blocker instead: seeds placed mid-swing complete one clean
stride, but the new trailing foot never re-initiates swing after a strike
(max observed clearance ~0.5 mm). The failure is liftoff geometry, not
power margin — which is why arc feet (rolling toe-off) and the
minimum-actuation measurement are the next two discriminators.


### Gradient fidelity horizon sweep

BPTT vs central-FD, matched E=64 batch, fd_eps plateau verified:

| H | ‖g_bptt‖ | ‖g_fd‖ | cosine | rel_err |
|---:|---:|---:|---:|---:|
| 8 | 2.3482 | 2.3482 | 1.00000 | 0.0000 |
| 32 | 2.3371 | 2.3371 | 1.00000 | 0.0000 |
| 64 | 2.4508 | 2.4508 | 1.00000 | 0.0000 |
| 128 | 2.9993 | 2.9993 | 1.00000 | 0.0000 |

Per-env distribution (E=16, H=32): median cos = 1.000000, worst = 1.000000,
worst rel_err = 5e-5 across all 16 environments individually.

### GPU validation [RTX 3090, CUDA]

fp32 and fp64 produce **identical** com_z trajectories on CUDA (100 control
steps, standing humanoid): `fp64=[0.8629...0.8616]`, `fp32=[0.8629...0.8616]`.
Throughput is currently ~500 env-steps/s at E=256 due to Python-loop kernel-
launch overhead (~200 launches × 8 substeps per control step). This is NOT
compute-bound — fp32≡fp64 timing confirms it. Fix requires vectorizing
body/dof loops into padded tensor operations.

## Known limitations

| Limitation | Impact | Status |
|---|---|---|
| No external ground truth (walker bifurcation diagram) | Every validation is self-referential | **Next priority** |
| Standing settles into crouch equilibrium | PD with finite gain cannot hold zero joint angles against gravity — expected physics, not a bug | Statics computed; task design issue |
| CPU fp64 throughput ~13k env-steps/s | Too slow for locomotion training | torch.compile + fp32 + CUDA queued |
| MJCF loader incomplete | Can't load real SOMA asset yet | Skeleton exists |
| Damped-equilibrium Benettin gap | Discrete-map exponent ≠ continuous-flow eigenvalue (4× at dt=1e-4); both negative → stable ✓ | Known integrator-vs-flow difference |

## Architecture

```
diffsim/
├── linalg.py        # batched quaternions, exp/log maps
├── spatial.py       # Featherstone spatial algebra
├── articulation.py  # FK, world-frame CRBA, Lagrangian-identity Coriolis,
│                    #   closed-form fast path, quaternion free joints
├── collision.py     # exact sphere/capsule/plane distances, softplus contact
├── sim.py           # DiffSim engine: batched substeps, contact J^T f via
│                    #   closed-form subspace Jacobians, limits, damping, PD
├── humanoid.py      # SOMA-class humanoid builder (21 bodies, 15 actuated)
├── lyapunov.py      # Benettin estimator (validated), delta0-invariance check
├── mjcf.py          # minimal MJCF XML parser (skeleton)
└── algo/shac.py     # short-horizon actor-critic trainer
```

## Usage

```python
import torch
from diffsim.humanoid import make_soma_humanoid, initial_pose
from diffsim import build_geoms_compat
from diffsim.sim import DiffSim, SimConfig, ContactConfig

model, gspec, feet = make_soma_humanoid()
sim = DiffSim(model, build_geoms_compat(gspec),
              SimConfig(dt=5e-4, n_substeps=8), dtype=torch.float64)

q, w = initial_pose(model, E=4096)
for _ in range(250):
    tau = sim.pd_torques(q, w, q_target, kp=400., kd=50.)
    r = sim.step(q, w, tau_ext=tau, train_mode=False)
    q, w = r.q, r.w if hasattr(r, 'w') else r.qd  # differentiable if train_mode=True
```

## Requirements

PyTorch ≥ 2.0 (CUDA build recommended). No compiled extensions.
fp64 recommended for variational work; fp32 for RL training on GPU.

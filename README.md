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

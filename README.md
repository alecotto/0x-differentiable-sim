# 0x-differentiable-sim

**A fully differentiable, GPU-accelerated articulated-body simulator for
humanoid policy learning and variational analysis of contact dynamics.**

Everything in the physics pipeline — forward kinematics, mass matrix,
Coriolis/gravity forces, collision detection, contact forces, integrator —
is implemented as batched PyTorch tensor operations. Gradients flow exactly
through arbitrary-length rollouts (BPTT), and an arbitrary number of
parallel environments `E` rides the leading dimension of every tensor, so
the same code runs CPU-fp64 for validation and CUDA for scale.

---

## Status: physics core validated

| Component | Validation |
|---|---|
| Mass matrix (CRBA, world-frame) | exact vs independent FK-geometry kinetic energy (`rel 3.8e-9`); vs brute-force `Σ JᵀIJ` (`1.6e-14`); `torch.autograd.gradcheck` ✓ |
| Gravity generalized forces | analytic pendulum torque to machine precision ✓ |
| Coriolis/bias forces | Lagrangian identity vs Christoffel FD reference ✓; energy conservation in *chaotic* double-pendulum regime ✓ |
| Floating-base dynamics | system COM acceleration ≡ `[0, 0, g]` under gravity-only free fall ✓; per-body momentum `P_lin = m·v_com` to `1e-16` ✓ |
| Joint subspaces S(q) | every row's velocity field matches finite-differenced FK over its subtree ✓ |
| Contact | stable PD-controlled standing of a 21-body / 15-DoF humanoid; free-fall physically exact |

## Architecture

```
diffsim/
├── linalg.py        # batched quaternions, exp/log maps — pure differentiable ops
├── spatial.py       # Featherstone spatial algebra (6D inertia, crosses)
├── articulation.py  # Model + Articulation: FK, world-frame CRBA,
│                    #   bias via Lagrangian identity + exact autodiff
│                    #   (jvp through CRBA + reverse-mode KE gradient)
├── collision.py     # exact sphere/capsule/plane distances, smooth_ramp contact
├── sim.py           # DiffSim engine: batched substeps, contact J^T f via
│                    #   closed-form subspace Jacobians, limits/damping, PD
├── humanoid.py      # SOMA-class humanoid builder (21 bodies, 15 actuated dofs)
└── build.py         # geom spec helpers
tests/               # energy conservation, Christoffel identity, momentum checks
benchmarks/          # throughput scaling (WIP)
```

### Design decisions that make it fast *and* exact

1. **World-coordinate formulation.** All subspaces, inertias, and twists are
   expressed about the world origin. Composite subtree inertias become plain
   sums and CRBA "propagation" becomes identity — no transform bookkeeping.
2. **Contacts as Jacobian-transpose point forces.** `tau_c = Σ Jₖᵀ fₖ` with
   `J = ∂p/∂q` in closed form from the joint subspaces — no nested autograd.
3. **Smooth everywhere.** Contact activation uses a centered smooth ramp
   (zero value *and* zero force at touch); friction is a regularized viscous
   Coulomb cone; normal damping is approach-velocity-capped. Gradients exist
   at every state.
4. **Autodiff as ground truth.** The Coriolis vector is computed by
   differentiating the validated mass matrix (`h = Ṁq̇ − ∇_q KE`) rather than
   hand-derived gyroscopic formulas — this eliminated an entire class of
   world-frame spatial-algebra sign bugs. A closed-form RNEA that matches
   this reference is the top performance TODO.

## Research roadmap

This simulator is the substrate for a three-track research program on
long-horizon gradients through contact:

- **Phase 0 — the instrument** *(next)*: saltation matrices across contact
  events, QR-reorthonormalized tangent batching `[E, nv, k]`, Lyapunov
  spectra, covariant Lyapunov vectors, and **Gate 1: measurement of `m_us`**
  (dimension of the unstable subspace) for humanoid locomotion.
- **Track A**: saltation-corrected shadowing (NILSS/NILSAS-style) for O(1)
  long-horizon policy/morphology gradients where BPTT explodes.
- **Track B**: ensemble/Ruelle response sensitivity — differentiate the
  invariant measure directly using massive GPU ensembles.
- **Track C**: implicitly-differentiated convex contact solver as a Newton
  backend.

Testbed ladder before humanoid claims: bouncing ball on wavy floor →
passive dynamic walker (published bifurcation diagrams as ground truth) →
humanoid.

## Measured: end-to-end neural-network training (SHAC-lite)

`scripts/train_balance.py` trains a 34→128→128→15 MLP policy (residual on
PD targets) with short-horizon actor-critic: BPTT through H=32 steps of
simulation + TD(λ)-bootstrapped critic, fp64, batched E=64 envs, with a
push-magnitude curriculum and joint-limit-respecting initialization.

* Training return J rose **20 → ~78 (+270%)** over 60 iterations under a
  rising perturbation curriculum; **zero falls** across all 60×64 env-rollouts.
* One more real bug found by training itself: advantage-style normalization
  of returns zeroes SHAC gradients near equilibrium (batch-mean subtraction
  cancels the shared θ-dependence — verified by bisection; see git history).
* Checkpointing/resume supported (`models/shac_balance.pt`).

**Honest status**: at the evaluated push regime the *fixed* PD baseline also
survives 100%, so the learned residual shows parity, not yet superiority.
Demonstrating a learned advantage requires task headroom: sustained random
force sequences, crouched/tilted starts, or COM-recentering objectives.
Throughput: ~50 s/iteration (E=64×H=32) on CPU fp64 — GPU/fp32/compile is
the obvious next performance step.

## Measured: the differentiable-sim gradient tradeoff

BPTT gradients of a stabilization cost w.r.t. 15 PD-target parameters,
checked against central finite differences (`scripts/probe_gradients.py`,
fp64, standing humanoid with contacts):

| Horizon H | ‖g_bptt‖ | ‖g_fd‖ (truth) | cosine | rel. error |
|---:|---:|---:|---:|---:|
| 8   |   25.7 |   25.7 | 1.00000 | **0.0000** |
| 32  |  149.2 |  149.2 | 1.00000 | **0.0000** |
| 64  |  837.2 | 1308.0 | 0.88749 | 0.5231 |
| 128 | 1856.0 | —      | —       | — |

**Findings**: analytic gradients are machine-exact through ~32 control
steps (0.13 s), then diverge from the FD truth (52% magnitude error,
cos 0.89 by H=64) while the norm grows ~6×/doubling — the Lyapunov-type
signature of chaotic contact dynamics. This is precisely why training uses
short-horizon actor-critic (gradients only where they are exact, value
bootstrap beyond), and why Phase 0's shadowing instrument matters.

## Usage sketch

```python
import torch
from diffsim.humanoid import make_soma_humanoid, initial_pose
from diffsim import build_geoms_compat
from diffsim.sim import DiffSim, SimConfig, ContactConfig

model, gspec, feet = make_soma_humanoid()
sim = DiffSim(model, build_geoms_compat(gspec),
              SimConfig(dt=5e-4, n_substeps=8), dtype=torch.float64)

q, qd = initial_pose(model, E=4096)          # [4096, 22], [4096, 21]
for _ in range(250):
    tau = sim.pd_torques(q, qd, q_target, kp=80., kd=10.)
    r = sim.step(q, qd, tau_ext=tau, train_mode=False)
    q, qd = r.q, r.qd                        # differentiable if train_mode=True
```

## Known limitations (honest list)

- **Throughput**: the autodiff-based bias term costs ~0.1 s/env-step on CPU
  fp64. Correctness first; the closed-form RNEA (validated against this
  reference), `torch.compile`, and fp32/CUDA are queued.
- Witness-point switching on tilting capsules is discontinuous (mitigated by
  velocity caps); implicit contact solve is Track C.
- MJCF/URDF asset loading not yet wired (procedural humanoid is the default).
- Test suite takes ~5 min (AD-heavy); will drop sharply with the RNEA path.

## Requirements

PyTorch ≥ 2.0. No compiled extensions. `float64` recommended for variational
work, `float32` for RL throughput on GPU.

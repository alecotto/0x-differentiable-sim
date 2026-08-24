# RESEARCH LOG — agent handoff state

**Purpose: if a chat thread dies, the next agent reads THIS FILE FIRST and
resumes. Update it after every milestone, commit, and background-job
status change. Keep it honest: record failures and dead ends with the
same care as successes.**

Last updated: 2026-08-24 (session 3, walker-twin phase)

---

## Mission (from project owner)

Work through 5 science questions for the differentiable simulator,
simplest proofs first:

| # | Question | Status |
|---|---|---|
| Q1 | Does our contact model produce correct physics under repeated dynamic impacts? (Garcia walker bifurcation validation) | **IN PROGRESS** — oracle done; twin search running |
| Q2 | Are exact gradients *useful* for training high-contact robots? | Baseline infra done (SHAC-lite + PPO-lite); locomotion task blocked on Q1c |
| Q3 | Optimal contact stiffness for gradient-based training (Pareto frontier) | Not started; blocked on walking twin |
| Q4 | Sim-to-real transfer of smoothed policies (MuJoCo comparison) | Not started; needs trained policy |
| Q5 | AD-based Lyapunov spectra via `torch.func.jvp` QR-Benettin | **COMPLETE** (commit 3468c14); extension to walker limit cycle pending |

## Repo layout quick reference

- `diffsim/` — engine (articulation.py = dynamics core; sim.py = DiffSim
  engine w/ softplus contact; collision.py; humanoid.py; walker.py =
  Garcia-style twin builder; lyapunov*.py)
- `scripts/garcia_oracle.py` — paper-faithful β→0 simplest-walker oracle
  (Poincaré map, fixed points, attractor classification). VALIDATED.
- `scripts/walker_oracle.py` — independent sympy compass-walker oracle at
  finite β ("oracle B"). VALIDATED (see anchors below).
- `scripts/walker_twin.py` — DiffSim soft-contact twin harness (Q1c).
- `scripts/walker_twin_search.py` — batched (k,b) × rate-scale walk search.
- `scripts/impact_metrics.py` — Q1d impact metrics harness. DONE.
- `scripts/train_balance.py` — SHAC-lite push-recovery training. DONE.
- `tests/test_walker_oracle.py` — 5 tests, ALL PASS (~96 s).
- Full suite: `python -m pytest tests/ -x -q` (~28 min).

## SESSION 4 FINAL STATE (wrap-up)

### Headline results this session
1. **Implicit contact damping** shipped (flag-gated, commit 1c442cb):
   frozen-coefficient (M+dt R^T B R) solve. Removes explicit-Euler
   constraint dt << m_eff/b. Accuracy validated vs fine-explicit.
2. **Feigenbaum delta_1 = 5.346** from matched-precision bisection
   (<=1e-5 brackets, Poincare point-count criterion):
   g2=0.014699, g4=0.017268, g8=0.017748. Published accumulation
   5.9 -> 4.67. Garcia cascade now QUANTITATIVELY reproduced
   (benchmarks/garcia_feigenbaum.json).
3. **Energy-budget prediction REFUTED**: no walking at predicted
   gamma=0.013 (or 0.009..0.028) at mu=0.9 post-fix. Refined mechanism:
   post-strike liftoff failure (structural), not power margin.
4. **beta=0.001 twin scans**: no walking in Garcia's own foot-mass
   regime either — negative result now covers the published morphology.
5. **Delta-cos sweep (Q2a proof) — first campaign complete, picture
   revised twice**:
   - Norm explosion at stiff/few-substep contact is REAL and PHYSICAL:
     eps-convergence shows FD norms GROW toward the analytic value
     (cos->0.976); the stride map's true Lipschitz constant detonates
     (rebound-timing amplification). Explicit-vs-implicit control: no
     difference.
   - "Soft contact is safe" also FALSE in general: soft contact at
     coarse dt explodes too (||g||=6e47 at 2.8 substeps/compression,
     k=2.5e4, dt=1e-3).
   - Pi1 = sqrt(m_eff/k)/dt alone does NOT collapse the data (matched-
     Pi1 pair differs). Narrowing softplus ramp beta_soft at fixed
     (k,dt) flips clean -> rotated: ramp-width group confirmed as a
     second axis.
   - Third mechanism observed: touch-timing bifurcation (event enters/
     leaves perturbed windows) gives FD>>analytic with perfect floor —
     needs v_n logging + window-edge analysis next session.
   - Current best account: BOTH norm explosion AND direction rotation
     are real, regime-dependent; transition governed jointly by contact
     resolution and ramp width; strike speed not yet controlled.

### Next session priorities (in order)
1. Delta-cos: add strike-speed (v_n) logging; re-attempt collapse with
   groups (Pi1, Pi_ramp, v_n); analyze touch-bifurcation rows
   (FD>>analytic with clean floor). Decide final claim language.
2. Minimum-actuation experiment on the twin ("how much hip torque
   restores walking?") — now well-motivated by the liftoff-failure
   finding; measure threshold as a number.
3. Arc feet if (2)'s threshold is large.
4. Q3 frontier table: the dcos stiffness axis already produces
   fidelity-vs-stiffness points; combine with horizon data.

### Background threads
- feig2 DONE (results committed). No other jobs running.

## SESSION 4 (feedback-driven)

### Feedback review committed as: implicit damping (1c442cb), process fixes
(58f6e48), README honesty rewrite + conditioning-not-magnitude elevation +
beta column (0e448e5).

### Implicit contact damping — DONE, validated
Flag `ContactConfig.implicit_damping` (default False): normal damper
integrated via frozen-coefficient solve (M + dt R^T B R) qd+ = M qd_euler.
- accuracy at dt=1e-4 in stable regime: rel err vs fine-explicit(2e-5)
  reference 0.142 (implicit) vs 0.154 (coarse explicit) — physics kept
- light-foot regime (beta=0.001): explicit transient |w|=38.8, implicit 3.7
- existing tests green with flag off

### ENERGY-BUDGET PREDICTION REFUTED (important)
Prediction: walking near gamma_twin = gamma_rigid/(1-f) = 0.009/0.71 =
0.0127 (f=0.29 measured single-strike dissipation fraction).
Result (mu=0.9 post-fix, implicit damping, E=48 ICs x gamma in
{0.009,0.011,0.013,0.015,0.018}): NO walking anywhere
(benchmarks/twin_prediction_scan.json). The dissipation-shift story is
WRONG as stated.

Refined mechanism hypothesis (from combined evidence): mid-swing seeds DO
complete one clean stride (15mm swing clearance achieved); the failure is
POST-STRIKE RE-INITIATION of swing — the new trailing foot never lifts
(max observed clearance ~0.5 mm before settling back into the rocking
attractor). Structural/geometric blocker, not a power margin. This is why
arc feet (geometric toe-off) are the natural next morphology, and why
minimum-actuation is the right Q2 framing.

### beta=0.001 twin scans LAUNCHED (unblocked by implicit damping)
/tmp/opencode/beta_scan.log — first runs in Garcia's asymptotic foot-mass
regime ever possible in this engine.

### Feigenbaum bisection v2 running: matched <=1e-5 brackets, criterion =
Poincare point-count (documented in script header). /tmp/opencode/feig2.log

### Done & pushed
1. Garcia oracle-A full bifurcation diagram + cascade refinement
   (benchmarks/garcia_oracle.json, garcia_cascade_refine.json):
   period-1 -> 2 (0.015) -> 4 (0.0176) -> 8 (0.0178) -> chaos (0.018)
   -> falls (>=0.019). Matches published structure at every stage.
2. Walker oracle-B (sympy+Newtonian impulse) validated: O(beta)
   acceleration anchor, exact impact momentum conservation, FP matches
   Table-1, stride closure. tests/test_walker_oracle.py 5/5 PASS.
3. Q2a gradient-transition diagnostics: no grad spikes at contact
   events (norm ratio 0.99); cosine shift -0.07 -> -0.23.
4. Twin negative result (thorough): NO compliant point-foot passive
   walking across gamma in [0.009,0.028], k in [2e4,4e5], b in [60,
   2500], mu in {0.9,3}; attractors = falls + rocking-in-place.
   Energy budget arithmetic explains it (strike+drag losses vs slope
   input). Data in benchmarks/twin_*.json.
5. Q5 x walker: hybrid-map Lyapunov spectra via FD-Benettin through
   sympy flow + Newtonian impact; lambda_1 cross-validated against
   independent FD multipliers at two betas (-1.216 vs -1.2248 @beta=0.02;
   -0.615 => rho=0.583 matching beta=0.001's 0.58). No saltation
   correction needed, extending the bouncing-ball finding to
   multi-DOF hybrid locomotion. benchmarks/walker_lyapunov.json.

### Still running (background; check ps aux)
- feig.py bisection of cascade onsets (hours): /tmp/opencode/feig.log

### Next session priorities
1. Feigenbaum delta from bisection results when done; compare to
   published accumulation 5.9/5.2/4.6.
2. Twin escape axes: arc feet r>=25mm (needs oracle rolling-contact
   rework), OR actuated walking (bridges directly to Q2 training),
   OR k>=1e6 with proportional damping (dt<=2e-5, expensive).
3. If arc feet adopted: extend oracle-B with rolling stance kinematics
   before any twin comparison.
4. Q2 SHAC locomotion attempt can proceed on a slightly-actuated twin
   (small hip torque budget) even without passive orbit — actually
   PREFERRED next step: turns the negative result into the Q2
   experiment design ("how much actuation restores walking?").

### CRITICAL BUG FIXED (earlier this session): degrees/radians slope
`walker.slope_gravity()` took DEGREES; every twin script passed RADIANS
=> ALL prior twin simulations ran at gamma_eff = 9e-5 rad (flat!).
Explains: no-walking everywhere, slope-independent attractor scans,
braced double-support traps (flat ground has no downhill energy input).
Fixed: slope_gravity now takes radians; call sites patched
(scripts/walker_twin*.py, twin_gait_shoot.py).
=> The (k,b) contact-parameter search MUST be redone at true slopes;
the design space is OPEN again, not exhausted.

### Current runs
- walker_twin_scan.py (corrected gravity): scanning gamma {0.009,0.012,
  0.015} x 64 random ICs, T=3s. First two slopes: no sustained walking
  yet (steps-then-fall dominant) at DEFAULT contact params (k=2.5e4,
  b=400, mu=3). Log /tmp/opencode/scan2.log.
- cascade_refine (PID may be long-running): period-4 hunt near gamma
  0.0172-0.0178 for Feigenbaum ratio.

### COMPLETED: Garcia oracle-A bifurcation sweep (Q1 external anchor)
`benchmarks/garcia_oracle.json` — full diagram:
- stable period-1 through γ=0.013; double multiplier splits at 0.014
  (0.773/0.406), largest reaches +0.918 at 0.0145 → flip imminent;
- attractors: period-1 @0.010, period-2 @0.015–0.017, CHAOS @0.018,
  falls @≥0.019. Matches published: limping ~0.017, cascade done by
  ~0.019. Cascade refinement (period-4 hunt for Feigenbaum ratio) in
  background log /tmp/opencode/cascade_refine.log.

### Walker twin gait shooting (the active front)
First shooting runs converged to a DEGENERATE tumble solution (legs lock
parallel, whole machine tips like a compass needle — periodicity alone
doesn't exclude it). Added anti-tumble penalties: leg-angle range
(|θ|≤0.5 sampled along trajectory) + forward-advance requirement.
Currently running 3 gammas {0.009,0.013,0.017} × horizons
{7000,8000,9000} steps @ dt=1e-4, logs /tmp/opencode/shootg_*.log.

Key parameterization facts (do not regress):
- periodicity = COMPONENTWISE SWAP (θa_N=θb₀ etc.), no negation — the
  validated oracle relabel map is a plain component swap;
- start pose must be NON-mirror (exact mirror ⇒ both feet touching ⇒
  braced four-bar, angles freeze);
- hip z starts at l·cos(θa)+r−δ with δ=mg/k sag;
- one-step grads healthy; long-horizon via checkpoint chunks of 200.

### If all γ fail to find walking orbits
Fallbacks in order: (1) multiple shooting (nodes every ~600 steps,
closure via swap); (2) arc feet r=25mm (needs oracle rolling-contact
rework — big); (3) accept negative result with thorough search evidence:
"no compliant point-foot passive walking at humanoid-default contact law
parameters" + characterize the shuffle/tumble attractors found instead.

### Next steps (in order)

1. Read shooter logs; if any run reaches loss <1e-3 with real advance,
   verify by unrolled rollout (strikes + hip travel), then measure the
   orbit's multipliers by FD and compare against oracle rigid value
   (~0.58) — that closes Q1c's comparison loop.
2. When cascade refinement lands: compute Feigenbaum-style ratio vs
   published 5.9/5.2/4.6 anchors, append to benchmarks.
3. Q2a instrument done (see benchmarks/q2a_grad_transitions*.json).
   Locomotion-grade Q2 answer rides on the walker gait.
4. Q5 extension can proceed on the ORACLE side without the twin:
   Lyapunov spectrum of the rigid hybrid walk via tangent propagation
   through flow+Newtonian-impact map (differentiable linear solves).

## Previous findings (session 3, first block)

### Background jobs possibly still running (check with `ps aux | grep python`)

1. **Garcia oracle A sweep** — `nohup python -u scripts/garcia_oracle.py
   > /tmp/opencode/garcia_run.log 2>&1 &` (log may be lost if /tmp
   cleared; rerun takes ~2-3 h). Writes `benchmarks/garcia_oracle.json`
   at END only. Last observed output: stable through γ=0.0145 with
   multiplier pair (+0.918, +0.338) racing toward −1; γ=0.0150–0.0160 in
   progress (flip region converges slowly). Published anchor: stable
   period-1 ends at γ≈0.0151, cascade complete by γ≈0.019.

2. **Twin walk parameter search** — `scripts/walker_twin_search.py`,
   log `/tmp/opencode/twin_search.log`. Grid:
   k∈{2.5e4,2.5e4,1e5,2e5,4e5,1e6}, b∈{400,100,200,300,400,600},
   E=5 rate scales {0.95..1.3}, T=2.6s each. Partial results:
   - (k=2.5e4, b=400): all 5 seeds → exactly 2 strikes, no fall
     (= the "crawl/shuffle" attractor: walks one step then braces into
     double-support shuffle)
   - (k=2.5e4, b=100): mostly fell after 1 strike (underdamped impact)

### Q1c findings so far (all reproduced, none speculative)

0. **CRITICAL INSIGHT (session 3, latest)**: the rigid Garcia orbit's
   swing leg travels UNDERGROUND mid-stride (the paper's scuffing
   fiction: tip height l(cos th1 - cos th2) dips negative).  Seeding
   the compliant twin anywhere near mid-stance therefore produces an
   immediate FALSE heelstrike (~20ms in) followed by backward rocking.
   NO contact-parameter scan can fix this -- the rigid orbit simply is
   not a feasible trajectory for point feet on real ground.
   **Consequence**: the twin's own limit cycle must be FOUND, not
   seeded.  Plan: differentiable shooting -- Adam on ||P(s)-s||^2 over
   one stride with gradients through the soft heelstrike events
   (scripts/twin_gait_shoot.py).  This doubles as the Q2 demonstration
   (exact gradients through contact events doing real work).

1. **Oracle B (sympy) fully validated** against published Garcia eqs:
   acceleration field matches to O(β) with EXACT linear scaling
   (8.945e-4 @ β=1e-4 → 8.946e-6 @ β=1e-6); energy conserved 1.7e-11;
   impact angular momentum about new pivot conserved to machine
   precision; KE ratio 0.834 (dissipative); FP @ γ=0.009 matches mapped
   Table-1 to ~2e-4, ρ=0.58; stride closes τ=0.8935s vs paper 3.88
   dimensionless (=0.876s real).

2. **DiffSim twin walks ONE clean heelstrike** from oracle mid-stance
   seed (swing arcs 15mm clear, touches at θ=−0.137, timing ≈ oracle).
   Then fails: trailing foot drags through the compliant load-transfer
   window (~30ms), extra dissipation → COM never crosses stance foot →
   rocks backward into a braced double-support CRAWL attractor.

3. **Spurious standing attractor**: seeding AT the TD section (both feet
   at ground) yields immediate double-support bracing (friction μ=3
   locks both tips; 4mm static sag sinks both feet). MUST seed mid-swing
   (`walker_oracle.midstance_state`) to give the gait a chance.

4. **Contact-parameter stability constraint** (why beta was raised):
   explicit Euler contact damping b requires m_foot/b >> dt. With
   Garcia-exact m_foot=0.01kg and b=400 → dt<<25µs (infeasible). Raised
   foot mass to β·M=0.5 kg (β=0.05) so dt=1e-4 has factor ~6 margin.
   Twin and oracle share WALKER_P so comparison stays morphology-exact;
   external anchoring runs through the O(β) acceleration check.

### Next steps (in order)

1. Read full twin-search results (`/tmp/opencode/twin_search.log`);
   if NO config walks ≥3 strides, the finding is "our contact law has
   no passive-walking-fidelity window at these parameters" — then try:
   b sweep DOWN (100–50) at k≥1e5 with dt=5e-5, and r_foot 25mm arc
   feet (requires oracle rolling-contact rework — do NOT do this
   casually; it breaks the point-foot impact map).
2. When Garcia sweep json lands in benchmarks/: verify cascade anchors
   (stable <0.0151, cascade by 0.019, Feigenbaum ratios 5.9/5.2/4.6),
   commit json + update README.
3. If twin finds a walking config: build Poincaré section on BOTH sides
   (twin + oracle at same β), compare multipliers/period/θ* vs (k,b);
   that curve IS the first Q3 data point set.
4. Q2: add per-contact-event gradient diagnostics to SHAC training
   (log grad-norm/direction aligned with heelstrike events); PPO-lite
   baseline already exists (scripts/test via tests/test_ppo.py).
5. Q5 extension: Lyapunov spectrum of the walker limit cycle once a
   stable periodic orbit exists on either side (oracle side is enough
   for methodology; twin side is the novel measurement).

## Technical learnings log (bugs found & fixed this session — do not refight)

- **torch.diagonal default dims**: on batched [E,nv,nv] it defaults to
  dims (0,1), NOT the matrix. Use `torch.diagonal(M, dim1=1, dim2=2)`.
  Cost me an hour thinking the mass matrix was broken when it wasn't.
- **sympy symbol identity**: `sp.Symbol("mf")` ≠ `_mf` created via
  `sp.symbols(..., positive=True)` (assumptions are part of identity).
  Lambdify silently leaves unbound symbols → "Cannot convert expression
  to float" or expressions containing literal symbols.
- **sympy solve() eliminates qdd**: the solved expression IS the
  acceleration field. Multiplying by Minv again double-processes the
  equations (produced ±1e3 garbage accelerations).
- **Garcia rates are dimensionless** (τ=t√(g/l)): real-rate = dimless ×
  √(g/l)=4.429/s for l=0.5. Un-rescaled seeds land inside the
  inverted-pendulum potential well and stall.
- **Coordinate map between world-angle and Garcia coords is θ_G=−θ_world**
  (their θ decreases through stance). φ_G = θ2−θ1; η maps WITHOUT
  negation. Verified numerically against garcia_oracle.accel.
- **Rigid rod constraint direction**: endpoint relative velocity ⊥ rod:
  `(v_end − v_hip)·n̂ = 0`. Writing it along the tangent turns legs into
  prismatic joints (energy-INCREASING impacts; caught by L-check).
- **Impact rate extraction must use post-relabel angles**: d1+ uses
  new-stance angle (=old θ2); d2+ from released-foot relative velocity.
- **β→0 is singular-perturbation stiff**: fast swing mode ω~1/√β makes
  RK4 integration unstable at small β; compare ACCELERATIONS not
  trajectories; d₂* is not comparable across the limit.
- **Perfectly antisymmetric ICs are an invariant manifold** (θ̇ sums to
  zero forever) — never seed mirrored states for closure tests.
- **Heelstrike branch discrimination**: strike = g=θ1+θ2 crossing UPWARD
  with θ1>0>θ2; the leg-parallel near-vertical crossing (both ≈0) also
  has g crossing but is scuffing — reject by branch signs.
- **Free-joint velocity indexing**: hinge v-index = `art._vs[body]`, NOT
  `_qs` (quaternion block shifts q indices past v).
- **Compliant double-support bracing**: seeding a passive walker at the
  TD instant (both feet touching) with high μ creates a stable braced
  arch attractor — always seed mid-swing for gait studies.

## Validation status snapshot (details in README.md)

All committed work passes its tests. check_repo.py clean. GitHub synced
through afbf282 (push after every milestone).

## Conventions / owner preferences

- Commit after every milestone with descriptive messages.
- Push to origin/main frequently (owner explicitly asked).
- No comments unless they carry scientific content; docstrings explain
  physics derivations and design decisions.
- fp64 for variational work; CPU fine for oracle, twin is slow
  (~4ms/step single-env; batching over E nearly free).

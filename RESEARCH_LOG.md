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

## WHERE I AM RIGHT NOW (session 3)

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

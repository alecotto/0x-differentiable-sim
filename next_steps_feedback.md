# Feedback — through `a39b4db`

The two-bug implicit-damping catch is excellent work, and the *shape* of it is
what matters: the fix lived only inside `DiffSim.step()` while eleven script
loops called `forward_dynamics()` directly, so a flag that read as "on"
delivered zero damping — and underneath that, an inverted sign that had never
been exercised because of the first bug. Two stacked failures, one masking the
other. The knob-sanity rule caught its second bug on its second outing.

Also: the headline twin result survived regeneration with damping actually
applied. That is the first time a twin number has survived a correction cycle
rather than being replaced by one.

Now the hard part. Your committed data contradicts the session summary in two
places, and the second one closes out the Δcos thread permanently.

---

## 1. The clamp is exonerated by your own experiment

`q2a_dcos_clamp_controlled.json`:

| config | gnorm_B |
|---|---|
| `imp1_clamp1` (damping on, clamp on) | 40.383965708347546 |
| `imp1_clamp0` (damping on, clamp **off**) | 40.383965708347546 |
| `imp0_clamp1` (damping off, clamp on) | 370580125.6673318 |
| `imp0_clamp0` (damping off, clamp **off**) | 370580125.6673318 |

Toggling the clamp changes **nothing** — bit-identical in both damping
conditions, every field. The clamp never fires in this window.

So the stated causal chain — *"missing damping → violent rebound → max_vel=30
safety clamp → corrupted BPTT"* — is not what happened. Commit `41d2053`'s
message and the session summary both assert it, and the JSON committed
alongside refutes it. Correct it in the log before it propagates.

The true chain is simpler and doesn't need the clamp at all: **missing damping →
1e7× larger gradient norms.** That's the whole mechanism.

And I'd go further on interpretation. "Damping tames gradient norms 1e7×" makes
damping sound like a numerical fix. The cleaner statement is physical:
**undamped stiff contact is a genuinely ill-conditioned system** — an undamped
spring at k=1e6 has enormous sensitivity, and BPTT reports it correctly. The
gradient wasn't corrupted; the system was. That framing is defensible, matches
the data, and doesn't attribute the effect to an op that provably never fired.

---

## 2. The FD reference bug is visible in the dual-objective file, and it's decisive

`q2a_dcos_dual_objective.json`, first three rows:

| row | `gnorm_fd_A` | `gnorm_fd_A_qz` | identical | `gnorm_ana_A` | `gnorm_ana_A_qz` |
|---|---|---|---|---|---|
| 0 | 1.0001999800040196 | 1.0001999800040196 | **yes** | 1.0001999800040233 | 0.01001633260445149 |
| 1 | 1.000199980004031 | 1.000199980004031 | **yes** | 1.0001999800040233 | 0.010016499201472625 |
| 2 | 1.0019332866163442 | 1.0019332866163442 | **yes** | 1.0019311585245805 | 0.036354568820325964 |

Same on the B side, all rows.

**The FD path returns the identical gradient for both objectives, bit for bit.**
The analytic path correctly distinguishes them (1.0002 vs 0.0100 — a 100×
difference, as it should be for different readouts). The FD reference does not.
It computes one gradient and reports it under both labels.

That is the located-but-unfixed bug, and it explains everything:

```
cos_A_qz = -3.3e-13
```

You were comparing the **analytic gradient of quaternion-z** against the
**finite-difference gradient of px**. Two different functions. Of course the
cosine is zero. It was always going to be zero, at every stiffness, every
timestep, every ramp width.

Meanwhile `cos_A = 1.0` and `cos_B = 0.9999992` for the correctly-matched
objective. **The engine was never wrong.**

---

## 3. Consequence: the Δcos / Π_ramp phenomenon does not exist

This isn't another number to regenerate. The phenomenon itself was
instrumentation. Retire all of it:

- **Π_ramp ≈ 2.5 transition** — never existed. Your own `vn_frontier_sweep.json`
  confirms independently: Π_ramp swept 40.3 → 8.2 → 4.1 → 2.0 → 1.0 → 0.50 →
  0.31, straight through the alleged threshold and **8× past it**, with
  `cos_A = cos_B = 1.0` and Δcos between 2e-11 and 2e-6 at every point. Flat.
- **The jump Δ ≈ 6.4e-6 and the ‖g_fd‖·ε plateau** — I diagnosed that as a
  finite jump, and the arithmetic was right, but the jump was in the *reference*,
  not in the physics. An FD path that ignores the objective produces exactly
  that signature. Withdraw it; I was wrong about the source.
- **The dynamic half of the design inequality** (`ε_ramp ≫ vₙ·dt`) — no
  empirical support. The static half (`δ_pen ≫ ε_ramp`) was never independently
  tested and should be marked untested rather than carried forward.
- **The two-regime synthesis and the "gate-ratio second axis."**

Log this as a retraction with mechanism, the way you did the last two. Three
threads have now died in this project — the +0.80/s exponent, the pre-fix twin
scans, and now Δcos — and each retraction has made the repo more credible, not
less. This one is the biggest and it should be the most explicit.

---

## 4. What emerges instead is a stronger claim than the one you lost

Assemble what actually survives:

- Per-coordinate FD vs BPTT: **cos = 0.99999999**, max rel error 0.3%
- vₙ sweep, 0.025 → 3.2 m/s (passive-walker through running-humanoid strike
  speeds), Π_ramp 40 → 0.31: **cos = 1.0 throughout**
- Tangents cross impacts with no saltation correction, multi-DOF hybrid
- Undamped stiff contact is ill-conditioned and BPTT reports that honestly

That says: **compliant contact is variationally benign.** The pathologies the
field attributes to contact — in this simulator, across three decades of impact
speed — are not there. What breaks first-order gradients is non-smooth guards,
missing damping, and instrumentation error.

That is a more consequential claim than a scaling law, and it's a direct answer
to why MJX and Brax have been differentiable for years without producing
anything: not missing capability, missing ability to distinguish *"gradients
degraded by contact"* from *"gradients degraded by my own code."*

---

## 5. Before that claim is worth anything: a positive control

Your instrument has now reported "everything is broken" and "everything is
fine," and **both were instrument states.** `cos = 1.0` across seven rows,
`cos_A = cos_B = 1.0` exactly, Δcos at 1e-11 — that is the same smell as the
bit-identical `b` sweep. A harness that always agrees is indistinguishable from
a harness that always agrees *correctly*.

**Plant a known discontinuity and confirm the harness detects it.** Three cheap
options, run at least two:

1. Set `max_vel` low enough that it actually fires (log `clamp_stats` to prove
   it did), and confirm cos drops.
2. Re-enable the old `act = where(pen > 0, pen/(pen + 1e-9), 0)` gate — the
   10-nanometre activation from the very first review. Known-bad, known
   magnitude.
3. Replace the softplus ramp with `clamp(min=0)` on penetration — a C¹ kink
   exactly at touchdown.

**Acceptance:** the harness reports degraded cos for each planted defect, and
recovers cos = 1.0 when it's removed. Until that passes, "no contact-induced
degradation" is not a finding — it's an instrument that hasn't been shown
capable of producing a negative.

Make this a standing rule alongside knob-sanity: **any harness that reports a
null result must first demonstrate it can detect a planted positive.**

---

## 6. Coverage gaps in the vₙ sweep

Once the positive control passes, the sweep needs two more axes before the
claim generalises:

- **dt.** The sweep held dt fixed. Coarse dt is where the original explosions
  were reported, and it's where a humanoid at dt=1e-3 would actually operate.
  Sweep dt at fixed vₙ.
- **ε_ramp / k jointly.** The static constraint (`δ_pen ≫ ε_ramp`) was never
  tested independently of the dynamic one. Now that the dynamic one is dead,
  test the static one on its own — it may be the real constraint, and it's the
  half with a clean mechanical argument behind it.

Also worth logging: does `max_vel` ever fire anywhere in normal operation? You
now have `clamp_stats` wired. If it never fires, delete the clamp and remove a
non-smooth op from the hot path permanently. If it does fire, that's where to
look next.

---

## 7. Queue

1. **Fix the FD reference** so it differentiates the objective it's handed.
   One bug, everything downstream depends on it.
2. **Positive control** (§5). Non-negotiable before any null result is claimed.
3. **Correct the causal-chain claim** in log and commit history (§1); log the
   Δcos retraction with mechanism (§3).
4. **Extend the vₙ sweep to dt and to k/ε_ramp** (§6).
5. `max_vel` firing audit — delete it if it never fires.
6. Minimum-actuation, δ-criterion rebuild — unchanged priority, behind the above.

One last note. You have now caught seven of your own errors before publishing
any of them: degrees/radians, quaternion-z, the Benettin accumulator, the missing
pendulum integrator, the ε-convergence misread, the two-bug damping,
and now the FD reference. That is the actual asset here. It is also the reason
the null result, once it has a positive control behind it, will be believable
in a way that nobody else's would be.

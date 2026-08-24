"""Correctness tests for the PPO baseline (diffsim/algo/ppo.py).

Run:  python -m pytest tests/test_ppo.py -q
"""
import math
import os
import sys
import tempfile
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from diffsim.algo.ppo import PPOAgent, PPOConfig, compute_gae  # noqa: E402


def _ref_gae(rew, val, done, gamma, lam, next_val=None):
    """Independent scalar GAE recursion (plain loops, explicit branching)."""
    T, N = len(rew), len(rew[0])
    adv = [[0.0] * N for _ in range(T)]
    carry = [0.0] * N
    for t in range(T - 1, -1, -1):
        for n in range(N):
            if done[t][n]:
                adv[t][n] = rew[t][n] - val[t][n]
            else:
                if t + 1 < T:
                    v_next = val[t + 1][n]
                else:
                    v_next = next_val[n] if next_val is not None else 0.0
                adv[t][n] = rew[t][n] + gamma * v_next - val[t][n] \
                    + gamma * lam * carry[n]
            carry[n] = adv[t][n]
    return adv


def test_gae_matches_reference_and_hand_computation():
    gamma, lam = 0.99, 0.95
    rew = torch.tensor([[1.0], [-0.5], [2.0], [0.3]])
    val = torch.tensor([[0.5], [0.4], [0.6], [-0.2]])
    done = torch.tensor([[False], [False], [True], [False]])
    adv, ret = compute_gae(rew, val, done, gamma, lam)

    # hand-computed: A3=0.3+0-(-0.2); A2=2.0-0.6 (done cut);
    # A1=-0.306+0.9405*A2; A0=0.896+0.9405*A1
    expected = torch.tensor([1.84656335, 1.0107, 1.4, 0.5])
    assert torch.allclose(adv[:, 0], expected, atol=1e-10)
    assert torch.allclose(ret[:, 0], adv[:, 0] + val[:, 0], atol=1e-12)

    ref = torch.tensor(_ref_gae(rew.tolist(), val.tolist(),
                                done.tolist(), gamma, lam))
    assert torch.allclose(adv, ref, atol=1e-10)

    # explicit horizon bootstrap (episode not terminated at the end)
    adv2, _ = compute_gae(rew, val, done, gamma, lam,
                          next_val=torch.tensor([0.7]))
    ref2 = torch.tensor(_ref_gae(rew.tolist(), val.tolist(), done.tolist(),
                                 gamma, lam, [0.7]))
    assert torch.allclose(adv2, ref2, atol=1e-10)

    # randomized sweep with mid-episode terminations
    g = torch.Generator().manual_seed(0)
    rew_r = torch.randn(7, 3, generator=g)
    val_r = torch.randn(7, 3, generator=g)
    done_r = torch.rand(7, 3, generator=g) < 0.3
    adv_r, _ = compute_gae(rew_r, val_r, done_r, gamma, lam)
    ref_r = torch.tensor(_ref_gae(rew_r.tolist(), val_r.tolist(),
                                  done_r.tolist(), gamma, lam))
    assert torch.allclose(adv_r, ref_r, atol=1e-10)


def test_tanh_gaussian_logprob_matches_analytic():
    torch.manual_seed(3)
    agent = PPOAgent(PPOConfig(obs_dim=3, act_dim=2, hidden=16))
    obs = torch.randn(128, 3, dtype=torch.float64)
    u = torch.randn(128, 2, dtype=torch.float64)   # pre-tanh samples
    act = torch.tanh(u)

    logp, ent, v = agent.evaluate(obs, act)
    dist = torch.distributions.Normal(agent.actor(obs),
                                      agent.log_std.exp().expand(128, 2))
    ref_logp = dist.log_prob(u).sum(-1) \
        - torch.log(1 - act.pow(2)).sum(-1)   # exact tanh correction
    assert torch.allclose(logp, ref_logp, atol=1e-9)

    ref_ent = torch.full((128,), float(
        (agent.log_std.detach()
         + 0.5 * math.log(2 * math.pi * math.e)).sum()),
        dtype=torch.float64)
    assert torch.allclose(ent, ref_ent, atol=1e-9)
    assert torch.equal(v, agent.critic(obs).squeeze(-1))

    a, _, _ = agent.act(obs)
    assert (a.abs() < 1).all()
    adet, _, _ = agent.act(obs, deterministic=True)
    assert torch.allclose(adet, torch.tanh(agent.actor(obs)), atol=1e-12)

    # saturated actions must not produce NaN/inf
    lp_sat, _, _ = agent.evaluate(obs[:2],
                                  torch.tensor([[1.0, -1.0],
                                                [0.999999, -0.999999]]))
    assert torch.isfinite(lp_sat).all()


def _random_rollout(T, N, obs_dim, act_dim, gen):
    return {"obs": torch.randn(T, N, obs_dim, generator=gen),
            "act": torch.tanh(torch.randn(T, N, act_dim, generator=gen)),
            "logp": torch.randn(T, N, generator=gen),
            "rew": torch.randn(T, N, generator=gen),
            "val": torch.randn(T, N, generator=gen),
            "done": torch.rand(T, N, generator=gen) < 0.25}


def test_update_deterministic_given_seed():
    def once():
        torch.manual_seed(2024)
        agent = PPOAgent(PPOConfig(obs_dim=2, act_dim=1, hidden=16))
        g = torch.Generator().manual_seed(77)
        stats = agent.update(_random_rollout(5, 3, 2, 1, g))
        return stats["first_mb_loss"]

    assert abs(once() - once()) <= 1e-12


def test_gradient_flow_and_stability():
    torch.manual_seed(9)
    agent = PPOAgent(PPOConfig(obs_dim=3, act_dim=2, hidden=16))
    g = torch.Generator().manual_seed(13)
    ro = _random_rollout(8, 4, 3, 2, g)
    for _ in range(20):
        stats = agent.update(ro)
        assert all(math.isfinite(x) for x in stats.values())
    for p in agent.actor.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.norm() > 0
    assert agent.log_std.grad is not None and math.isfinite(
        float(agent.log_std.grad.norm()))
    for p in agent.critic.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.norm() > 0


def test_save_load_roundtrip():
    torch.manual_seed(21)
    cfg = PPOConfig(obs_dim=2, act_dim=1, hidden=8)
    agent = PPOAgent(cfg)
    obs = torch.randn(6, 2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "agent.pt")
        agent.save(path)
        other = PPOAgent(cfg)
        other.load(path)
    a1, _, v1 = agent.act(obs, deterministic=True)
    a2, _, v2 = other.act(obs, deterministic=True)
    assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_contextual_bandit_learning():
    """1D contextual bandit: x ~ U(-1,1), optimal action tanh(1.5x),
    reward r = 1 - (a - tanh(1.5x))^2.  Episode length 1."""
    torch.manual_seed(11)
    agent = PPOAgent(PPOConfig(obs_dim=1, act_dim=1, hidden=32))
    rng = torch.Generator().manual_seed(5)

    def rollout(T=64, N=8):
        x = torch.rand(T, N, 1, generator=rng) * 2 - 1
        opt = torch.tanh(1.5 * x)
        obs, acts, lps, rews, vals = [], [], [], [], []
        for t in range(T):
            a, lp, v = agent.act(x[t])
            r = 1 - (a - opt[t]).pow(2).sum(-1)
            obs.append(x[t]); acts.append(a); lps.append(lp)
            rews.append(r); vals.append(v)
        return {"obs": torch.stack(obs), "act": torch.stack(acts),
                "logp": torch.stack(lps), "rew": torch.stack(rews),
                "val": torch.stack(vals),
                "done": torch.ones(T, N, dtype=torch.bool)}

    t0 = time.time()
    curve = []
    for _ in range(150):
        ro = rollout()
        curve.append(float(ro["rew"].mean()))
        agent.update(ro)
    dt = time.time() - t0

    xs = torch.linspace(-1, 1, 4096, dtype=torch.float64).unsqueeze(-1)
    with torch.no_grad():
        adet = torch.tanh(agent.actor(xs))
    optv = torch.tanh(1.5 * xs)
    eval_r = float((1 - (adet - optv).pow(2)).mean())
    av, ov = adet.flatten(), optv.flatten()
    corr = float(((av - av.mean()) * (ov - ov.mean())).mean()
                 / (av.std() * ov.std()))

    print(f"\nbandit: first mean reward {curve[0]:.4f} | "
          f"last rollout {curve[-1]:.4f} | best rollout {max(curve):.4f} | "
          f"deterministic eval {eval_r:.4f} | corr(a, a*) {corr:.4f} | "
          f"train time {dt:.1f}s")
    assert curve[0] <= 0.7
    assert eval_r > 0.85
    assert corr > 0.9

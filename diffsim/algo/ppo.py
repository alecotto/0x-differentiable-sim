"""PPO baseline: clipped-surrogate proximal policy optimization.

Model-free policy-gradient control baseline for the Q2 comparison against
the exact differentiable-physics gradients exploited by SHAC.  Tanh-Gaussian
stochastic actor with exact squashing correction (Spinning Up style),
GAE(lambda) advantages, clipped surrogate, plain-MSE value loss (no value
clipping).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from .shac import mlp


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    hidden: int = 128
    clip_ratio: float = 0.2
    gamma: float = 0.99
    lam_gae: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    epochs_per_update: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    act_scale: float = 0.4  # parity with ActorCritic; tanh squash already bounds actions to (-1,1)


def compute_gae(rew, val, done, gamma, lam, next_val=None):
    """GAE(lambda) for a [T, N] rollout.

    ``done[t]`` terminates the episode AFTER step t: it cuts both the value
    bootstrap and the advantage carry-over.  Bootstrap V(s_T) at the horizon
    defaults to zero (truncation treated as termination) unless ``next_val``
    [N] is supplied.  Returns (advantages, returns = adv + val).
    """
    boot = (torch.zeros_like(val[0]) if next_val is None
            else torch.as_tensor(next_val).reshape(val[0].shape)
            .to(device=val.device, dtype=val.dtype))
    adv = torch.empty_like(rew)
    carry = torch.zeros_like(val[0])
    for t in range(rew.shape[0] - 1, -1, -1):
        cont = (~done[t]).to(val.dtype)
        v_next = boot if t == rew.shape[0] - 1 else val[t + 1]
        delta = rew[t] + gamma * cont * v_next - val[t]
        carry = delta + gamma * lam * cont * carry
        adv[t] = carry
    return adv, adv + val


class PPOAgent(nn.Module):
    """tanh-Gaussian PPO agent.

    Mirrors shac.ActorCritic layout (.actor / .critic MLPs) plus a
    state-independent learnable log_std (init -0.5).
    """

    def __init__(self, cfg: PPOConfig, dtype: torch.dtype = torch.float64,
                 device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.actor = mlp(cfg.obs_dim, cfg.hidden, cfg.act_dim, dtype)
        self.critic = mlp(cfg.obs_dim, cfg.hidden, 1, dtype)
        self.log_std = nn.Parameter(torch.full((cfg.act_dim,), -0.5,
                                               dtype=dtype))
        self._dtype = dtype
        self._eps = 1e-6 if dtype == torch.float32 else 1e-7
        self.device = torch.device(device)
        self.to(self.device)
        self.actor_params = list(self.actor.parameters()) + [self.log_std]
        self.critic_params = list(self.critic.parameters())
        self.opt_actor = torch.optim.Adam(self.actor_params, lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic_params,
                                           lr=cfg.lr_critic)

    # ------------------------------------------------------------------
    def _prep(self, x, dtype):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _dist(self, obs):
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def _correction(self, u):
        """log |d tanh(u) / du| summed over action dims."""
        a = torch.tanh(u).clamp(-(1 - self._eps), 1 - self._eps)
        return -torch.log1p(-a.pow(2)).sum(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Sample (or mean of) the squashed policy.
        Returns (action [..., A], logp [...], value [...])."""
        obs = self._prep(obs, self._dtype)
        dist = self._dist(obs)
        u = dist.loc if deterministic else dist.sample()
        logp = dist.log_prob(u).sum(-1) + self._correction(u)
        return torch.tanh(u), logp, self.critic(obs).squeeze(-1)

    def evaluate(self, obs, act):
        """Log-prob / entropy / value of stored squashed actions.
        Pre-tanh recovered via atanh on safely clamped actions."""
        obs = self._prep(obs, self._dtype)
        act = self._prep(act, self._dtype).clamp(-(1 - self._eps),
                                                 1 - self._eps)
        dist = self._dist(obs)
        u = torch.atanh(act)
        logp = dist.log_prob(u).sum(-1) + self._correction(u)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, self.critic(obs).squeeze(-1)

    # ------------------------------------------------------------------
    def update(self, rollout):
        """One PPO update over a dict {obs, act, logp, rew, val, done} with
        [T, N, ...] leading dims; optional 'next_val': [N] horizon bootstrap.
        Clipped surrogate + value_coef * MSE - entropy_coef * H, joint
        backward, joint grad-norm clip, separate Adam optimizers.
        Returns a stats dict."""
        cfg = self.cfg
        f = self._dtype
        obs = self._prep(rollout["obs"], f)
        act = self._prep(rollout["act"], f)
        old_logp = self._prep(rollout["logp"], f)
        rew = self._prep(rollout["rew"], f)
        val = self._prep(rollout["val"], f)
        done = self._prep(rollout["done"], torch.bool)
        nv = rollout.get("next_val")
        nv = None if nv is None else self._prep(nv, f)
        adv, ret = compute_gae(rew, val, done, cfg.gamma, cfg.lam_gae, nv)
        T, N = rew.shape[:2]

        def flat(x):
            return x.reshape(T * N, *x.shape[2:]) if x.dim() > 2 \
                else x.reshape(T * N)

        obs, act, old_logp = flat(obs), flat(act), flat(old_logp)
        adv, ret = flat(adv), flat(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {"pi_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
                 "first_mb_loss": None}
        n_mb = 0
        c = cfg.clip_ratio
        for _ in range(cfg.epochs_per_update):
            for sel in torch.randperm(T * N).split(cfg.minibatch_size):
                logp, ent, v = self.evaluate(obs[sel], act[sel])
                ratio = (logp - old_logp[sel]).exp()
                a = adv[sel]
                pi_loss = -torch.min(ratio * a,
                                     ratio.clamp(1 - c, 1 + c) * a).mean()
                v_loss = (v - ret[sel]).pow(2).mean()
                loss = pi_loss + cfg.value_coef * v_loss \
                    - cfg.entropy_coef * ent.mean()
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.actor_params + self.critic_params,
                    cfg.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()
                if stats["first_mb_loss"] is None:
                    stats["first_mb_loss"] = float(loss.detach())
                stats["pi_loss"] += float(pi_loss.detach())
                stats["value_loss"] += float(v_loss.detach())
                stats["entropy"] += float(ent.mean().detach())
                stats["approx_kl"] += float(
                    (((ratio - 1) - (logp - old_logp[sel])).mean()).detach())
                stats["clip_frac"] += float(
                    (((ratio - 1).abs() > c).float().mean()).detach())
                stats["grad_norm"] += float(gnorm)
                n_mb += 1
        for k in ("pi_loss", "value_loss", "entropy", "approx_kl",
                  "clip_frac", "grad_norm"):
            stats[k] /= max(n_mb, 1)
        return stats

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device,
                                        weights_only=True))

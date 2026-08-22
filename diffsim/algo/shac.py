"""SHAC-lite: short-horizon actor-critic for differentiable simulation.

The policy gradient is backpropagated through the simulator only across a
short horizon H that is *measured* to have exact gradients (see
scripts/probe_gradients.py); everything beyond the horizon is bootstrapped
from a learned value function.  This is the practical mitigation for the
chaotic-gradient tradeoff of contact-rich simulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(inp: int, hidden: int, out: int, dtype: torch.dtype = torch.float64):
    return nn.Sequential(
        nn.Linear(inp, hidden, dtype=dtype), nn.ELU(),
        nn.Linear(hidden, hidden, dtype=dtype), nn.ELU(),
        nn.Linear(hidden, out, dtype=dtype),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128,
                 act_scale: float = 0.4, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim, dtype)
        self.critic = mlp(obs_dim, hidden, 1, dtype)
        self.act_scale = act_scale

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic bounded action (residual on neutral PD targets)."""
        return self.act_scale * torch.tanh(self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class ShacTrainer:
    def __init__(self, ac: ActorCritic, gamma: float = 0.995,
                 lam: float = 0.95, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, grad_clip: float = 1.0):
        self.ac = ac
        self.gamma = gamma
        self.lam = lam
        self.grad_clip = grad_clip
        self.opt_actor = torch.optim.Adam(ac.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(ac.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    def policy_step(self, rewards: torch.Tensor, v_terminal: torch.Tensor):
        """rewards [E,H] differentiable w.r.t. policy params;
        v_terminal [E] detached bootstrap.  Returns (J_mean, grad_norm).

        NOTE: no batch normalization of J.  Advantage-style centering is
        correct for policy-gradient estimators but WRONG here: SHAC
        maximizes the true objective, and subtracting the batch mean
        cancels the shared theta-dependence of the returns (the very
        signal we need), leaving only per-env constants -> exact zero
        gradients near equilibrium.
        """
        E, H = rewards.shape
        discounts = self.gamma ** torch.arange(H, dtype=rewards.dtype,
                                               device=rewards.device)
        J = (rewards * discounts).sum(-1) \
            + (self.gamma ** H) * v_terminal.reshape(E)
        loss = -J.mean()

        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(),
                                               self.grad_clip)
        self.opt_actor.step()
        return float(J.mean().detach()), float(gnorm)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def td_lambda_targets(self, rewards, values_next, fell=None):
        """rewards [E,K], values_next [E,K] = V(s_{t+1}), optional fell mask.

        Returns G_t = TD(lambda) targets for t=0..K-1 with recursion
            G_t = r_t + gamma[(1-lam)V(s_{t+1}) + lam G_{t+1}]
        Falls are absorbing: after a fall, future reward is a fixed -2.
        """
        E, K = rewards.shape
        dev = rewards.device
        if fell is None:
            fell = torch.zeros(E, K, dtype=torch.bool, device=dev)
        r = torch.where(fell, torch.full_like(rewards, -2.0), rewards)

        G = torch.zeros_like(r)
        g = values_next[:, -1].clone()
        # walk backwards; once fallen, freeze at discounted penalty stream
        post = torch.zeros(E, dtype=r.dtype, device=dev)
        for t in range(K - 1, -1, -1):
            fell_here = fell[:, t]
            cont = r[:, t] + self.gamma * ((1 - self.lam) * values_next[:, t]
                                           + self.lam * g)
            g_new = torch.where(fell_here, post, cont)
            G[:, t] = g_new
            g = g_new
            post = -2.0 + self.gamma * post
        return G

    # ------------------------------------------------------------------
    def value_step(self, obs: torch.Tensor, targets: torch.Tensor):
        """One regression minibatch step.  Returns MSE.
        Runs under enable_grad regardless of caller context."""
        with torch.enable_grad():
            pred = self.ac.value(obs)
            loss = (pred - targets).pow(2).mean()
            self.opt_critic.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 10.0)
            self.opt_critic.step()
        return float(loss.detach())

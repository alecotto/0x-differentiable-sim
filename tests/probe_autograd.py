import os
"""Probe: is einsum/index_put backward broken in this torch build?"""
import sys
import torch

DT = torch.float64
torch.manual_seed(0)
eps = 1e-6
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def fd_grad(f, x):
    xf = x.detach().reshape(-1)
    g = torch.zeros_like(xf)
    for k in range(xf.numel()):
        xp = x.detach().clone().reshape(-1); xp[k] += eps
        xm = x.detach().clone().reshape(-1); xm[k] -= eps
        g[k] = (f(xp.reshape(x.shape)) - f(xm.reshape(x.shape))) / (2 * eps)
    return g.reshape(x.shape)


def check(name, f_a, f_b, shape):
    x = (torch.rand(shape, dtype=DT) + 0.5).requires_grad_(True)
    g1 = torch.autograd.grad(f_a(x).sum(), x)[0]
    x2 = x.detach().clone().requires_grad_(True)
    g2 = torch.autograd.grad(f_b(x2).sum(), x2)[0]
    gfd = fd_grad(f_a, x.detach())
    ok1 = torch.allclose(g1, gfd, atol=1e-6)
    ok2 = torch.allclose(g2, gfd, atol=1e-6)
    print(f"{name:30s} einsum-vs-FD={ok1}  matmul-vs-FD={ok2}")
    if not ok1:
        print("   einsum grad:", g1.flatten()[:6].tolist())
        print("   FD grad    :", gfd.flatten()[:6].tolist())
    if not ok2:
        print("   matmul grad:", g2.flatten()[:6].tolist())


I3 = torch.eye(3, dtype=DT)

# Pattern 1: three-factor chain R @ I @ R^T
I3b = I3.expand(1, 3, 3, 3)
check(
    "R@I@R^T",
    lambda x: torch.einsum("ebij,ebjk,ebkl->ebil", x, I3b, x.transpose(-1, -2)).sum(),
    lambda x: torch.matmul(torch.matmul(x, I3), x.transpose(-1, -2)).sum(),
    (1, 3, 3, 3),
)

# Pattern 2: row-weighted dots ("eci,ei->ec")
W = torch.arange(12, dtype=DT).reshape(1, 2, 6) / 10.0
check(
    "row-weighted dots",
    lambda x: torch.einsum("eci,eci->", x, W).sum(),
    lambda x: ((x * W)).sum(),
    (1, 2, 6),
)

# Pattern 3: index_put assembly identical to mass_matrix
Wm = torch.tensor([[1., 2.], [3., 4.]], dtype=DT)


def p4_put(x):
    M = torch.zeros(1, 2, 2, dtype=DT)
    H = x[:, :2]
    M[:, [1, 0], 0] = H
    M[:, 0, [1, 0]] = H
    return (M * Wm).sum()


check("index_put assembly", p4_put, p4_put, (1, 4))

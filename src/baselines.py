"""Identifiers from the literature, re-implemented from the papers, for comparison inside the same
experiment loop (only the identifier changes; the acquisition gets their coefficient estimate).

  wsindy  Weak SINDy (Messenger & Bortz 2021): weak form with smooth test functions
          phi(t) = ((t - a)(b - t))^p on sliding windows, derivatives moved onto phi by parts;
          MSTLS with scale-aware bounds and lambda chosen by the paper's loss.         trajectories only
  sbl     Bayesian SINDy by sparse Bayesian learning (Zhang & Lin 2018): ARD prior per
          coefficient, evidence updates, then thresholding at lambda as in the paper.  all settings
  msindy  Modified SINDy (Kaheman, Brunton & Kutz 2022): coefficients and the noise on every
          measured state are optimised together (derivative loss + multi-step simulation
          loss), with sequential thresholding.                                        trajectories only

All use the same library and lambda = 0.2 as E-SINDy, and no ground truth.
"""
import numpy as np

import noise_aware as na


# ------------------------------------------------------------------ Weak SINDy
def weak_rows(lib, Ys, dt, m, p=4, stride=1):
    """Weak-form rows: G[k] = int phi_k theta(y) dt,  b[k] = -int phi_k' y dt (trapezoid)."""
    G, B = [], []
    s = np.linspace(0, 1, m)
    phi = (s * (1 - s)) ** p
    dphi = p * (s * (1 - s)) ** (p - 1) * (1 - 2 * s) / ((m - 1) * dt)
    w = np.full(m, dt); w[0] = w[-1] = dt / 2
    scale = 1.0 / np.sum(w * phi)
    for Y in Ys:
        th = lib(Y)
        for a in range(0, len(Y) - m + 1, stride):
            G.append(scale * (w * phi) @ th[a:a + m])
            B.append(-scale * (w * dphi) @ Y[a:a + m])
    return np.array(G), np.array(B)


def mstls(G, b, lams=np.logspace(-4, 0, 40)):
    """Modified sequential thresholding (Messenger & Bortz 2021)."""
    J = G.shape[1]
    w0 = np.linalg.lstsq(G, b, rcond=None)[0]
    nb = np.linalg.norm(b) + 1e-300
    ng = np.linalg.norm(G, axis=0) + 1e-300
    best, best_loss = w0, np.inf
    for lam in lams:
        lo = lam * np.maximum(1, nb / ng); hi = np.minimum(1, nb / ng) / lam
        S = np.arange(J)
        w = w0.copy()
        for _ in range(20):
            keep = (np.abs(w[S]) >= lo[S]) & (np.abs(w[S]) <= hi[S])
            if keep.all():
                break
            S = S[keep]
            w = np.zeros(J)
            if not len(S):
                break
            w[S] = np.linalg.lstsq(G[:, S], b, rcond=None)[0]
        loss = np.linalg.norm(G @ (w - w0)) / (np.linalg.norm(G @ w0) + 1e-300) + np.count_nonzero(w) / J
        if loss < best_loss:
            best, best_loss = w, loss
    return best


def wsindy(lib, Ys, dt, d):
    n = min(len(Y) for Y in Ys)
    m = n if n <= 12 else max(7, n // 5)                    # test-function support: whole short trajectory, else ~1/5
    stride = 1 if n <= 12 else max(1, m // 4)
    G, B = weak_rows(lib, Ys, dt, m, stride=stride)
    return np.stack([mstls(G, B[:, c]) for c in range(d)], 1)


# --------------------------------------------------------- sparse Bayesian learning
sbl = na.sbl


def whitened_sbl(rows, block, lam, passes=3):
    """Our noise model inside sparse Bayesian learning: SBL assumes every row has the same noise,
    so rows are first whitened with sigma^2 J J^T at the current estimate (iteratively reweighted)."""
    A, B, dA, dB = (None if x is None else x.reshape((-1, block) + x.shape[1:]) for x in rows)
    p, d = A.shape[-1], B.shape[-1]
    C = np.zeros((p, d))
    for c in range(d):
        xi = sbl(A.reshape(-1, p), B[..., c].ravel(), lam)
        for _ in range(passes if dA is not None else 0):
            S = np.nonzero(xi)[0]
            if not len(S):
                break
            e = dict(support=S, xi=xi[S])
            Aw, bw = na.whiten(A, B[..., c], dA, None if dB is None else dB[..., c, :], e)
            xi = sbl(Aw.reshape(-1, p), bw.ravel(), lam)
        C[:, c] = xi
    return C


# ------------------------------------------------------------------- Modified SINDy
def msindy(lib, Ys, dt, sim, lam, C0, iters=300, cycles=4, q=3):
    import torch
    torch.set_num_threads(1)
    Y = torch.tensor(np.stack(Ys), dtype=torch.float64)                     # [E, n, d]
    E, n, d = Y.shape
    q = min(q, n - 1)
    scale = Y.reshape(-1, d).std(0) + 1e-12
    idx = torch.tensor(lib.idx)
    Xi = torch.tensor(C0, dtype=torch.float64, requires_grad=True)
    N = torch.zeros_like(Y, requires_grad=True)
    mask = torch.ones_like(Xi)

    def theta(X):
        X1 = torch.cat([torch.ones(X.shape[:-1] + (1,), dtype=X.dtype), X], -1)
        out = X1[..., idx[:, 0]]
        for k in range(1, idx.shape[1]):
            out = out * X1[..., idx[:, k]]
        return out

    f = lambda X: theta(X) @ (Xi * mask)
    h = dt / sim["every"]

    def step(X):                                                             # one sampling interval, RK4 sub-steps
        for _ in range(sim["every"]):
            k1 = f(X); k2 = f(X + h / 2 * k1); k3 = f(X + h / 2 * k2); k4 = f(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    for _ in range(cycles):
        opt = torch.optim.Adam([Xi, N], lr=1e-2)
        for _ in range(iters):
            opt.zero_grad()
            X = Y - N
            dX = (X[:, 2:] - X[:, :-2]) / (2 * dt)
            loss = (((dX - f(X[:, 1:-1])) / scale) ** 2).mean()
            Xs = X[:, :-1]
            for j in range(1, q + 1):
                Xs = step(Xs)
                loss = loss + 0.9 ** j * (((Xs[:, :n - j] - X[:, j:]) / scale) ** 2).mean()
                Xs = Xs[:, :-1]
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
        with torch.no_grad():
            mask *= (Xi.abs() >= lam).double()
            Xi *= mask
    return (Xi * mask).detach().numpy()


# ---------------------------------------------------------------- shared plumbing
def as_eqs(rows, C, block):
    """Coefficients -> the per-equation dicts the acquisition scores use (rows whitened at C)."""
    A, B, dA, dB = (None if x is None else x.reshape((-1, block) + x.shape[1:]) for x in rows)
    eqs = []
    for c in range(C.shape[1]):
        S = np.nonzero(C[:, c])[0]
        e = dict(c=c, support=S, xi=C[S, c], ensemble=np.tile(C[:, c], (2, 1)))
        Aw, _ = na.whiten(A, B[..., c], dA, None if dB is None else dB[..., c, :], e)
        e["A"] = Aw.reshape(-1, A.shape[-1])
        eqs.append(e)
    return eqs


def identify(name, rows, prob, rng):
    A, B = rows[0], rows[1]
    d = B.shape[1]
    if name == "sbl":
        C = np.stack([sbl(A, B[:, c], prob.lam) for c in range(d)], 1)
    elif name == "wsbl":
        C = whitened_sbl(rows, prob.block, prob.lam)
    else:
        Ys = prob.Ys
        if name == "wsindy":
            C = wsindy(prob.lib, Ys, prob.dt, d)
        else:
            C0 = np.stack([np.linalg.lstsq(A, B[:, c], rcond=None)[0] for c in range(d)], 1)
            C = msindy(prob.lib, Ys, prob.dt, prob.sim, prob.lam, C0)
    return as_eqs(rows, C, prob.block)

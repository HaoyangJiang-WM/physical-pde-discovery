"""Noise-aware E-SINDy: identification and experiment selection from one noise model.

A regression row is  r = b - a . xi,  where a (library features) and b (target)
are computed from noisy measurements m = m_true + sigma * eps.  To first order

    Cov(r) = sigma^2 * J J^T,      J = d r / d m = db/dm - (da/dm) xi

so the noise of a row depends on where it was measured and on the equation
itself, and rows that share measurements are correlated.  E-SINDy and the acquisition scores built on it assume it is constant.
Three consequences, all from this one formula:

  estimation   whiten the rows with this covariance       (`whiten`)
  selection    with correct weights the standard errors are valid, so a term
               must also be distinguishable from zero    (`stlsq(test=True)`)
  acquisition  measure where the model's prediction variance over the experimental
               range drops most, rows noise-whitened     (`prediction_score`, I-optimal)

Rows are passed as a tuple (A [n, p], B [n, d], dA [n, p, M] | None, dB [n, d, M] | None);
dA = None means the features are noise-free (weights 1), dB = None means the
targets are.  Candidate experiments carry extra leading axes [Na, m, ...]: each
candidate would add m rows.  Nothing here uses the true system or a candidate's label.
"""
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


# ----------------------------------------------------------------- library
class PolyLibrary:
    """Monomials of total degree 1..degree in d variables (no constant, as in
    the official Active E-SINDy code)."""

    def __init__(self, d, degree, names):
        combos = [c for k in range(1, degree + 1)
                  for c in itertools.combinations_with_replacement(range(d), k)]
        self.d, self.p = d, len(combos)
        self.idx = np.zeros((self.p, degree), dtype=int)       # 0 -> factor 1
        for j, c in enumerate(combos):
            self.idx[j, : len(c)] = np.array(c, dtype=int) + 1
        self.names = ["*".join(names[i] for i in c) for c in combos]

    def _factors(self, X):
        X1 = np.concatenate([np.ones(X.shape[:-1] + (1,)), X], axis=-1)
        return [X1[..., self.idx[:, s]] for s in range(self.idx.shape[1])]

    def __call__(self, X):
        return np.prod(self._factors(X), axis=0)

    def grad(self, X):
        """d Theta / d x : [..., p, d]."""
        cols = self._factors(X)
        out = np.zeros(X.shape[:-1] + (self.p, self.d))
        for s in range(len(cols)):
            rest = np.prod([c for t, c in enumerate(cols) if t != s], axis=0)
            for k in range(self.d):
                out[..., k] += rest * (self.idx[:, s] == k + 1)
        return out


def simulate(lib, C, x0, h, n_steps, every, lo, hi):
    """RK4 of dx/dt = Theta(x) C from x0 [B, d] -> [B, n_steps / every + 1, d]."""
    f = lambda x: lib(x) @ C
    x = np.clip(x0, lo, hi)
    out = [x]
    with np.errstate(all="ignore"):          # a diverging model is clipped to the box
        for s in range(n_steps):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = np.clip(np.nan_to_num(x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)), lo, hi)
            if (s + 1) % every == 0:
                out.append(x)
    return np.stack(out, axis=1)


# ------------------------------------------------------------------- rows
def stack(*rows):
    return tuple(None if parts[0] is None else np.concatenate(parts) for parts in zip(*rows))


def flat(rows, lead=2):
    return tuple(None if x is None else x.reshape((-1,) + x.shape[lead:]) for x in rows)


def integral_rows(lib, Y, dt, window):
    """Trajectories Y [E, n, d] -> rows [E, n - w, ...] of the integral form
    x(t + w dt) - x(t) = int Theta(x) dt . xi  (trapezoid), with their Jacobians
    w.r.t. all n * d measured states of the trajectory."""
    E, n, d = Y.shape
    K, q = n - window, dt * np.r_[0.5, np.ones(window - 1), 0.5]
    th, gr = lib(Y), lib.grad(Y)
    A, dA, dB = np.zeros((E, K, lib.p)), np.zeros((E, K, lib.p, n, d)), np.zeros((K, d, n, d))
    for k in range(K):
        A[:, k] = np.einsum("j,ejp->ep", q, th[:, k:k + window + 1])
        dA[:, k, :, k:k + window + 1] = (q[:, None, None] * gr[:, k:k + window + 1]).transpose(0, 2, 1, 3)
        dB[k, np.arange(d), k + window, np.arange(d)], dB[k, np.arange(d), k, np.arange(d)] = 1.0, -1.0
    return (A, Y[:, window:] - Y[:, :-window], dA.reshape(E, K, lib.p, -1),
            np.broadcast_to(dB.reshape(K, d, -1), (E, K, d, n * d)))


# ------------------------------------------------------------ noise model
def whiten(A, b, dA, dB, e):
    """Rows of one experiment share measurements, so their residuals have
    covariance sigma^2 J J^T with J = dr/dm = dB - dA . xi  [..., K, M].
    Returns L^-1 A (and L^-1 b), L L^T = J J^T: rows with unit, uncorrelated
    noise.  For a single row this is division by |dr/dm|."""
    if dA is None or not len(e["support"]):
        return A, b
    J = -(dA[..., e["support"], :] * e["xi"][:, None]).sum(-2)
    J = J if dB is None else J + dB
    cov = J @ np.swapaxes(J, -1, -2)
    floor = e.get("floor") or 1e-3 * np.mean(np.diagonal(cov, axis1=-2, axis2=-1)) + 1e-300
    e["floor"] = floor
    L = np.linalg.cholesky(cov + floor * np.eye(cov.shape[-1]))
    return np.linalg.solve(L, A), None if b is None else np.linalg.solve(L, b[..., None])[..., 0]


# --------------------------------------------------------- identification
def stlsq(A, b, lam, test):
    """Sequential thresholding: drop every term with |xi| < lam.  With `test`,
    then also drop the least significant term while its t^2 < log n (the BIC
    rule for one coefficient): a term must be large *and* distinguishable from 0."""
    n, p = A.shape
    S = np.arange(p)
    xi = np.zeros(0)
    while len(S):
        G = np.linalg.pinv(A[:, S].T @ A[:, S])
        xi = G @ (A[:, S].T @ b)
        small = np.abs(xi) < lam
        if not small.any() and test:
            s2 = np.sum((A[:, S] @ xi - b) ** 2) / max(n - len(S), 1)
            t2 = xi ** 2 / (s2 * np.diag(G) + 1e-300)
            small[np.argmin(t2)] = t2.min() < np.log(n)
        if not small.any():
            break
        S = S[~small]
    out = np.zeros(p)
    out[S] = xi[: len(S)]
    return out


def sbl(A, b, lam, iters=300):
    """ARD / relevance-vector regression (Tipping 2001) as used by Zhang & Lin (2018), followed by
    thresholding of the posterior mean at lam and re-running on the kept terms."""
    n, p = A.shape
    keep = np.arange(p)
    mu = np.zeros(p)
    for _ in range(5):
        if not len(keep):
            break
        X = A[:, keep]; s = np.linalg.norm(X, axis=0) + 1e-300; X = X / s
        alpha = np.ones(len(keep)); beta = 1.0 / (np.var(b) + 1e-300) * 100
        active = np.ones(len(keep), bool)
        for _ in range(iters):
            Xa = X[:, active]
            Sig = np.linalg.inv(beta * Xa.T @ Xa + np.diag(alpha[active]))
            m_ = beta * Sig @ Xa.T @ b
            gam = 1 - alpha[active] * np.diag(Sig)
            alpha_new = gam / (m_ ** 2 + 1e-300)
            beta = (n - gam.sum()) / (np.sum((b - Xa @ m_) ** 2) + 1e-300)
            a = alpha.copy(); a[active] = alpha_new
            converged = np.max(np.abs(np.log(a[active] + 1e-300) - np.log(alpha[active] + 1e-300))) < 1e-3
            alpha = a
            active = active & (alpha < 1e6)
            if not active.any() or converged:
                break
        mu_s = np.zeros(len(keep))
        if active.any():
            Xa = X[:, active]
            Sig = np.linalg.inv(beta * Xa.T @ Xa + np.diag(alpha[active]))
            mu_s[active] = beta * Sig @ Xa.T @ b
        coef = mu_s / s
        new_keep = keep[(coef != 0) & (np.abs(coef) >= lam)]
        mu = np.zeros(p); mu[keep] = coef
        if len(new_keep) == len(keep):
            break
        keep = new_keep
        mu = np.zeros(p)
        if len(keep):
            mu[keep] = np.linalg.lstsq(A[:, keep], b, rcond=None)[0]
    mu[np.abs(mu) < lam] = 0
    return mu


def esindy(A, b, rng, lam, test, n_lib=100, n_data=100, lib_frac=0.9, lib_tol=0.4, inc_tol=0.6, solver="stlsq"):
    """E-SINDy double bagging (Fasel et al. 2022).  Returns (support, mean
    coefficients on it, ensemble [n_data, p]).  `solver` is the sparse regression inside each
    bag: "stlsq" (thresholding, plus the t-test if `test`) or "ard" (sparse Bayesian learning)."""
    n, p = A.shape
    fit = (lambda A_, b_: sbl(A_, b_, lam, iters=50)) if solver == "ard" else (lambda A_, b_: stlsq(A_, b_, lam, test))
    hits, drawn = np.zeros(p), np.zeros(p)
    for _ in range(n_lib):
        cols = rng.choice(p, int(lib_frac * p), replace=False)
        drawn[cols] += 1
        hits[cols] += fit(A[:, cols], b) != 0
    keep = np.nonzero(hits / np.maximum(drawn, 1) >= lib_tol)[0]
    ens = np.zeros((n_data, p))
    for k in range(n_data if len(keep) else 0):
        rows = rng.integers(0, n, n)
        ens[k, keep] = fit(A[rows][:, keep], b[rows])
    S = np.nonzero((ens != 0).mean(0) >= inc_tol)[0] if len(keep) else keep
    return S, ens[:, S].mean(0), ens


def identify(rows, rng, lam, weighted, test, block=1, passes=3, solver="stlsq"):
    """One E-SINDy per equation.  `weighted` re-fits on rows whitened with the
    noise model of the previous estimate (iteratively reweighted least squares);
    `block` = number of rows per experiment."""
    A, B, dA, dB = (None if x is None else x.reshape((-1, block) + x.shape[1:]) for x in rows)
    eqs = []
    for c in range(B.shape[-1]):
        Aw, bw = A, B[..., c]
        for _ in range(passes if weighted and dA is not None else 1):
            S, xi, ens = esindy(Aw.reshape(-1, A.shape[-1]), bw.ravel(), rng, lam, test, solver=solver)
            e = dict(c=c, support=S, xi=xi, ensemble=ens, A=Aw.reshape(-1, A.shape[-1]))
            if not weighted:
                break
            Aw, bw = whiten(A, B[..., c], dA, None if dB is None else dB[..., c, :], e)
        eqs.append(e)
    return eqs


# ------------------------------------------------------------- acquisition
def dopt_score(eqs, cand):
    """Baseline from "How Low Can You Go?": D-optimal (leverage) on the raw library features,
    noise-blind, log det(I + A G^-1 A^T) with a unit-information prior diag(G)/n."""
    Aa = cand[0]
    total = 0.0
    for e in eqs:
        G = e["A"].T @ e["A"]
        G = G + np.diag(np.diag(G)) / len(e["A"]) + 1e-300 * np.eye(len(G))
        M = np.eye(Aa.shape[1]) + np.einsum("amp,pq,anq->amn", Aa, np.linalg.inv(G), Aa)
        total = total + np.linalg.slogdet(M)[1]
    return total


def prediction_score(eqs, cand):
    """I-optimal design: how much a candidate experiment lowers the average variance of the
    model's prediction over the candidate pool (the experimental range),

        tr(P [G^-1 - (G + A_w^T A_w)^-1]),   P = mean of a_w a_w^T over the pool,

    with all rows noise-whitened.  Directions that change no prediction anywhere in the range
    get no weight; it is invariant to how the library columns are scaled, so no prior is needed."""
    Aa, _, dAa, dBa = cand
    total = 0.0
    for e in eqs:
        Aw, _ = whiten(Aa, None, dAa, None if dBa is None else dBa[..., e["c"], :], e)
        G = e["A"].T @ e["A"]
        d = np.sqrt(np.diag(G)); d[d <= 1e-12 * max(d.max(), 1e-300)] = 1.0
        Gi = np.linalg.inv(G / np.outer(d, d) + 1e-6 * np.eye(len(G)))
        As = Aw / d
        bad = ~np.isfinite(As).all((1, 2)) | (np.abs(np.nan_to_num(As)).max((1, 2)) > 1e100)   # e.g. a diverging predicted trajectory
        As[bad] = 0.0
        P = np.einsum("amp,amq->pq", As, As) / max(((~bad).sum()) * As.shape[1], 1)
        U = As @ Gi                                                          # [Na, m, p]
        M = np.eye(As.shape[1]) + np.einsum("amp,anp->amn", U, As)
        bad = bad | ~np.isfinite(M).all((1, 2)) | ~np.isfinite(U).all((1, 2))
        M[bad] = np.eye(As.shape[1]); U[bad] = 0.0
        total = total + np.where(bad, -np.inf, np.einsum("amp,amn,anq,qp->a", U, np.linalg.pinv(M), U, P))
    return total


def variance_score(eqs, cand):
    """Published Active E-SINDy score: ensemble std of the prediction, averaged
    over equations (and over the rows of a candidate)."""
    return np.mean([(cand[0] @ e["ensemble"].T).std(-1).mean(-1) for e in eqs], axis=0)

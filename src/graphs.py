"""Graph scenario: network dynamics  dx_i/dt = F(x_i) + sum_j A_ij G(x_i, x_j),  topology known.

One experiment = set the network to one of 200 candidate states and measure every node (a snapshot);
the state measurements are noisy, the derivatives exact (as in the point protocol).  Node i's
regression row uses its own and its neighbours' measured states, so the noise Jacobian J follows the
graph and the rows of one snapshot are correlated: each snapshot is one block, whitened with
sigma^2 J J^T.  The library is shared by all nodes; columns and target are scaled to unit RMS over
the sampling box so lambda = 0.2 means the same as elsewhere.

    python src/graphs.py [n_seeds] [systems] [noise levels] [arms] [out.json]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import experiments as ex

N_NODES = 10
NAMES = ["x", "x²", "x³", "Σ A·x_j", "Σ A·x_j²", "x·Σ A·x_j", "k·x", "k·x²"]
SYSTEMS = {   # true terms (in NAMES), sampling box
    "sis": ({"x": -1.0, "Σ A·x_j": 0.5, "x·Σ A·x_j": -0.5}, (0.0, 1.0)),
    "bistable": ({"x": 1.0, "x³": -1.0, "Σ A·x_j": 0.5, "k·x": -0.5}, (-1.5, 1.5)),
    "competition": ({"x": 1.0, "x²": -1.0, "x·Σ A·x_j": -0.3}, (0.0, 2.0)),
}


class GraphLibrary:
    """Per-node features of a network state X [..., N] -> [..., N, p], with exact d/dX -> [..., N, p, N]."""

    def __init__(self, adj, lo, hi):
        self.adj, self.k = adj, adj.sum(1)
        self.p, self.d, self.names = len(NAMES), 1, NAMES
        self.scale = 1.0
        X = np.random.default_rng(12345).uniform(lo, hi, (4000, len(adj)))
        self.scale = np.sqrt(np.mean(self(X) ** 2, axis=(0, 1))) + 1e-300

    def __call__(self, X):
        Ax, Ax2 = X @ self.adj.T, (X ** 2) @ self.adj.T
        F = np.stack([X, X ** 2, X ** 3, Ax, Ax2, X * Ax, self.k * X, self.k * X ** 2], -1)
        return F / self.scale

    def grad(self, X):
        N = len(self.adj)
        I, Adj = np.eye(N), self.adj
        Ax = X @ Adj.T
        Xi = X[..., :, None]                                     # x_i
        Xm = X[..., None, :]                                     # x_m
        g = np.stack([
            np.broadcast_to(I, X.shape[:-1] + (N, N)),
            2 * Xi * I,
            3 * Xi ** 2 * I,
            np.broadcast_to(Adj, X.shape[:-1] + (N, N)),
            2 * Xm * Adj,
            Ax[..., :, None] * I + Xi * Adj,
            self.k[:, None] * I * np.ones_like(Xi),
            2 * self.k[:, None] * Xi * I,
        ], -2)                                                   # [..., N, p, N]
        return g / self.scale[:, None]


def random_graph(rng, n=N_NODES, p=0.3):
    while True:
        A = np.triu((rng.random((n, n)) < p).astype(float), 1)
        A = A + A.T
        if (A.sum(1) > 0).all():
            return A


class GraphSnapshots:
    lam, n_init, n_steps, n_cand = 0.2, 3, 10, 200

    def __init__(self, env, system, noise):
        terms, (lo, hi) = SYSTEMS[system]
        self.env, self.lo, self.hi = env, lo, hi
        self.adj = random_graph(env)
        self.lib = GraphLibrary(self.adj, lo, hi)
        xi = np.zeros(len(NAMES))
        for t, v in terms.items():
            xi[NAMES.index(t)] = v * self.lib.scale[NAMES.index(t)]
        X = env.uniform(lo, hi, (2000, N_NODES))
        self.tau = np.sqrt(np.mean((self.lib(X) @ xi) ** 2))
        self.C = (xi / self.tau)[:, None]                         # [p, 1]
        self.truth = self.C != 0
        self.sigma = noise * np.sqrt(np.mean(X ** 2))
        self.block = self.per = N_NODES
        self.n_init_snap = self.n_init
        self.n_init = self.n_init * N_NODES                      # rows, for the log

    def _states(self, n):
        return self.env.uniform(self.lo, self.hi, (n, N_NODES))

    def _measure(self, X):                                       # X [s, N] -> rows [s*N, ...]
        noisy = X + self.sigma * self.env.standard_normal(X.shape)
        A = self.lib(noisy).reshape(-1, self.lib.p)
        B = (self.lib(X) @ self.C).reshape(-1, 1)
        dA = self.lib.grad(noisy).reshape(-1, self.lib.p, N_NODES)
        return A, B, dA, None

    def initial(self):
        return self._measure(self._states(self.n_init_snap))

    def candidates(self, eqs):
        self.X = self._states(self.n_cand)
        return (self.lib(self.X), None, self.lib.grad(self.X), None), np.arange(self.n_cand)

    def measure(self, i):
        return self._measure(self.X[i:i + 1])


ex.PROTOCOLS["graph"] = GraphSnapshots
ARMS = ["esindy+variance", "esindy+dopt", "sbl+iopt", "esindy+random", "esindy_wt+random", "esindy_wt+iopt"]


def check(noise=0.0):
    """Noise-free identifiability and the noise formula on one graph per system."""
    import noise_aware as na
    for system in SYSTEMS:
        env = np.random.default_rng(0)
        prob = GraphSnapshots(env, system, noise)
        rows = prob._measure(prob._states(13))
        eqs = na.identify(rows, np.random.default_rng(1), prob.lam, True, True, prob.block)
        found = set(np.nonzero([j in eqs[0]["support"] for j in range(prob.lib.p)])[0])
        true = set(np.nonzero(prob.truth[:, 0])[0])
        X = prob._states(3000); Xn = X + 0.05 * np.sqrt(np.mean(X ** 2)) * env.standard_normal(X.shape)
        r = (prob.lib(X) @ prob.C - prob.lib(Xn) @ prob.C)[..., 0]
        J = np.einsum("snpm,p->snm", prob.lib.grad(Xn), prob.C[:, 0])
        pred = 0.05 * np.sqrt(np.mean(X ** 2)) * np.linalg.norm(J, axis=-1)
        q = np.quantile(pred, [0, .25, .5, .75, 1]); ratio = []
        for a, b in zip(q[:-1], q[1:]):
            m = (pred >= a) & (pred <= b); ratio.append(r[m].std() / pred[m].mean())
        print(f"{system:12s} noise-free 13 snapshots exact: {found == true}   measured/predicted residual sd by quartile: {np.round(ratio, 2)}"
              f"   spread of predicted sd {q[-1] / max(q[0], 1e-12):.1f}x   neighbours in J per row {np.mean((np.abs(J) > 1e-12).sum(-1)):.1f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        check(); sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    arg = lambda i, d: sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "-" else d
    systems = arg(2, ",".join(SYSTEMS)).split(",")
    noises = [float(v) for v in arg(3, "0.05,0.1,0.2").split(",")]
    arms = arg(4, ",".join(ARMS)).split(",")
    out = arg(5, "results/graph.json")
    jobs = [("graph", s, nz, a, sd) for s in systems for nz in noises for a in arms for sd in range(n)]
    t0 = time.time()
    with Pool() as pool:
        R = pool.map(ex.run_trial, jobs, chunksize=1)
    json.dump(R, open(out, "w"))
    print(f"{len(jobs)} trials in {time.time() - t0:.0f}s")
    ex.show(R)

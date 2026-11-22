"""Full pipeline on biology / chemistry / materials models, point-observation protocol.

Each model gets a candidate library that contains its true terms plus distractors
(polynomials, Hill functions, saturating rates, Arrhenius factors).  The library is
generic: any list of column functions; gradients are taken by central differences.
Columns and targets are scaled to unit RMS over the sampling box so that the
E-SINDy threshold lambda = 0.2 means the same thing for every model (applied to
every arm alike).  Three models had a true term whose normalised coefficient was
below lambda, i.e. unrecoverable by any method; their parameters or boxes were
changed so every true term is at least 0.3 (Selkov a = 0.3, smaller box;
Brusselator box [0,2]x[0,3]; Landau damping 1 and P^5 coefficient -1).  Protocol as in Active E-SINDy: 20 random states, then +1 chosen
among 200 fresh candidates, up to 100; states noisy, derivatives exact.

    python src/domains.py check                 # normalised true coefficients, noise-free recovery
    python src/domains.py [n_seeds] [models] [noise levels] [arms] [out.json]
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import experiments as ex
import noise_aware as na


# ----------------------------------------------------------------- library
class FuncLibrary:
    """Columns are functions of the state X [..., d]."""

    def __init__(self, cols, d, lo, hi):
        self.names, self.f = [c[0] for c in cols], [c[1] for c in cols]
        self.p, self.d = len(cols), d
        X = np.random.default_rng(12345).uniform(lo, hi, (20000, d))
        self.scale = 1.0
        self.scale = np.sqrt(np.mean(self(X) ** 2, 0)) + 1e-300
        self.h = 1e-5 * (np.asarray(hi, float) - np.asarray(lo, float))

    def __call__(self, X):
        return np.stack([np.broadcast_to(f(X), X.shape[:-1]) for f in self.f], -1) / self.scale

    def grad(self, X):
        out = np.zeros(X.shape[:-1] + (self.p, self.d))
        for k in range(self.d):
            e = np.zeros(self.d); e[k] = self.h[k]
            out[..., k] = (self(X + e) - self(X - e)) / (2 * self.h[k])
        return out


def poly(names, degree, const):
    d = len(names); cols = [("1", lambda X: np.ones(X.shape[:-1]))] if const else []
    for k in range(1, degree + 1):
        for c in itertools.combinations_with_replacement(range(d), k):
            cols.append(("*".join(names[i] for i in c), lambda X, c=c: np.prod([X[..., i] for i in c], 0)))
    return cols


hill = lambda s: 1 / (1 + s ** 2)
arr = lambda T: np.exp(5 * (1 - 1 / T))

# name: field, state names, box, columns, true terms {(column, eq): coefficient}
MODELS = {
    "sir": ("biology", "SIR epidemic", ["S", "I", "R"], [0, 0, 0], [1, 1, 1], lambda n: poly(n, 2, True),
            {("S*I", 0): -0.5, ("S*I", 1): 0.5, ("I", 1): -0.2, ("I", 2): 0.2}),
    "fhn": ("biology", "FitzHugh–Nagumo neuron", ["v", "w"], [-2.5, -1], [2.5, 2], lambda n: poly(n, 3, True),
            {("v", 0): 1, ("v*v*v", 0): -1 / 3, ("w", 0): -1, ("1", 0): 0.5,
             ("v", 1): 0.08, ("1", 1): 0.056, ("w", 1): -0.064}),
    "selkov": ("biology", "Selkov glycolysis", ["x", "y"], [0, 0], [1.5, 2.5], lambda n: poly(n, 3, True),
               {("x", 0): -1, ("y", 0): 0.3, ("x*x*y", 0): 1, ("1", 1): 0.6, ("y", 1): -0.3, ("x*x*y", 1): -1}),
    "toggle": ("biology", "genetic toggle switch (Hill)", ["u", "v"], [0, 0], [4, 4],
               lambda n: [("1", lambda X: np.ones(X.shape[:-1]))] + poly(n, 2, False) + [
                   ("1/(1+u²)", lambda X: hill(X[..., 0])), ("1/(1+v²)", lambda X: hill(X[..., 1])),
                   ("u/(1+u²)", lambda X: X[..., 0] * hill(X[..., 0])), ("v/(1+v²)", lambda X: X[..., 1] * hill(X[..., 1]))],
               {("1/(1+v²)", 0): 4, ("u", 0): -1, ("1/(1+u²)", 1): 4, ("v", 1): -1}),
    "mm": ("chemistry", "Michaelis–Menten enzyme kinetics", ["S", "P"], [0, 0], [3, 3],
           lambda n: [("1", lambda X: np.ones(X.shape[:-1]))] + poly(n, 2, False) + [
               (f"S/({K}+S)", lambda X, K=K: X[..., 0] / (K + X[..., 0])) for K in (0.1, 0.5, 2.0)],
           {("S/(0.5+S)", 0): -2, ("S/(0.5+S)", 1): 2}),
    "brusselator": ("chemistry", "Brusselator", ["x", "y"], [0, 0], [2, 3], lambda n: poly(n, 3, True),
                    {("1", 0): 1, ("x", 0): -4, ("x*x*y", 0): 1, ("x", 1): 3, ("x*x*y", 1): -1}),
    "binding": ("chemistry", "reversible binding A+B⇌C", ["A", "B", "C"], [0, 0, 0], [2, 2, 2], lambda n: poly(n, 2, True),
                {("A*B", 0): -2, ("C", 0): 1, ("A*B", 1): -2, ("C", 1): 1, ("A*B", 2): 2, ("C", 2): -1}),
    "arrhenius": ("chemistry", "exothermic Arrhenius reaction", ["c", "T"], [0, 0.8], [1, 1.6],
                  lambda n: poly(n, 2, True) + [("e(T)", lambda X: arr(X[..., 1])), ("c·e(T)", lambda X: X[..., 0] * arr(X[..., 1]))],
                  {("c·e(T)", 0): -1, ("c·e(T)", 1): 0.5, ("T", 1): -0.3, ("1", 1): 0.3}),
    "allencahn": ("materials", "Allen–Cahn order parameter", ["φ"], [-1.5], [1.5], lambda n: poly(n, 5, True),
                  {("φ", 0): 1, ("φ*φ*φ", 0): -1}),
    "grain": ("materials", "grain-growth phase field", ["a", "b"], [-1.2, -1.2], [1.2, 1.2], lambda n: poly(n, 3, False),
              {("a", 0): 1, ("a*a*a", 0): -1, ("a*b*b", 0): -3, ("b", 1): 1, ("b*b*b", 1): -1, ("a*a*b", 1): -3}),
    "landau": ("materials", "Landau ferroelectric polarisation", ["P", "v"], [-1.2, -1.5], [1.2, 1.5], lambda n: poly(n, 5, False),
               {("v", 0): 1, ("P", 1): 2, ("P*P*P", 1): -4, ("P*P*P*P*P", 1): -1, ("v", 1): -1}),
}


class DomainPoints(ex.Points):
    def __init__(self, env, system, noise):
        field, title, names, lo, hi, cols, terms = MODELS[system]
        self.lo, self.hi, self.env = np.array(lo, float), np.array(hi, float), env
        self.lib = FuncLibrary(cols(names), len(names), self.lo, self.hi)
        C = np.zeros((self.lib.p, self.lib.d))
        for (t, c), v in terms.items():
            C[self.lib.names.index(t), c] = v * self.lib.scale[self.lib.names.index(t)]
        X = np.random.default_rng(12345).uniform(self.lo, self.hi, (20000, self.lib.d))
        self.tau = np.sqrt(np.mean((self.lib(X) @ C) ** 2, 0))        # unit-RMS targets
        self.C = C / self.tau
        self.truth = self.C != 0
        self.sigma = noise * np.sqrt(np.mean(X ** 2))


ex.PROTOCOLS["domain"] = DomainPoints
ARMS = ["esindy+variance", "esindy+dopt", "esindy_wt+variance", "esindy_wt+iopt"]


def check():
    for key, m in MODELS.items():
        prob = DomainPoints(np.random.default_rng(0), key, 0.0)
        small = np.abs(prob.C[prob.truth]).min()
        rows = prob._measure(prob._states(100))
        eqs = na.identify(rows, np.random.default_rng(1), prob.lam, True, True)
        mask = np.zeros_like(prob.truth)
        for e in eqs:
            mask[e["support"], e["c"]] = True
        cond = np.linalg.cond(rows[0])
        print(f"{m[0]} {m[1]:22s} p={prob.lib.p:2d} d={prob.lib.d}  smallest |normalised true coef| {small:.3f}  "
              f"cond(Θ) {cond:9.1f}  noise-free exact: {bool((mask == prob.truth).all())}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        check()
        sys.exit()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    models = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "-" else list(MODELS)
    noises = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] != "-" else [0.05, 0.1]
    arms = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "-" else ARMS
    out = sys.argv[5] if len(sys.argv) > 5 else "results/domains.json"
    jobs = [("domain", s, nz, a, sd) for s in models for nz in noises for a in arms for sd in range(n_seeds)]
    t0 = time.time()
    with Pool() as pool:
        results = pool.map(ex.run_trial, jobs, chunksize=1)
    json.dump(results, open(out, "w"))
    print(f"{len(jobs)} trials in {time.time() - t0:.0f}s")
    ex.show(results)

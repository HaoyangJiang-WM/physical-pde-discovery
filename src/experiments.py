"""Active E-SINDy vs its noise-aware version, on the published protocols.

  points  Active E-SINDy (Fasel et al. 2022, official code): 20 random states,
          then +1 state per step chosen among 200 fresh candidates, up to 100;
          states are noisy, derivatives exact.
  traj    "How Low Can You Go?" ODE: one experiment = choose an initial state,
          record a 10-point noisy trajectory; start from 1, add 9; candidates
          are 100 Latin-hypercube initial states.
  pde     "How Low Can You Go?" PDE: candidates are space-time grid points with
          known library features; measuring one returns u_t there with noise.
          Noise is on the target only, so the noise weights are 1.

An arm is  identification + acquisition:
  identification  esindy | esindy_w (noise weights) | esindy_t (significance
                  test) | esindy_wt (both)
  acquisition     variance (Active E-SINDy) | dopt (How Low's D-optimal, noise-blind) |
                  iopt (ours: I-optimal on noise-whitened rows) | random (reference only, not active)
Our method: esindy_wt+iopt (points), esindy_t+iopt (pde), esindy_wt+variance (traj: I-optimal needs the
candidate trajectories predicted by the current model, and that prediction is the weak point).
No arm sees a candidate's label before measuring it.

    python src/experiments.py points|traj|pde [n_seeds] [systems] [noise levels] [arms] [out.json]
    python src/experiments.py show results.json
"""
import json
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

import baselines as bl
import noise_aware as na

ODE_ARMS = ["esindy+variance", "esindy+dopt", "esindy+random", "esindy_w+random", "esindy_t+random",
            "esindy_wt+random", "esindy_wt+variance", "esindy_wt+iopt"]
PDE_ARMS = ["esindy+variance", "esindy+dopt", "esindy+random",
            "esindy_t+random", "esindy_t+variance", "esindy_t+iopt"]
DEFAULT = dict(points=(["lorenz", "vanderpol", "hopf", "lotka"], [0.05, 0.1, 0.2], ODE_ARMS),
               traj=(["lorenz", "vanderpol", "hopf", "lotka"], [0.05, 0.1], ODE_ARMS),
               pde=(["burgers", "ks"], [0.5, 2.0], PDE_ARMS))


# ---------------------------------------------------------------- systems
SYSTEMS = dict(     # true terms, box (lo, hi), reference state, sampling step and RK4 sub-steps
    lorenz=({("x", 0): -10, ("y", 0): 10, ("x", 1): 28, ("y", 1): -1, ("x*z", 1): -1, ("x*y", 2): 1, ("z", 2): -8 / 3},
            [-25, -25, 0], [25, 25, 50], [-8, 7, 27], 0.01, 2),          # box and dt as in the Active E-SINDy code
    chen=({("x", 0): -35, ("y", 0): 35, ("x", 1): -7, ("y", 1): 28, ("x*z", 1): -1, ("x*y", 2): 1, ("z", 2): -3},
          [-25, -25, 0], [25, 25, 50], [-5, -5, 20], 0.005, 2),
    vanderpol=({("y", 0): 1, ("x", 1): -1, ("y", 1): 2, ("x*x*y", 1): -2}, [-3, -4], [3, 4], [2, 0], 0.1, 5),
    duffing=({("y", 0): 1, ("x", 1): 1, ("y", 1): -0.5, ("x*x*x", 1): -1}, [-2, -2], [2, 2], [1.5, 0], 0.1, 5),
    hopf=({("x", 0): 1, ("y", 0): -1, ("x*x*x", 0): -1, ("x*y*y", 0): -1,
           ("x", 1): 1, ("y", 1): 1, ("x*x*y", 1): -1, ("y*y*y", 1): -1}, [-2, -2], [2, 2], [1, 0], 0.1, 5),
    lotka=({("x", 0): 1, ("x*y", 0): -1, ("y", 1): -1, ("x*y", 1): 1}, [0, 0], [3, 3], [1.5, 1], 0.1, 5))


def make_system(name, noise, n_points=10):
    terms, lo, hi, ref, dt, every = SYSTEMS[name]
    lib = na.PolyLibrary(len(lo), 3, ["x", "y", "z"][: len(lo)])
    s = dict(lo=lo, hi=hi, dt=dt, every=every, ref=ref)
    C = np.zeros((lib.p, lib.d))
    for (t, c), v in terms.items():
        C[lib.names.index(t), c] = v
    lo, hi = np.array(s["lo"], float), np.array(s["hi"], float)
    sim = dict(h=s["dt"] / s["every"], n_steps=(n_points - 1) * s["every"], every=s["every"],
               lo=lo - 3 * (hi - lo), hi=hi + 3 * (hi - lo))
    ref = na.simulate(lib, C, np.array([s["ref"]], float), **{**sim, "n_steps": 2000 * s["every"]})
    return dict(lib=lib, C=C, lo=lo, hi=hi, dt=s["dt"], sim=sim, sigma=noise * np.sqrt(np.mean(ref ** 2)))


# --------------------------------------------------------------- protocols
class Points:
    lam, n_init, n_steps, n_cand, block, per = 0.2, 20, 80, 200, 1, 1

    def __init__(self, env, system, noise):
        self.__dict__.update(make_system(system, noise), env=env)
        self.truth = self.C != 0

    def _states(self, n):
        return self.lo + (self.hi - self.lo) * self.env.random((n, self.lib.d))

    def _measure(self, X):
        noisy = X + self.sigma * self.env.standard_normal(X.shape)
        return self.lib(noisy), self.lib(X) @ self.C, self.lib.grad(noisy), None

    def initial(self):
        return self._measure(self._states(self.n_init))

    def candidates(self, eqs):                  # fresh every step, as in the original code
        self.X = self._states(self.n_cand)
        return (self.lib(self.X)[:, None], None, self.lib.grad(self.X)[:, None], None), np.arange(self.n_cand)

    def measure(self, i):
        return self._measure(self.X[i:i + 1])


class Trajectories:
    lam, n_steps, n_cand, n_init, per, n_points = 0.2, 9, 100, 10, 10, 10

    def __init__(self, env, system, noise):
        self.__dict__.update(make_system(system, noise, self.n_points), env=env)
        self.truth = self.C != 0
        self.actions = qmc.scale(qmc.LatinHypercube(self.lib.d, seed=env).random(self.n_cand), self.lo, self.hi)
        self.free = np.ones(self.n_cand, bool)
        self.Ys = []                            # raw noisy trajectories, for identifiers that need them

    def _measure(self, x0):
        y = na.simulate(self.lib, self.C, x0[None], **self.sim)
        y = y + self.sigma * self.env.standard_normal(y.shape)
        self.Ys.append(y[0])
        return na.flat(na.integral_rows(self.lib, y, self.dt, self.window))

    def initial(self):
        return self._measure(self.env.uniform(self.lo, self.hi))

    def candidates(self, eqs):                  # what the current model predicts each experiment would record
        idx = np.nonzero(self.free)[0]
        C = np.zeros_like(self.C)
        for e in eqs or []:
            C[e["support"], e["c"]] = e["xi"]
        return na.integral_rows(self.lib, na.simulate(self.lib, C, self.actions[idx], **self.sim), self.dt, self.window), idx

    def measure(self, i):
        self.free[i] = False
        return self._measure(self.actions[i])


def etdrk4(u0, L, k, dt, n_steps, every):
    """u_t = L u - (u^2/2)_x, pseudo-spectral ETDRK4 (Kassam & Trefethen 2005)."""
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, 17) - 0.5) / 16)
    LR = dt * L[:, None] + r[None]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, 1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, 1))
    N = lambda v: -0.5j * k * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
    v, out = np.fft.fft(u0), []
    for s in range(n_steps):
        if s % every == 0:
            out.append(v)
        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + N(c) * f3
    return np.array(out)


class PDEPoints:
    """Library: {1, u, u^2} x {u_x, u_xx, u_xxx, u_xxxx} plus u, u^2 (14 terms)."""
    lam, n_pool, block, per = 0.2, 5000, 1, 1
    names = ["u", "u^2"] + [f"{a}{d}" for d in ["u_x", "u_xx", "u_xxx", "u_xxxx"]
                            for a in ["", "u*", "u^2*"]]

    def __init__(self, env, kind, noise):
        if kind == "burgers":       # u_t = -u u_x + 0.1 u_xx
            n, length, self.n_steps = 256, 16.0, 34
            x = np.arange(n) * length / n - 8
            k = 2 * np.pi * np.fft.fftfreq(n, length / n)
            L, u0, dt, steps, every, skip = -0.1 * k ** 2, np.exp(-(x + 2) ** 2), 0.01, 1000, 10, 0
            true = {"u*u_x": -1, "u_xx": 0.1}
        else:                       # Kuramoto-Sivashinsky: u_t = -u u_x - u_xx - u_xxxx
            n, length, self.n_steps = 128, 32 * np.pi, 84
            x = np.arange(n) * length / n
            k = 2 * np.pi * np.fft.fftfreq(n, length / n)
            L, u0 = k ** 2 - k ** 4, np.cos(x / 16) * (1 + np.sin(x / 16))
            dt, steps, every, skip = 0.25, 600, 2, 50
            true = {"u*u_x": -1, "u_xx": -1, "u_xxxx": -1}
        V = etdrk4(u0, L, k, dt, steps, every)[skip:]
        u = np.real(np.fft.ifft(V)).ravel()
        D = [np.real(np.fft.ifft((1j * k) ** m * V)).ravel() for m in range(1, 5)]
        F = np.stack([u, u ** 2] + [p * d for d in D for p in (1, u, u ** 2)], 1)
        xi = np.zeros(len(self.names))
        for t, v in true.items():
            xi[self.names.index(t)] = v
        pool = env.choice(len(u), self.n_pool, replace=False)
        clean = F[pool] @ xi
        scale = np.sqrt(np.mean(clean ** 2))
        self.F = F[pool] / F.std(0)                       # unit-variance features, unit-RMS target
        self.y = (clean / scale + noise * env.standard_normal(self.n_pool))[:, None]
        self.truth = (xi != 0)[:, None]
        self.n_init = len(self.names) + 2
        self.free = np.ones(self.n_pool, bool)

    def initial(self):
        self.free[: self.n_init] = False                  # pool order is already random
        return self.F[: self.n_init], self.y[: self.n_init], None, None

    def candidates(self, eqs):
        idx = np.nonzero(self.free)[0]
        return (self.F[idx][:, None], None, None, None), idx

    def measure(self, i):
        self.free[i] = False
        return self.F[i:i + 1], self.y[i:i + 1], None, None


PROTOCOLS = dict(points=Points, traj=Trajectories, pde=PDEPoints)


# --------------------------------------------------------------- one trial
def run_trial(args):
    protocol, system, noise, arm, seed = args
    ident, acq = arm.split("+")
    flags = ident.partition("_")[2]            # w: noise weights, t: significance test, b: ARD inside the ensemble
    env = np.random.default_rng([seed, 0])                    # shared by all arms
    rng = np.random.default_rng([seed, zlib.crc32(arm.encode())])
    prob = PROTOCOLS[protocol](env, system, noise)
    baseline = ident in ("wsindy", "sbl", "wsbl", "msindy")     # literature identifiers (baselines.py)
    if protocol == "traj":      # whitening makes the shortest window optimal; unweighted rows need a long one
        prob.window = 1 if ("w" in flags or baseline) else 5
        prob.block = prob.n_points - prob.window
    rows = prob.initial()
    checks = set(np.unique(np.linspace(0, prob.n_steps, 9).astype(int)).tolist()) | {prob.n_steps}
    if protocol == "traj":
        checks = set(range(prob.n_steps + 1))
    log = []
    for step in range(prob.n_steps + 1):
        eqs = None
        if acq != "random" or step in checks:
            eqs = (bl.identify(ident, rows, prob, rng) if baseline else
                   na.identify(rows, rng, prob.lam, "w" in flags, "t" in flags, prob.block,
                               solver="ard" if "b" in flags else "stlsq"))
        if step in checks:
            mask = np.zeros_like(prob.truth)
            for e in eqs:
                mask[e["support"], e["c"]] = True
            log.append(dict(n=prob.n_init + step * prob.per, exact=bool((mask == prob.truth).all()),
                            extra=int((mask & ~prob.truth).sum()), missing=int((~mask & prob.truth).sum())))
        if step == prob.n_steps:
            break
        cand, idx = prob.candidates(eqs)
        score = (na.variance_score(eqs, cand) if acq == "variance" else
                 na.prediction_score(eqs, cand) if acq == "iopt" else
                 na.dopt_score(eqs, cand) if acq == "dopt" else rng.random(len(idx)))
        rows = na.stack(rows, prob.measure(idx[int(np.argmax(np.nan_to_num(score, nan=-np.inf)))]))
    return dict(protocol=protocol, system=system, noise=noise, arm=arm, seed=seed, log=log)


def show(results):
    cfgs = list(dict.fromkeys((t["system"], t["noise"]) for t in results))
    arms = list(dict.fromkeys(t["arm"] for t in results))
    for s, nz in cfgs:
        H = {a: np.array([[r["exact"] for r in t["log"]] for t in sorted(
            (t for t in results if (t["system"], t["noise"], t["arm"]) == (s, nz, a)),
            key=lambda t: t["seed"])], float) for a in arms}
        ns = [r["n"] for r in next(t for t in results if (t["system"], t["noise"]) == (s, nz))["log"]]
        print(f"\n== {s}, noise {100 * nz:g}%: exact-support recovery (%) by number of measured points")
        print(f"{'identification+acquisition':22s}" + "".join(f"{n:>5d}" for n in ns) + "   mean")
        for a, h in H.items():
            print(f"{a:22s}" + "".join(f"{100 * v:5.0f}" for v in h.mean(0)) + f"   {100 * h.mean():.1f}")
        full, base = arms[-1], [a for a in arms if a != arms[-1]]
        print(f"paired difference of {full} to each other arm (percentage points, +- 1 s.e.)")
        for b in base:
            d = H[full].mean(1) - H[b].mean(1)
            print(f"  - {b:20s} {100 * d.mean():+6.1f} +- {100 * d.std() / np.sqrt(len(d)):.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "show":
        show(json.load(open(sys.argv[2])))
        sys.exit()
    protocol = sys.argv[1]
    arg = lambda i, default, f=str: [f(v) for v in sys.argv[i].split(",")] if len(sys.argv) > i and sys.argv[i] != "-" else default
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    systems, noises, arms = arg(3, DEFAULT[protocol][0]), arg(4, DEFAULT[protocol][1], float), arg(5, DEFAULT[protocol][2])
    out = sys.argv[6] if len(sys.argv) > 6 else f"results/{protocol}.json"
    jobs = [(protocol, s, nz, a, sd) for s in systems for nz in noises for a in arms for sd in range(n_seeds)]
    t0 = time.time()
    with Pool() as pool:
        results = pool.map(run_trial, jobs, chunksize=1)
    json.dump(results, open(out, "w"))
    print(f"{len(jobs)} trials in {time.time() - t0:.0f}s")
    show(results)

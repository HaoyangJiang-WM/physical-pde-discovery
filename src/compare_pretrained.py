"""Passive identification from one noisy trajectory: E-SINDy, our identification, and pretrained models.

Each seed: one trajectory of 100 noisy points from a random initial state in the sampling box
(systems and noise as in experiments.py).  Every method gets exactly the same data.
  esindy      original E-SINDy (double bagging, lambda = 0.2) on integral rows, window 5
  ours        whitening + significance test on integral rows, window 1, whole trajectory as one block
  odeformer   pretrained ODEFormer (d'Ascoli et al. 2023), beam size 50, no training or tuning
Structure is "exact" when the expanded equations contain exactly the true monomials; any
non-polynomial term (sin, 1/x, ...) counts as an extra term.

    python src/compare_pretrained.py ODEFORMER_PKG_DIR ODEFORMER_WEIGHTS [n_seeds] [out.json]
"""
import json
import pathlib
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import sympy as sp

import experiments as ex
import noise_aware as na

SYSTEMS = ["lorenz", "chen", "vanderpol", "duffing", "hopf", "lotka"]
NOISES = [0.05, 0.1]
N_POINTS = 100
_REG = {}


def data(system, noise, seed):
    s = ex.make_system(system, noise, n_points=N_POINTS)
    rng = np.random.default_rng([seed, 7])
    x0 = rng.uniform(s["lo"], s["hi"])
    Y = na.simulate(s["lib"], s["C"], x0[None], **s["sim"])[0]
    return s, Y + s["sigma"] * rng.standard_normal(Y.shape)


def true_terms(s):
    names = s["lib"].names
    return [frozenset(names[j] for j in np.nonzero(s["C"][:, c])[0]) for c in range(s["lib"].d)]


def sindy(args):
    system, noise, seed, arm = args
    s, Yn = data(system, noise, seed)
    rng = np.random.default_rng([seed, zlib.crc32(arm.encode())])
    w = 5 if arm == "esindy" else 1
    rows = na.flat(na.integral_rows(s["lib"], Yn[None], s["dt"], w))
    eqs = na.identify(rows, rng, 0.2, arm == "ours", arm == "ours", block=N_POINTS - w)
    found = [frozenset(s["lib"].names[j] for j in e["support"]) for e in eqs]
    return dict(system=system, noise=noise, seed=seed, arm=arm, found=[sorted(f) for f in found],
                exact=found == true_terms(s))


def monomials(expr_str, d, names):
    """Expanded equation -> set of monomial names in the library's spelling; non-polynomial terms -> 'other'."""
    syms = sp.symbols(f"x_0:{d}")
    var = dict(zip(syms, names[:d]))
    expr = sp.expand(sp.sympify(expr_str, locals={f"x_{i}": syms[i] for i in range(d)}))
    terms = sp.Add.make_args(expr)
    coefs = []
    out = {}
    for t in terms:
        c, m = t.as_coeff_Mul()
        powers = m.as_powers_dict() if m != 1 else {}
        if m == 1:
            key = "1"
        elif all(k in var for k in powers) and all(float(p).is_integer() and p > 0 for p in powers.values()):
            key = "*".join(var[k] for k in syms if k in powers for _ in range(int(powers[k])))
        else:
            key = "other:" + str(m)[:30]
        out[key] = out.get(key, 0.0) + float(c)
        coefs.append(abs(float(c)))
    top = max([abs(v) for v in out.values()] + [1e-300])
    return frozenset(k for k, v in out.items() if abs(v) > 1e-6 * top)


def init_odeformer(pkg, weights):
    sys.path.insert(0, pkg)
    import torch
    pathlib.PosixPath = pathlib.WindowsPath                 # checkpoint was saved on Linux
    torch.set_num_threads(4)
    from odeformer.model import SymbolicTransformerRegressor
    model = torch.load(weights, weights_only=False, map_location="cpu")
    reg = SymbolicTransformerRegressor(model=model, from_pretrained=False)
    reg.set_model_args({"beam_size": 50, "beam_type": "sampling", "beam_temperature": 0.1})
    _REG["reg"] = reg


def odeformer(args):
    system, noise, seed = args
    s, Yn = data(system, noise, seed)
    reg = _REG["reg"]
    t = np.arange(N_POINTS) * s["dt"]
    try:
        reg.fit(t, Yn)
        eqs = reg.predictions[0][0].infix().split("|")
        names = ["x", "y", "z"]
        lib_names = {n: n for n in s["lib"].names}
        found = [monomials(e, s["lib"].d, names) for e in eqs]
        text = [e.strip() for e in eqs]
    except Exception as err:                                 # no valid prediction
        found, text = [frozenset({"failed"})] * s["lib"].d, [repr(err)[:80]]
    return dict(system=system, noise=noise, seed=seed, arm="odeformer", found=[sorted(f) for f in found],
                exact=found == true_terms(s), text=text)


if __name__ == "__main__":
    pkg, weights = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    out = sys.argv[4] if len(sys.argv) > 4 else "results/pretrained.json"
    t0 = time.time()
    jobs = [(s, nz, sd, a) for s in SYSTEMS for nz in NOISES for sd in range(n) for a in ("esindy", "ours")]
    with Pool() as pool:
        R = pool.map(sindy, jobs, chunksize=1)
    print(f"SINDy arms: {len(jobs)} fits in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    with Pool(6, initializer=init_odeformer, initargs=(pkg, weights)) as pool:
        R += pool.map(odeformer, [(s, nz, sd) for s in SYSTEMS for nz in NOISES for sd in range(n)], chunksize=1)
    print(f"ODEFormer: {len(SYSTEMS) * len(NOISES) * n} fits in {time.time() - t0:.0f}s")
    json.dump(R, open(out, "w"))
    arms = ["esindy", "ours", "odeformer"]
    print(f"\nexact structure from one trajectory of {N_POINTS} noisy points (% of {n} seeds)")
    print(f"{'system':12s}{'noise':>6s}" + "".join(f"{a:>11s}" for a in arms))
    for s in SYSTEMS:
        for nz in NOISES:
            print(f"{s:12s}{int(nz*100):5d}%" + "".join(f"{100*np.mean([t['exact'] for t in R if (t['system'], t['noise'], t['arm']) == (s, nz, a)]):11.1f}" for a in arms))

"""Acquisition chooses the experimental range, not only points inside a fixed box.

Point protocol (as in experiments.Points / domains.DomainPoints) except that each step's 200
candidates are drawn from a wider feasible range.  One rule for every model and every method:
each side of the original box moves out by `grow` x its width, except a side at a physical
bound of 0 (concentrations, populations) which stays.  The first 20 points, the library scaling
and the sensor noise (absolute) are those of the original box.  Reported per arm: recovery, and
the share of acquired points outside the original box.

    python src/expand.py [n_seeds] [systems] [noise levels] [arms] [grow] [out.json]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import domains as dm
import experiments as ex

GROW = 0.5
LAST = {}                        # the problem instance of the trial running in this process


def wide_box(lo, hi, grow):
    w = hi - lo
    return np.where(lo == 0, lo, lo - grow * w), hi + grow * w


class _Wide:
    def _widen(self, grow):
        self.wlo, self.whi = wide_box(self.lo, self.hi, grow)
        self.outside = []
        LAST["p"] = self

    def candidates(self, eqs):
        self.X = self.wlo + (self.whi - self.wlo) * self.env.random((self.n_cand, self.lib.d))
        return (self.lib(self.X)[:, None], None, self.lib.grad(self.X)[:, None], None), np.arange(self.n_cand)

    def measure(self, i):
        self.outside.append(bool(((self.X[i] < self.lo) | (self.X[i] > self.hi)).any()))
        return super().measure(i)


class WideDomain(_Wide, dm.DomainPoints):
    def __init__(self, env, system, noise):
        super().__init__(env, system, noise); self._widen(GROW)


class WideODE(_Wide, ex.Points):
    def __init__(self, env, system, noise):
        super().__init__(env, system, noise); self._widen(GROW)


ex.PROTOCOLS["wide"] = lambda env, system, noise: (WideDomain if system in dm.MODELS else WideODE)(env, system, noise)


def run(args):
    global GROW
    grow, job = args
    GROW = grow
    r = ex.run_trial(job)
    r["outside"] = float(np.mean(LAST["p"].outside))
    return r


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    arg = lambda i: sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "-" else None
    systems = arg(2).split(",") if arg(2) else list(dm.MODELS)
    noises = [float(v) for v in arg(3).split(",")] if arg(3) else [0.05, 0.1]
    arms = arg(4).split(",") if arg(4) else ["esindy+variance", "esindy+dopt", "esindy_wt+variance", "esindy_wt+iopt"]
    grow = float(arg(5)) if arg(5) else GROW
    out = sys.argv[6] if len(sys.argv) > 6 else "results/wide.json"
    jobs = [(grow, ("wide", s, nz, a, sd)) for s in systems for nz in noises for a in arms for sd in range(n)]
    t0 = time.time()
    with Pool() as pool:
        R = pool.map(run, jobs, chunksize=1)
    json.dump(R, open(out, "w"))
    print(f"{len(jobs)} trials in {time.time() - t0:.0f}s   (each side of the box grows by {grow} x its width)")
    for s in systems:
        for nz in noises:
            print(f"\n== {s} {int(nz * 100)}%   recovery over the curve / at 100 points / share of picks outside the original box")
            for a in arms:
                T = [t for t in R if (t["system"], t["noise"], t["arm"]) == (s, nz, a)]
                print(f"  {a:20s} {100*np.mean([[r['exact'] for r in t['log']] for t in T]):6.1f} / "
                      f"{100*np.mean([t['log'][-1]['exact'] for t in T]):6.1f} / {np.mean([t['outside'] for t in T]):.2f}")

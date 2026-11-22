"""Markdown tables of the final comparison, computed from results/*.json.

    python src/report.py            # all settings
    python src/report.py graph      # one setting

Metric: exact-support recovery averaged over the acquisition curve (%), then over seeds.
better / tie / worse: paired difference over common seeds beyond 2 standard errors.
"""
import json
import os
import sys

import numpy as np

R_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ACTIVE_ESINDY, DOPT, SBL = "esindy+variance", "esindy+dopt", "sbl+iopt"
SETTINGS = {   # name: (result file, protocol in baselines.json, our arm, competitors, reference arms)
    "points": ("points", "points", "esindy_wt+iopt", [ACTIVE_ESINDY, DOPT, SBL], ["esindy_wt+random"]),
    "domains": ("domains", "domain", "esindy_wt+iopt", [ACTIVE_ESINDY, DOPT, SBL], []),
    "wide": ("wide", "wide", "esindy_wt+iopt", [ACTIVE_ESINDY, DOPT, SBL], []),
    "traj": ("traj", "traj", "esindy_wt+variance", [ACTIVE_ESINDY, SBL, "msindy+iopt", "wsindy+iopt"], ["esindy_wt+random"]),
    "pde": ("pde", "pde", "esindy_t+iopt", [ACTIVE_ESINDY, DOPT, SBL], ["esindy_t+random"]),
    "graph": ("graph", "graph", "esindy_wt+iopt", [ACTIVE_ESINDY, DOPT, SBL], ["esindy_wt+random"]),
}
LABEL = {ACTIVE_ESINDY: "Active E-SINDy", DOPT: "D-optimal", SBL: "Bayesian SINDy + active",
         "msindy+iopt": "Modified SINDy + active", "wsindy+iopt": "Weak SINDy + active",
         "esindy_wt+random": "our identification + random", "esindy_t+random": "our identification + random"}


def load(setting):
    f, proto, ours, comp, ref = SETTINGS[setting]
    R = json.load(open(os.path.join(R_DIR, f + ".json")))
    B = [r for r in json.load(open(os.path.join(R_DIR, "baselines.json"))) if r.get("protocol") == proto]
    score = {}
    for r in R + B:
        score.setdefault((r["system"], r["noise"], r["arm"]), {})[r["seed"]] = np.mean([s["exact"] for s in r["log"]])
    cfgs = list(dict.fromkeys((r["system"], r["noise"]) for r in R))
    return cfgs, score, ours, comp, ref


def table(setting):
    cfgs, score, ours, comp, ref = load(setting)
    comp = [a for a in comp if any((s, n, a) in score for s, n in cfgs)]
    cols = comp + [ours] + ref
    out = ["| system | noise | " + " | ".join(LABEL.get(a, "**ours**" if a == ours else a) for a in cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    tally = {a: [0, 0, 0] for a in comp + ref}
    means = {a: [] for a in cols}
    for s, n in cfgs:
        row = []
        vals = {a: np.mean(list(score[(s, n, a)].values())) * 100 for a in cols if (s, n, a) in score}
        best = max(vals.values())
        for a in cols:
            if a not in vals:
                row.append("–"); continue
            means[a].append(vals[a])
            row.append(f"**{vals[a]:.1f}**" if vals[a] >= best - 1e-9 else f"{vals[a]:.1f}")
            if a != ours:
                o, b = score[(s, n, ours)], score[(s, n, a)]
                common = sorted(set(o) & set(b))
                d = np.array([o[k] - b[k] for k in common])
                se = d.std() / np.sqrt(len(d)) if len(d) > 1 else 0
                tally[a][0 if d.mean() > 2 * se and d.mean() > 0 else 2 if d.mean() < -2 * se and d.mean() < 0 else 1] += 1
        out.append(f"| {s} | {100 * n:g}% | " + " | ".join(row) + " |")
    out.append("| **mean** | | " + " | ".join(f"{np.mean(means[a]):.1f}" if means[a] else "–" for a in cols) + " |")
    out.append("| ours: better / tie / worse | | " + " | ".join("/".join(map(str, tally[a])) if a in tally else "" for a in cols) + " |")
    return "\n".join(out), {a: tally[a] for a in comp}, len(cfgs)


if __name__ == "__main__":
    for s in sys.argv[1:] or SETTINGS:
        t, tally, n = table(s)
        print(f"\n### {s} ({n} configurations)\n\n{t}")

"""python src/plot.py results/points.json   ->  figs/points.png   (recovery curves, one panel per configuration)"""
import os
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]
res = json.load(open(path))
mkey = "arm" if "arm" in res[0] else "method"
group = (lambda t: f'{t["system"]}, noise {100 * t["noise"]:g}%') if "noise" in res[0] else \
        (lambda t: t["system" if "system" in t else "problem"])
groups = list(dict.fromkeys(group(t) for t in res))
methods = list(dict.fromkeys(t[mkey] for t in res))
ncols = len({t["noise"] for t in res}) if "noise" in res[0] else len(groups)
style = {"ensemble_gain": ("#c0392b", "-", 2.5),
         "variance": ("#2c6fbb", "--", 1.5), "dopt_fixed": ("#2e8b57", "-.", 1.5),
         "leverage": ("#2e8b57", "-.", 1.5), "random": ("#888888", ":", 1.5),
         "esindy_wt+iopt": ("#c0392b", "-", 2.5), "esindy_t+iopt": ("#c0392b", "-", 2.5), "esindy+dopt": ("#6c8ebf", "-.", 1.5), "esindy_wt+info": ("#bbbbbb", "-", 1.0), "esindy_wt+random": ("#e08e0b", "-", 1.5),
         "esindy_wt+variance": ("#8e44ad", "--", 1.2), "esindy+variance": ("#2c6fbb", "--", 1.5),
         "esindy+random": ("#888888", ":", 1.5), "esindy_w+random": ("#2e8b57", "-.", 1.0),
         "esindy_t+random": ("#16a085", ":", 1.0), "esindy+info": ("#2e8b57", "-.", 1.2),
         "esindy_t+info": ("#c0392b", "-", 2.5), "esindy_t+variance": ("#8e44ad", "--", 1.2)}
nrows = -(-len(groups) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
for ax, name in zip(axes.ravel(), groups):
    for m in methods:
        logs = [t["log"] for t in res if group(t) == name and t[mkey] == m]
        hits = np.array([[s["exact"] if isinstance(s, dict) else s for s in log] for log in logs], float)
        mean, se = hits.mean(0), hits.std(0) / np.sqrt(len(hits))
        c, ls, lw = style.get(m, ("k", "-", 1.5))
        x = [s["n"] for s in logs[0]] if isinstance(logs[0][0], dict) and "n" in logs[0][0] else np.arange(len(mean))
        ax.plot(x, 100 * mean, ls, color=c, lw=lw, label=m)
        ax.fill_between(x, 100 * (mean - se), 100 * (mean + se), color=c, alpha=0.12)
    ax.set_title(f"{name} (n={len(hits)})")
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("measured points" if mkey == "arm" else "acquisitions")
for ax in axes[:, 0]:
    ax.set_ylabel("exact support recovery (%)")
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join("figs", os.path.basename(path).replace(".json", ".png")), dpi=160)

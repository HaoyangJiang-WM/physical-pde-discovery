"""Figures used in the slides (slides/), from results/*.json.

    python src/slide_figs.py            ->  figs/slide_*.png

p3_where re-runs Active E-SINDy and random sampling on Lorenz 10% (16 seeds each) to record where they sample.
"""
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as dm
import experiments as ex
import noise_aware as na

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
INK, MUTE, BG = "#14213D", "#5B6577", "#FBFBF8"
OURS, BASE, RAND, BAD, GRN, PUR = "#D9661F", "#2C6FBB", "#A9B0BA", "#B23A48", "#6C8EBF", "#9B7BB8"
plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False, "font.size": 20,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "legend.frameon": False})
load = lambda name: json.load(open(os.path.join(RES, name + ".json"), encoding="utf8"))
save = lambda fig, name: (fig.tight_layout(), fig.savefig(os.path.join(OUT, f"slide_{name}.png"), dpi=150), plt.close(fig))
logs = lambda R, s, nz, a: [t["log"] for t in R if (t["system"], t["noise"], t["arm"]) == (s, nz, a)]
rec = lambda R, s, nz, a: 100 * np.mean([[r["exact"] for r in l] for l in logs(R, s, nz, a)])
THREE = [("esindy+variance", "Active E-SINDy", BASE), ("esindy+dopt", "D-optimal (How Low)", GRN), ("esindy_wt+iopt", "ours", OURS)]


def bars(R, cfg, arms, name, size=(13, 6.2), fs=15, ylabel="recovery (%)"):
    fig, ax = plt.subplots(figsize=size); x = np.arange(len(cfg)); w = 0.8 / len(arms)
    for j, (a, lab, c) in enumerate(arms):
        v = [rec(R, s, nz, a) for _, s, nz in cfg]
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, v, w, color=c, label=lab)
        for i, u in enumerate(v):
            ax.text(i + (j - (len(arms) - 1) / 2) * w, u + 1.5, f"{u:.0f}", ha="center", fontsize=fs)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cfg]); ax.set_ylabel(ylabel); ax.set_ylim(0, 120)
    ax.legend(loc="upper center", ncol=len(arms), bbox_to_anchor=(0.5, 1.03), fontsize=16)
    save(fig, name)


def picks(args):
    acq, seed = args
    arm = "esindy+" + acq
    env = np.random.default_rng([seed, 0]); rng = np.random.default_rng([seed, zlib.crc32(arm.encode())])
    prob = ex.Points(env, "lorenz", 0.1)
    rows = prob.initial(); out = []
    for _ in range(prob.n_steps):
        cand, idx = prob.candidates(None)
        if acq == "variance":
            k = int(np.argmax(na.variance_score(na.identify(rows, rng, prob.lam, False, False), cand)))
        else:
            k = int(rng.integers(len(idx)))
        out.append(prob.X[k]); rows = na.stack(rows, prob.measure(idx[k]))
    return acq, np.array(out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    P, T, D, W, Dm = (load(k) for k in ("points", "traj", "pde", "wide", "domains"))

    # problem 1: failures are extra terms
    cfg = [("points\nLorenz 5%", P, "lorenz", 0.05), ("points\nLorenz 10%", P, "lorenz", 0.1), ("trajectory\nLorenz 5%", T, "lorenz", 0.05),
           ("trajectory\nHopf 10%", T, "hopf", 0.1), ("PDE\nBurgers 50%", D, "burgers", 0.5), ("PDE\nKS 100%", D, "ks", 1.0)]
    ext = [np.mean([l[-1]["extra"] for l in logs(R, s, nz, "esindy+variance")]) for _, R, s, nz in cfg]
    mis = [np.mean([l[-1]["missing"] for l in logs(R, s, nz, "esindy+variance")]) for _, R, s, nz in cfg]
    fig, ax = plt.subplots(figsize=(13, 6.2)); x = np.arange(len(cfg))
    ax.bar(x - 0.2, ext, 0.4, color=BAD, label="extra terms"); ax.bar(x + 0.2, mis, 0.4, color=RAND, label="missing terms")
    for i, (a, b) in enumerate(zip(ext, mis)):
        ax.text(i - 0.2, a + 0.08, f"{a:.1f}", ha="center"); ax.text(i + 0.2, b + 0.08, f"{b:.1f}", ha="center", color=MUTE)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cfg], fontsize=17); ax.set_ylabel("mean number of terms\nat the end of acquisition")
    ax.legend(loc="upper left"); save(fig, "p1_extra")

    # problem 2: residual noise is not constant (Lorenz dy/dt)
    env = np.random.default_rng(0); prob = ex.Points(env, "lorenz", 0.05)
    X = prob._states(20000); Xn = X + prob.sigma * env.standard_normal(X.shape)
    r = prob.lib(X) @ prob.C[:, 1] - prob.lib(Xn) @ prob.C[:, 1]
    g = np.linalg.norm(np.einsum("npd,p->nd", prob.lib.grad(Xn), prob.C[:, 1]), axis=1)
    q = np.quantile(g, np.linspace(0, 1, 21)); mid, sd = [], []
    for a, b in zip(q[:-1], q[1:]):
        m = (g >= a) & (g <= b); mid.append(g[m].mean()); sd.append(r[m].std())
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.scatter(g[:3000], np.abs(r[:3000]), s=8, color=RAND, alpha=0.5, label="|residual| of one measurement")
    ax.plot(mid, sd, "o-", color=BAD, lw=3, ms=9, label="residual sd per bin")
    ax.plot([0, max(mid)], [0, prob.sigma * max(mid)], "--", color=INK, lw=2.5, label="formula: σ·‖∂f/∂x‖")
    ax.axhline(r.std(), color=BASE, lw=2.5, ls=":", label="E-SINDy assumption: constant")
    ax.set_xlabel("Jacobian norm ‖∂f/∂x‖ (Lorenz dy/dt)"); ax.set_ylabel("residual"); ax.set_ylim(0, np.quantile(np.abs(r), 0.995))
    ax.legend(loc="upper left", fontsize=17); save(fig, "p2_noise")

    # problem 2 in other fields: spread of the noise level inside each equation
    S = [d for d in load("noise_spread") if d["noise"] == 0.05]
    col = {"biology": "#2C6FBB", "chemistry": "#D9661F", "materials": "#3B8F6B"}
    fig, ax = plt.subplots(figsize=(13, 6.2)); seen = set()
    for i, d in enumerate(S):
        ax.bar(i, d["spread"], 0.7, color=col[d["field"]], label=None if d["field"] in seen else d["field"]); seen.add(d["field"])
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=1.5, ls="--"); ax.set_ylim(0.8, 80)
    ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    names, last = [], None
    for d in S:
        names.append(d["name"] if d["name"] != last else ""); last = d["name"]
    ax.set_xticks(range(len(S))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=15)
    ax.set_ylabel("noise sd within one equation\nmax ÷ min (×)"); ax.legend(loc="upper right", ncol=3); save(fig, "p2_domains")

    # problem 3: variance acquisition goes where the noise is large
    with Pool(8) as pool:
        res = pool.map(picks, [(a, s) for a in ("variance", "random") for s in range(16)])
    Pk = {a: np.concatenate([r for n, r in res if n == a]) for a in ("variance", "random")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    xx, zz = np.meshgrid(np.linspace(-25, 25, 200), np.linspace(0, 50, 200))
    for ax, a, title in zip(axes, ("random", "variance"), ("random sampling", "Active E-SINDy (ensemble variance)")):
        cs = ax.contourf(xx, zz, np.sqrt((28 - zz) ** 2 + 1 + xx ** 2), levels=12, cmap="OrRd", alpha=0.85)
        ax.scatter(Pk[a][:, 0], Pk[a][:, 2], s=14, color=INK, alpha=0.75, lw=0)
        ax.set_title(title, fontsize=20); ax.set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.tight_layout(rect=(0, 0, 0.88, 1)); cb = fig.colorbar(cs, cax=fig.add_axes([0.9, 0.17, 0.015, 0.73])); cb.set_label("local noise factor ‖∂f/∂x‖")
    fig.savefig(os.path.join(OUT, "slide_p3_where.png"), dpi=150); plt.close(fig)

    # problem 3 consequence: active worse than random at high noise
    bars(P, [("Van der Pol\n10%", "vanderpol", 0.1), ("Hopf\n10%", "hopf", 0.1), ("Van der Pol\n20%", "vanderpol", 0.2), ("Lotka-Volterra\n20%", "lotka", 0.2)],
         [("esindy+random", "E-SINDy + random", RAND), ("esindy+variance", "Active E-SINDy (ensemble variance)", BASE)], "p3b_worse", fs=18)

    # results
    bars(P, [("Lorenz\n10%", "lorenz", .1), ("Lorenz\n20%", "lorenz", .2), ("Chen\n10%", "chen", .1), ("Chen\n20%", "chen", .2),
             ("V. d. Pol\n20%", "vanderpol", .2), ("Duffing\n20%", "duffing", .2), ("Hopf\n20%", "hopf", .2), ("Lotka-V.\n20%", "lotka", .2)],
         THREE, "r_points")
    bars(W, [("Lorenz\n20%", "lorenz", .2), ("V. d. Pol\n20%", "vanderpol", .2), ("Selkov\n10%", "selkov", .1), ("Brussel.\n10%", "brusselator", .1),
             ("Arrhenius\n5%", "arrhenius", .05), ("Allen–C.\n10%", "allencahn", .1), ("Landau\n10%", "landau", .1), ("toggle\n10%", "toggle", .1)],
         THREE, "r_wide")
    bars(P, [("Lorenz\n10%", "lorenz", .1), ("Lorenz\n20%", "lorenz", .2), ("Van der Pol\n20%", "vanderpol", .2), ("Hopf\n20%", "hopf", .2)],
         [("esindy+random", "original", RAND), ("esindy_w+random", "whitening only", "#7FA8D6"),
          ("esindy_t+random", "test only", "#E9B48A"), ("esindy_wt+random", "whitening + test", OURS)], "r_ablation", fs=15)
    bars(T, [("Lorenz\n5%", "lorenz", .05), ("Lorenz\n10%", "lorenz", .1), ("V. d. Pol\n5%", "vanderpol", .05), ("V. d. Pol\n10%", "vanderpol", .1),
             ("Hopf\n5%", "hopf", .05), ("Hopf\n10%", "hopf", .1), ("Lotka-V.\n5%", "lotka", .05), ("Lotka-V.\n10%", "lotka", .1)],
         [("esindy+variance", "Active E-SINDy", BASE), ("esindy_wt+variance", "ours (our identification + variance)", OURS)], "r_traj", size=(13, 5.6), fs=13)
    bars(D, [("Burgers\n50%", "burgers", .5), ("Burgers\n100%", "burgers", 1.0), ("KS\n50%", "ks", .5), ("KS\n100%", "ks", 1.0)],
         [("esindy+variance", "Active E-SINDy", BASE), ("esindy+dopt", "D-optimal", GRN), ("esindy_t+iopt", "ours", OURS)], "r_pde", size=(9, 5.6), fs=13)

    # two kinds of error, 20% noise
    cfg = [("Lorenz 20%", "lorenz", .2), ("Chen 20%", "chen", .2), ("Van der Pol 20%", "vanderpol", .2), ("Hopf 20%", "hopf", .2)]
    arms = [("esindy+variance", "original"), ("esindy_wt+variance", "our ident.\n+ variance"), ("esindy_wt+iopt", "ours")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 6.2), sharey=True)
    for ax, (name, s, nz) in zip(axes, cfg):
        lg = lambda a: [r for l in logs(P, s, nz, a) for r in l]
        e_ = [100 * np.mean([r["extra"] > 0 for r in lg(a)]) for a, _ in arms]
        m_ = [100 * np.mean([r["missing"] > 0 for r in lg(a)]) for a, _ in arms]
        xx = np.arange(3); ax.bar(xx - 0.2, e_, 0.4, color=BAD, label="extra terms"); ax.bar(xx + 0.2, m_, 0.4, color=INK, label="missing terms")
        for i in range(3):
            ax.text(i - 0.2, e_[i] + 2, f"{e_[i]:.0f}", ha="center", fontsize=15); ax.text(i + 0.2, m_[i] + 2, f"{m_[i]:.0f}", ha="center", fontsize=15)
        ax.set_title(name, fontsize=20); ax.set_xticks(xx); ax.set_xticklabels([l for _, l in arms], fontsize=14); ax.set_ylim(0, 112)
    axes[0].set_ylabel("runs with this error (%)"); axes[0].legend(loc="upper right", fontsize=15); save(fig, "r_errors")

    # domain models, original range
    order = ["sir", "fhn", "selkov", "toggle", "mm", "brusselator", "binding", "arrhenius", "allencahn", "grain", "landau"]
    short = dict(sir="SIR", fhn="FitzHugh–\nNagumo", selkov="Selkov", toggle="toggle\n(Hill)", mm="Michaelis–\nMenten", brusselator="Brusselator",
                 binding="binding", arrhenius="Arrhenius", allencahn="Allen–\nCahn", grain="grain\ngrowth", landau="Landau")
    arms = [("esindy+variance", "Active E-SINDy", BASE), ("esindy+dopt", "D-optimal (How Low)", GRN),
            ("esindy_wt+variance", "our identification + variance", PUR), ("esindy_wt+iopt", "ours", OURS)]
    fig, axes = plt.subplots(2, 1, figsize=(17, 9.5), sharex=True)
    for ax, nz in zip(axes, (0.05, 0.1)):
        x = np.arange(len(order))
        for j, (a, lab, c) in enumerate(arms):
            ax.bar(x + (j - 1.5) * 0.2, [rec(Dm, s, nz, a) for s in order], 0.2, color=c, label=lab)
        ax.set_ylim(0, 108); ax.set_ylabel(f"recovery (%)\nnoise {int(nz * 100)}%")
        for i, s in enumerate(order):
            f = dm.MODELS[s][0]
            if i == 0 or dm.MODELS[order[i - 1]][0] != f:
                ax.axvline(i - 0.5, color=MUTE, lw=1, ls=":")
                if nz == 0.05:
                    ax.text(i - 0.4, 101, f, fontsize=18, color=MUTE)
    axes[1].set_xticks(np.arange(len(order))); axes[1].set_xticklabels([short[s] for s in order], fontsize=15)
    axes[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.22), fontsize=16)
    save(fig, "r_domains")
    print("written", sorted(f for f in os.listdir(OUT) if f.startswith("slide_")))

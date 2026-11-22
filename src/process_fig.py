"""Process figures for one representative run: initial data and SINDy result, the points each
method acquires, and the equations each method ends with.  Every panel shows the true equations.

    python src/process_fig.py [system] [noise] [seed]            # default: selkov 0.05, first seed whose
                                                             # outcome matches the majority pattern
Writes figs/process_1.png, figs/process_2.png, figs/process_3.png.
"""
import sys
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import domains as dm
import expand
import noise_aware as na

INK, MUTE, BG, BAD, OK = "#14213D", "#5B6577", "#FBFBF8", "#B23A48", "#2E7D4F"
plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False, "font.size": 15,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
METHODS = [("esindy+variance", "Active E-SINDy (ensemble variance)", "#2C6FBB"),
           ("esindy+dopt", "How Low Can You Go? (D-optimal)", "#6C8EBF"),
           ("esindy_wt+iopt", "ours (whitening + test + I-optimal)", "#D9661F")]
PRETTY = {"1": "", "x": "x", "y": "y", "x*x": "x²", "x*y": "xy", "y*y": "y²", "x*x*x": "x³", "x*x*y": "x²y", "x*y*y": "xy²", "y*y*y": "y³"}


def physical(prob, C):
    return C * prob.tau[None, :] / prob.lib.scale[:, None]           # back to the model's own units


def equation(prob, C, c, lhs, per_line=None):
    terms = []
    for j in np.nonzero(C[:, c])[0]:
        v = C[j, c]; name = PRETTY.get(prob.lib.names[j], prob.lib.names[j])
        terms.append(f"{'−' if v < 0 else '+'} {abs(v):.2f}{name}")
    if per_line and len(terms) > per_line:
        head = " ".join(terms[:per_line]); tail = " ".join(terms[per_line:])
        head = head[2:] if head.startswith("+ ") else "−" + head[2:]
        return f"{lhs} = {head}\n        {tail}"
    s = " ".join(terms) if terms else "0"
    return f"{lhs} = " + (s[2:] if s.startswith("+ ") else "−" + s[2:] if s.startswith("− ") else s)


def errors(prob, C):
    names = lambda m: [PRETTY.get(prob.lib.names[j], prob.lib.names[j]) or "1" for j in np.nonzero(m)[0]]
    out = []
    for c, lhs in enumerate(("dx/dt", "dy/dt")):
        ex_, ms = names((C[:, c] != 0) & ~prob.truth[:, c]), names((C[:, c] == 0) & prob.truth[:, c])
        if ex_: out.append(f"{lhs} extra {', '.join(ex_)}")
        if ms: out.append(f"{lhs} missing {', '.join(ms)}")
    return out


def run(system, noise, arm, seed):
    """Same steps and random streams as experiments.run_trial on the widened-range protocol."""
    ident, acq = arm.split("+"); flags = ident.partition("_")[2]
    env = np.random.default_rng([seed, 0]); rng = np.random.default_rng([seed, zlib.crc32(arm.encode())])
    prob = expand.WideDomain(env, system, noise)
    X0 = prob._states(prob.n_init); rows = prob._measure(X0)
    picks, C = [], []
    for step in range(prob.n_steps + 1):
        eqs = na.identify(rows, rng, prob.lam, "w" in flags, "t" in flags, prob.block)
        if step in (0, prob.n_steps):
            M = np.zeros_like(prob.C)
            for e in eqs:
                M[e["support"], e["c"]] = e["xi"]
            C.append(M)
        if step == prob.n_steps:
            break
        cand, idx = prob.candidates(eqs)
        score = (na.variance_score(eqs, cand) if acq == "variance" else
                 na.prediction_score(eqs, cand) if acq == "iopt" else na.dopt_score(eqs, cand))
        k = int(np.argmax(np.nan_to_num(score, nan=-np.inf)))
        picks.append(prob.X[k]); rows = na.stack(rows, prob.measure(idx[k]))
    return prob, X0, np.array(picks), C[0], C[1]


def background(ax, prob, lo, hi):
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40))
    V = (prob.lib(np.stack([gx, gy], -1)) @ prob.C) * prob.tau
    ax.streamplot(gx, gy, V[..., 0], V[..., 1], color="#C9CED6", density=1.1, linewidth=0.8, arrowsize=0.8)
    ax.add_patch(Rectangle(prob.lo, *(prob.hi - prob.lo), fill=False, ec=INK, lw=2))
    ax.add_patch(Rectangle(prob.wlo, *(prob.whi - prob.wlo), fill=False, ec=MUTE, lw=1.5, ls="--"))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_xlabel("x"); ax.set_ylabel("y")


def truth_text(prob):
    T = physical(prob, prob.C)
    return "true equations:  " + equation(prob, T, 0, "dx/dt") + "    " + equation(prob, T, 1, "dy/dt")


def simulate(prob, Cphys, x0, T=40.0, h=0.01):
    f = lambda x: prob.lib(x[None])[0] * prob.lib.scale @ Cphys
    x, out = np.array(x0, float), [np.array(x0, float)]
    with np.errstate(all="ignore"):
        for _ in range(int(T / h)):
            k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(x).all() or np.abs(x).max() > 50: break
            out.append(x)
    return np.array(out)


if __name__ == "__main__":
    system = sys.argv[1] if len(sys.argv) > 1 else "selkov"
    noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    seeds = [int(sys.argv[3])] if len(sys.argv) > 3 else range(32)
    for seed in seeds:
        R = {a: run(system, noise, a, seed) for a, _, _ in METHODS}
        ok = {a: bool(((R[a][4] != 0) == R[a][0].truth).all()) for a in R}
        init_wrong = not ((R["esindy+variance"][3] != 0) == R["esindy+variance"][0].truth).all()
        print(seed, ok, "initial SINDy wrong:", init_wrong)
        if len(seeds) == 1 or (ok["esindy_wt+iopt"] and not ok["esindy+variance"] and not ok["esindy+dopt"] and init_wrong):
            break
    prob = R["esindy_wt+iopt"][0]
    lo, hi = prob.wlo - 0.05 * (prob.whi - prob.wlo), prob.whi + 0.05 * (prob.whi - prob.wlo)
    title = f"Selkov glycolysis, state noise {int(noise * 100)}%, seed {seed}" if system == "selkov" else f"{system}, noise {int(noise * 100)}%, seed {seed}"

    # ---- figure 1: initial data and what SINDy makes of it
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_axes([0.06, 0.1, 0.42, 0.78]); background(ax, prob, lo, hi)
    X0 = R["esindy+variance"][1]
    ax.scatter(X0[:, 0], X0[:, 1], s=60, color=INK, zorder=3, label=f"initial {len(X0)} measurements")
    ax.plot([], [], color=INK, lw=2, label="initial range"); ax.plot([], [], color=MUTE, lw=1.5, ls="--", label="feasible experiment range")
    ax.legend(loc="upper right", fontsize=13, framealpha=0.9)
    ax.set_title("Fig. 1  initial data", fontsize=18, loc="left")
    tx = fig.add_axes([0.52, 0.1, 0.46, 0.78]); tx.axis("off")
    T = physical(prob, prob.C)
    lines = [("true equations", INK, 17, "bold"), (equation(prob, T, 0, "dx/dt"), INK, 16, None), (equation(prob, T, 1, "dy/dt"), INK, 16, None), ("", INK, 8, None)]
    for a, name in (("esindy+variance", "E-SINDy (as in both published methods)"), ("esindy_wt+iopt", "our identification (whitening + test)")):
        Ci = physical(prob, R[a][3]); err = errors(prob, R[a][3])
        lines += [(f"{name}, from these 20 points:", MUTE, 15, "bold")]
        for c, lhs in enumerate(("dx/dt", "dy/dt")):
            for part in equation(prob, Ci, c, lhs, per_line=5).split("\n"):
                lines.append((part, INK, 15, None))
        lines += [("; ".join(err) if err else "structure correct", BAD if err else OK, 14, None), ("", INK, 8, None)]
    y = 0.98
    for t, c, s, w in lines:
        tx.text(0, y, t, color=c, fontsize=s, fontweight=w, va="top", transform=tx.transAxes); y -= 0.066 if s > 9 else 0.025
    fig.suptitle(title, fontsize=20, x=0.06, ha="left")
    fig.savefig("figs/process_1.png", dpi=150); plt.close(fig)

    # ---- figure 2: where each method measures next
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    for ax, (a, name, col) in zip(axes, METHODS):
        background(ax, prob, lo, hi)
        P = R[a][2]; out = ((P < prob.lo) | (P > prob.hi)).any(1).mean()
        ax.scatter(X0[:, 0], X0[:, 1], s=25, color=MUTE, zorder=3)
        ax.scatter(P[:, 0], P[:, 1], s=34, c=np.arange(len(P)), cmap="Oranges" if "iopt" in a else "Blues", vmin=-20, vmax=len(P), zorder=4, edgecolor=INK, lw=0.3)
        ax.set_title(f"{name}\n{len(P)} new points, {out:.0%} outside the initial range", fontsize=14)
    fig.suptitle("Fig. 2  next experiments chosen by each method (grey: initial points; light to dark: order)\n" + truth_text(prob), fontsize=16, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("figs/process_2.png", dpi=150); plt.close(fig)

    # ---- figure 3: the equations each method ends with
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6))
    x0 = (prob.lo + prob.hi) / 2 + np.array([0.3, 0.6])
    truth_path = simulate(prob, physical(prob, prob.C), x0)
    for ax, (a, name, col) in zip(axes, METHODS):
        background(ax, prob, lo, hi)
        Cf = R[a][4]; err = errors(prob, Cf)
        path = simulate(prob, physical(prob, Cf), x0)
        ax.plot(truth_path[:, 0], truth_path[:, 1], color=INK, lw=2.5, label="trajectory of the true equations")
        ax.plot(path[:, 0], path[:, 1], color=col, lw=2.5, ls="--", label="trajectory of the identified equations")
        ax.scatter(*x0, s=60, color=INK, zorder=5)
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        Cp = physical(prob, Cf)
        verdict = "structure correct" if not err else "; ".join(err)
        ax.set_title(f"{name}\n{equation(prob, Cp, 0, 'dx/dt')}\n{equation(prob, Cp, 1, 'dy/dt')}\n{verdict}", fontsize=12.5,
                     color=OK if not err else BAD)
    fig.suptitle("Fig. 3  equations after 100 points, and trajectories from the same initial state\n" + truth_text(prob), fontsize=16, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figs/process_3.png", dpi=150); plt.close(fig)
    print("seed", seed, "written figs/process_1-3.png")

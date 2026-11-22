"""How the "noise is not the same everywhere" figure is built, step by step (Lorenz, dy/dt = 28x - y - xz).
    python src/noise_explain.py   ->  figs/noise_explain.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiments as ex

INK, MUTE, BG, A_C, B_C = "#14213D", "#5B6577", "#FBFBF8", "#2C6FBB", "#D9661F"
plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False, "font.size": 15,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
prob = ex.Points(np.random.default_rng(0), "lorenz", 0.05)
sig = prob.sigma
f = lambda X: 28 * X[..., 0] - X[..., 1] - X[..., 0] * X[..., 2]          # dy/dt of Lorenz
J = lambda X: np.sqrt((28 - X[..., 2]) ** 2 + 1 + X[..., 0] ** 2)        # |df/dx|
A, B = np.array([1.0, 1.0, 27.0]), np.array([20.0, 1.0, 2.0])
rng = np.random.default_rng(1)
fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))

# (1) the measurement noise on the state is the same everywhere
for P, c, name in ((A, A_C, "A"), (B, B_C, "B")):
    N = P + sig * rng.standard_normal((300, 3))
    ax[0].scatter(N[:, 0], N[:, 2], s=6, color=c, alpha=0.5)
    ax[0].scatter(P[0], P[2], s=80, color=c, edgecolor=INK, zorder=3)
    ax[0].annotate(f"state {name} = ({P[0]:.0f}, {P[1]:.0f}, {P[2]:.0f})", P[[0, 2]], xytext=(10, 12), textcoords="offset points", fontsize=14, color=c)
ax[0].set_xlim(-25, 25); ax[0].set_ylim(0, 50); ax[0].set_xlabel("x"); ax[0].set_ylabel("z")
ax[0].set_title(f"(1) 300 measurements at each of two states\nthe state noise is the same (sd {sig:.2f})", fontsize=15)

# (2) but the error it causes in dy/dt is very different
for P, c, name in ((A, A_C, "A"), (B, B_C, "B")):
    N = P + sig * rng.standard_normal((5000, 3))
    r = f(P) - f(N)
    ax[1].hist(r, bins=80, range=(-120, 120), density=True, color=c, alpha=0.6,
               label=f"state {name}: error sd {r.std():.1f}\nformula σ·‖∂f/∂x‖ = {sig * J(P):.1f}")
ax[1].set_xlabel("error in dy/dt = f(true state) − f(measured state)"); ax[1].set_yticks([]); ax[1].legend(fontsize=13, loc="upper left")
ax[1].set_title(f"(2) the same state noise, as error in dy/dt\nB is {J(B) / J(A):.0f}x A: the equation is more sensitive at B", fontsize=15)

# (3) do it for 20,000 random states and group them by sensitivity
X = prob._states(20000); Xn = X + sig * rng.standard_normal(X.shape)
r = f(X) - f(Xn); g = J(Xn)
q = np.quantile(g, np.linspace(0, 1, 21)); mx, sd = [], []
for a, b in zip(q[:-1], q[1:]):
    m = (g >= a) & (g <= b); mx.append(g[m].mean()); sd.append(r[m].std())
ax[2].scatter(g[:3000], np.abs(r[:3000]), s=5, color="#A9B0BA", alpha=0.5, label="|error| of each point")
ax[2].plot(mx, sd, "o-", color="#B23A48", lw=3, ms=7, label="error sd in 20 sensitivity bins")
ax[2].plot([0, max(mx)], [0, sig * max(mx)], "--", color=INK, lw=2.5, label="formula: σ × sensitivity")
ax[2].axhline(r.std(), color=A_C, ls=":", lw=2.5, label="E-SINDy assumption: constant")
for P, c in ((A, A_C), (B, B_C)):
    ax[2].axvline(J(P), color=c, lw=1.5, alpha=0.7)
ax[2].set_xlabel("sensitivity of the equation to the state ‖∂f/∂x‖"); ax[2].set_ylabel("error"); ax[2].set_ylim(0, np.quantile(np.abs(r), 0.995))
ax[2].legend(fontsize=12, loc="upper left"); ax[2].set_title("(3) the same for 20,000 random states\nthe error grows with sensitivity; it is not constant", fontsize=15)
fig.suptitle("Regression noise is not the same everywhere (Lorenz: dy/dt = 28x − y − xz, sensitivity ‖∂f/∂x‖ = √((28 − z)² + 1 + x²))", fontsize=17, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("figs/noise_explain.png", dpi=150)
print("sigma", round(sig, 3), "J(A)", round(J(A), 2), "J(B)", round(J(B), 2))

"""Render the benchmark summary and one fixed held-out experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from benchmark import METHODS, ROOT, SETTINGS, SPACING, load_experiments, rollout, rms


COLORS = ("#675c9f", "#82919c", "#127b82", "#4173aa", "#cf7432")
NAMES = {"al": "Aluminum", "cu": "Copper", "ss": "Stainless steel"}


def summary(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True, layout="constrained")
    for axis, material in zip(axes, SPACING):
        axis.set_title(NAMES[material], loc="left", fontweight="bold")
        for i, (method, color) in enumerate(zip(METHODS[:5], COLORS)):
            scores = [row["rmse_c"] for row in results if row["material"] == material
                      and row["method"] == method]
            shifts = np.arange(len(SETTINGS)) + .16 * (i - 2)
            axis.bar(shifts, [score or 0 for score in scores], width=.14, color=color,
                     label=method, zorder=3)
            invalid = [shift for shift, score in zip(shifts, scores) if score is None]
            if invalid:
                axis.scatter(invalid, [.06] * len(invalid), color=color, marker="x", s=23, zorder=4)
        axis.set_xticks(range(len(SETTINGS)), SETTINGS)
        axis.set_xlabel("Held-out heater setting")
        axis.grid(axis="y", color="#e0e7eb", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Interior temperature RMSE (°C)")
    fig.suptitle("Can a discovered PDE transfer to a new heater setting?", fontsize=15,
                 fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def animate(results, experiments, output):
    material, setting = "al", 75
    measured = experiments[f"{setting}{material}1.csv"]
    boundary = measured[:, (0, -1)]
    predictions = []
    for method in ("Weak rollout", "Fixed heat + loss"):
        model = next(row for row in results if row["material"] == material
                     and row["test_setting"] == setting and row["method"] == method)
        interior = rollout(model["coefficients"], SPACING[material], .5, measured[0, 1:-1], boundary)
        predictions.append(np.column_stack((boundary[:, 0], interior, boundary[:, 1])))
    x = SPACING[material] * np.arange(measured.shape[1])
    fig, ax = plt.subplots(figsize=(7.4, 3.75), layout="constrained")
    ax.set_title("A whole heater setting held out  |  Aluminum", loc="left", fontweight="bold")
    ax.set(xlim=(-2, x[-1] + 2), ylim=(float(measured.min()) - 2, float(measured.max()) + 3),
           xlabel="Distance along rod (mm)", ylabel="Recorded temperature rise (°C)")
    ax.grid(color="#e8edef")
    ax.spines[["top", "right"]].set_visible(False)
    predicted, = ax.plot(x, measured[0], color=COLORS[2], lw=2, label="Discovered PDE")
    fixed, = ax.plot(x, measured[0], "--", color=COLORS[4], lw=1.5, label="Fixed heat + loss")
    actual, = ax.plot(x[1:-1], measured[0, 1:-1], "o", color="#152d43", ms=5,
                      label="Measured interior")
    ends, = ax.plot(x[[0, -1]], boundary[0], "D", color="#697786", ms=5, label="Supplied boundaries")
    clock = ax.text(.98, .95, "", ha="right", va="top", transform=ax.transAxes, color="#2e4358")
    ax.legend(loc="upper right", bbox_to_anchor=(1, .87), frameon=False, fontsize=9)

    def update(frame):
        predicted.set_ydata(predictions[0][frame])
        fixed.set_ydata(predictions[1][frame])
        actual.set_ydata(measured[frame, 1:-1])
        ends.set_ydata(boundary[frame])
        clock.set_text(f"t = {frame * .5:3.0f} s")

    frames = np.linspace(0, len(measured) - 1, 55, dtype=int)
    FuncAnimation(fig, update, frames=frames).save(output, writer=PillowWriter(fps=11), dpi=95)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    output = ROOT / "results"
    results = json.loads((output / "benchmark.json").read_text())["real"]
    summary(results, output / "transfer.png")
    animate(results, load_experiments(args.data_dir), output / "heldout.gif")


if __name__ == "__main__":
    main()

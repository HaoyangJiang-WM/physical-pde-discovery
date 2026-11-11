"""Compare effective axial heat-transport models on held-out packed-bed runs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import nnls

RESULTS = Path(__file__).parent / "results" / "packed_bed_pde"
TIME = "Time since Start [min]"
HEIGHT = np.array([25, 48, 75, 98]) / 1000
PROBES = (
    ("TR401-045a", "TR401-090b", "TR401-135c", "TR401-157d", "TR401-180e"),
    ("TR402-225a", "TR402-270b", "TR402-315c", "TR402-337d", "TR402-000e"),
    ("TR403-153a", "TR403-198b", "TR403-243c", "TR403-265d", "TR403-288e"),
    ("TR404-333a", "TR404-018b", "TR404-063c", "TR404-085d", "TR404-108e"),
)
RADIUS_MM = np.array([[0, 30, 58, 80, 95], [5, 30, 60, 80, 95],
                      [0, 26, 60, 80, 95], [0, 30, 60, 80, 95]])
REPEATS = {"low": (1, 2, 3), "high": (6, 7, 8)}
REGIME = {1: "low", 2: "low", 3: "low", 4: "low", 5: "low",
          6: "high", 7: "high", 8: "high", 9: "high"}
MODELS = {
    "local cooling": (2,),
    "diffusion + cooling": (0, 2),
    "advection + cooling": (1, 2),
    "diffusion + advection + cooling": (0, 1, 2),
}


def read_run(path):
    with path.open(encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split(";")
        names = [TIME, "Heater_AO", "FRC101", *sum(PROBES, ())]
        rows = ([line.strip().replace(",", ".") for line in handle if line.strip()])
        data = np.loadtxt(rows, delimiter=";",
                          usecols=[header.index(name) for name in names])
    if not np.all(np.isfinite(data)) or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError(f"Invalid readings or timestamps in {path}")
    return data


def axial_derivatives(field, scheme):
    """Derivatives of local three-point quadratic fits at interior heights."""
    gradient, curvature = [], []
    for center in (1, 2):
        dx = HEIGHT[center - 1:center + 2] - HEIGHT[center]
        basis = np.column_stack([np.ones(3), dx, dx**2 / 2])
        coefficients = np.linalg.solve(basis, field[:, center - 1:center + 2].T)
        gradient.append(coefficients[1])
        curvature.append(coefficients[2])
    if scheme == "upwind":
        return np.diff(field[:, :3], axis=1) / np.diff(HEIGHT[:3]), np.column_stack(curvature)
    return np.column_stack(gradient), np.column_stack(curvature)


def design(data, aggregation, scheme="centered"):
    time, heater, flow = data[:, :3].T
    shutoff = np.flatnonzero(heater > 12)[-1] + 1
    elapsed = (time - time[shutoff]) * 60
    grid = np.arange(5 * 60, 55 * 60 + 1, 10)
    probes = data[:, 3:].reshape(-1, 4, 5)
    # The instrument range is 273–823 K. Median bins also reduce spikes.
    probes = np.where((probes >= -0.15) & (probes <= 549.85), probes, np.nan)
    left = np.searchsorted(elapsed, grid - 5)
    right = np.searchsorted(elapsed, grid + 5)
    sampled = np.array([np.nanmedian(probes[lo:hi], axis=0)
                        for lo, hi in zip(left, right)])
    if not np.all(np.isfinite(sampled)):
        raise ValueError("Missing temperature support on the cooling grid")
    if aggregation == "area":
        edges = np.column_stack([np.zeros(4),
                                 (RADIUS_MM[:, 1:] + RADIUS_MM[:, :-1]) / 2,
                                 np.full(4, 100)])
        weights = np.diff(edges**2, axis=1) / 100**2
        field = np.sum(sampled * weights, axis=2)
    else:
        field = sampled[:, :, 0]
    idle = flow < 1
    if not np.any(idle[:60]):
        raise ValueError("No pre-run idle readings to estimate ambient temperature")
    initial = np.nanmedian(probes[:60][idle[:60]])
    gradient, curvature = axial_derivatives(field, scheme)
    terms = np.stack([curvature, -gradient, initial - field[:, 1:3]], axis=-1)
    features, increments = [], []
    for start in range(0, len(grid) - 1, 12):
        stop = start + 12
        features.append(trapezoid(terms[start:stop + 1], x=grid[start:stop + 1], axis=0))
        increments.append(field[stop, 1:3] - field[start, 1:3])
    return {
        "features": np.vstack(features), "target": np.concatenate(increments),
        "field": field, "grid": grid, "scheme": scheme, "initial_ambient": float(initial),
        "shutoff_min": float(time[shutoff]),
        "flow_Nm3_h": float(np.median(flow[(elapsed >= grid[0]) & (elapsed <= grid[-1])])),
    }


def rollout(run, columns, coefficients, boundary_field=None):
    """Predict both interior histories using only initial state and measured boundaries."""
    grid, field = run["grid"], run["field"]
    boundary = field if boundary_field is None else boundary_field
    gradient, curvature = axial_derivatives(np.eye(4), run["scheme"])
    operator = np.column_stack([curvature.ravel(), -gradient.ravel()]).reshape(4, 2, 2)
    expanded = np.zeros(3)
    expanded[list(columns)] = coefficients
    transport = operator @ expanded[:2]
    cooling = expanded[2]

    def rhs(time, interior):
        values = np.array([np.interp(time, grid, boundary[:, 0]), *interior,
                           np.interp(time, grid, boundary[:, -1])])
        return values @ transport + cooling * (run["initial_ambient"] - interior)

    solution = solve_ivp(rhs, (grid[0], grid[-1]), field[0, 1:3], t_eval=grid,
                         method="LSODA", rtol=1e-6, atol=1e-6, max_step=10)
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(np.sqrt(np.mean((solution.y.T[1:] - field[1:, 1:3])**2)))


def fit_coefficients(training, columns):
    x = np.vstack([item["features"] for item in training])[:, columns]
    y = np.concatenate([item["target"] for item in training])
    scale = np.linalg.norm(x, axis=0)
    return nnls(x / scale, y)[0] / scale


def evaluate(runs, models):
    scores = {}
    for run_id, run in runs.items():
        same_regime = REPEATS[REGIME[run_id]]
        training = [runs[i] for i in same_regime if i != run_id]
        scores[str(run_id)] = {
            "training_runs": [i for i in same_regime if i != run_id],
            "shutoff_min": run["shutoff_min"], "flow_Nm3_h": run["flow_Nm3_h"],
            "window_layer_observations": len(run["target"]),
            "models": {},
        }
        for model, columns in models.items():
            coefficients = fit_coefficients(training, columns)
            predicted = run["features"][:, columns] @ coefficients
            scores[str(run_id)]["models"][model] = {
                "coefficients": coefficients.tolist(),
                "rmse_K": float(np.sqrt(np.mean((predicted - run["target"])**2))),
                "boundary_conditioned_rollout_rmse_K": rollout(run, columns, coefficients),
            }
    return scores


def plot(runs, scores, upwind_scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for level, color in zip(range(4), ("#253494", "#2c7fb8", "#41b6c4", "#207c60")):
        axes[0].plot(runs[6]["grid"] / 60, runs[6]["field"][:, level],
                     label=f"{HEIGHT[level] * 1000:.0f} mm", color=color, lw=2)
    axes[0].set(xlabel="Minutes after heater shutoff", ylabel="Temperature (°C)",
                title="Measured cooling at four heights (run 6)")
    axes[0].legend(frameon=False, ncol=2)
    order = [1, 2, 3, 6, 7, 8, 4, 5, 9]
    for index, (name, color) in enumerate(zip(MODELS, ("#707a86", "#4679ad", "#2a9d8f", "#f09a53"))):
        axes[1].bar(np.arange(len(order)) + (index - 1.5) * 0.19,
                    [scores[str(i)]["models"][name]["boundary_conditioned_rollout_rmse_K"] for i in order],
                    width=0.18, label=name, color=color)
    axes[1].scatter(np.arange(len(order)) + 0.26,
                    [upwind_scores[str(i)]["models"]["advection + cooling"]
                     ["boundary_conditioned_rollout_rmse_K"] for i in order],
                    color="#171d2b", s=15, marker="D", label="upwind advection + cooling", zorder=3)
    axes[1].axvline(5.5, color="#aaa", lw=1)
    axes[1].set(xticks=np.arange(len(order)), xticklabels=order,
                xlabel="Held-out run (repeats / changed setup)",
                ylabel="Interior rollout RMSE (K)", title="Models tested on whole unseen runs")
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root of the unpacked Mendeley dataset")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    manifest = json.loads((RESULTS / "source_manifest.json").read_text())
    recordings = {}
    for relative, expected in manifest["files"].items():
        if relative.endswith("/MeasurementData.CSV"):
            path = args.source / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Checksum mismatch for {path}")
            recordings[int(relative.split("_", 1)[0])] = read_run(path)
    area = {i: design(raw, "area") for i, raw in recordings.items()}
    center = {i: design(raw, "center") for i, raw in recordings.items()}
    upwind = {i: design(raw, "area", "upwind") for i, raw in recordings.items()}
    scores = {"source_doi": manifest["doi"], "area": evaluate(area, MODELS),
              "near_axis": evaluate(center, MODELS), "area_upwind": evaluate(upwind, MODELS)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    plot(area, scores["area"], scores["area_upwind"], args.out / "heldout.png")


if __name__ == "__main__":
    main()

"""Discover transport PDEs and validate them on entire unseen experiments."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parent
TERMS = ("u_xx", "u_x", "u", "1")
SPACING = {"al": 10., "cu": 15., "ss": 7.}
SETTINGS = (50, 75, 100)
SUPPORTS = tuple(itertools.chain.from_iterable(
    itertools.combinations(range(len(TERMS)), n) for n in range(1, len(TERMS) + 1)
))
METHODS = ("PDE-FIND", "Weak residual", "Weak rollout", "Weak rollout + solid prior",
           "Fixed heat + loss", "Linear boundary")


def load_experiments(data_dir):
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    experiments = {}
    for name, metadata in manifest["files"].items():
        path = data_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Unexpected file digest for {name}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            samples = np.array([
                [float(row[key]) for key in ("Time", *(f"TC{n}" for n in range(1, 7)))]
                for row in csv.DictReader(handle)
            ])
        times = np.arange(0., 240.5, .5)
        if np.any(np.diff(samples[:, 0]) <= 0) or samples[-1, 0] < times[-1]:
            raise ValueError(f"Incomplete or nonmonotone timestamps for {name}")
        experiments[name] = np.stack([
            np.interp(times, samples[:, 0], samples[:, channel]) for channel in range(1, 7)
        ], axis=1)
    return experiments


def regression(field, dx, dt, weak):
    if weak:
        # Compact time and spatial test functions. Apply the transposed
        # difference matrices to the spatial weights before multiplying by u.
        n = field.shape[1]
        weights = np.zeros((n - 2, n))
        for center in range(1, n - 1):
            first = max(1, center - 1)
            last = min(n - 2, center + 1)
            weights[center - 1, first:last + 1] = .25
            weights[center - 1, center] = .5
        weights /= weights.sum(axis=1, keepdims=True)
        core = weights[:, 1:-1]
        identity = np.eye(n)
        d1 = (identity[2:] - identity[:-2]) / (2 * dx)
        d2 = (identity[2:] - 2 * identity[1:-1] + identity[:-2]) / dx**2
        features = np.stack((field @ (core @ d2).T, field @ (core @ d1).T,
                             field @ weights.T, np.ones((len(field), len(core)))), axis=-1)
        grid = np.linspace(-1, 1, 25)
        phi = (1 - grid**2)**3
        dphi = -6 * grid * (1 - grid**2)**2 / (12 * dt)
        target = field @ weights.T
        starts = range(0, len(field) - len(grid) + 1, 6)
        x = [np.einsum("t,txk->xk", phi, features[i:i + len(grid)]) for i in starts]
        y = [-dphi @ target[i:i + len(grid)] for i in starts]
        return np.concatenate(x), np.concatenate(y)

    smooth = savgol_filter(field, 11, 3, axis=0)
    center = smooth[5:-5, 1:-1]
    features = np.stack(((smooth[5:-5, 2:] - 2 * center + smooth[5:-5, :-2]) / dx**2,
                         (smooth[5:-5, 2:] - smooth[5:-5, :-2]) / (2 * dx),
                         center, np.ones_like(center)), axis=-1)
    target = savgol_filter(field, 11, 3, deriv=1, delta=dt, axis=0)[5:-5, 1:-1]
    return features.reshape(-1, len(TERMS)), target.ravel()


def fit(x, y, support):
    coefficients = np.zeros(len(TERMS))
    columns = x[:, support]
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-12)
    coefficients[list(support)] = np.linalg.lstsq(columns / scale, y, rcond=None)[0] / scale
    return coefficients


def stridge_path(x, y):
    """Normalized sequential thresholded ridge, followed by unshrunk refits."""
    scale = np.maximum(np.linalg.norm(x, axis=0), 1e-12)
    normalized = x / scale
    reference = np.linalg.norm(y)
    supports = set()
    for threshold in np.geomspace(.001, .5, 24):
        active = np.arange(len(TERMS))
        for _ in TERMS:
            columns = normalized[:, active]
            coef = np.linalg.solve(columns.T @ columns + 1e-5 * np.eye(len(active)), columns.T @ y)
            survivors = active[np.abs(coef) >= threshold * reference]
            if len(survivors) == len(active) or not len(survivors):
                break
            active = survivors
        supports.add(tuple(int(i) for i in active))
    return sorted(supports)


def operators(coefficients, dx, interior_count):
    diffusion, drift, reaction, source = coefficients
    left = diffusion / dx**2 - drift / (2 * dx)
    right = diffusion / dx**2 + drift / (2 * dx)
    center = reaction - 2 * diffusion / dx**2
    matrix = (np.eye(interior_count) * center + np.eye(interior_count, k=-1) * left
              + np.eye(interior_count, k=1) * right)
    forcing = np.zeros((interior_count, 3))
    forcing[0, 0] = left
    forcing[-1, 1] = right
    forcing[:, 2] = source
    return matrix, forcing


def consistent(coefficients, dx, n):
    matrix, _ = operators(coefficients, dx, n)
    return bool(coefficients[0] >= 0 and np.max(np.linalg.eigvals(matrix).real) <= 1e-10)


def rollout(coefficients, dx, dt, initial, boundary):
    """Advance the centered spatial discretization with linear boundary interpolation."""
    n = len(initial)
    matrix, forcing = operators(coefficients, dx, n)
    lifted = np.zeros((n + 6, n + 6))
    lifted[:n, :n] = matrix
    lifted[:n, n:n + 3] = forcing
    lifted[n:n + 3, n + 3:] = np.eye(3)
    transition = expm(lifted * dt)[:n]
    boundary = np.column_stack((boundary, np.ones(len(boundary))))
    forcing_steps = (boundary[:-1] @ transition[:, n:n + 3].T
                     + np.diff(boundary, axis=0) @ transition[:, n + 3:].T / dt)
    result = np.empty((len(boundary), n))
    result[0] = initial
    for index, step in enumerate(forcing_steps, 1):
        result[index] = transition[:, :n] @ result[index - 1] + step
    return result


def rms(errors):
    return float(np.sqrt(np.mean(np.asarray(errors)**2)))


def rollout_error(coef, dx, dt, field):
    predicted = rollout(coef, dx, dt, field[0, 1:-1], field[:, (0, -1)])
    return rms(predicted[1:] - field[1:, 1:-1])


def stack_regressions(rows):
    return np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows])


def select(train, validation, dx, dt, method):
    weak = method != "PDE-FIND"
    x, y = stack_regressions([regression(field, dx, dt, weak) for field in train])
    use_rollout = method.startswith("Weak rollout")
    if not use_rollout:
        vx, vy = stack_regressions([regression(field, dx, dt, weak) for field in validation])
    supports = stridge_path(x, y) if method == "PDE-FIND" else SUPPORTS
    if method == "Weak rollout + solid prior":
        supports = [support for support in SUPPORTS if 0 in support and 1 not in support]
    candidates = []
    for support in supports:
        coef = fit(x, y, support)
        if method == "Weak rollout + solid prior" and (coef[0] <= 0 or coef[2] > 0):
            continue
        if use_rollout:
            if not consistent(coef, dx, train[0].shape[1] - 2):
                continue
            score = rms([rollout_error(coef, dx, dt, field) for field in validation])
        else:
            score = rms(vx @ coef - vy)
        candidates.append((score, support, coef))
    if not candidates:
        # Include stationary dynamics as a fallback if every candidate is unstable.
        coef = np.zeros(len(TERMS))
        return (), coef
    best = min(score for score, _, _ in candidates)
    _, support, coef = min((item for item in candidates if item[0] <= 1.01 * best),
                           key=lambda item: (len(item[1]), item[0]))
    return support, coef


def fixed_heat_loss(train, dx, dt):
    """Fit the published model family directly to whole training trajectories."""
    def residual(parameters):
        coef = (parameters[0], 0., -parameters[1], 0.)
        return np.concatenate([
            (rollout(coef, dx, dt, field[0, 1:-1], field[:, (0, -1)])[1:]
             - field[1:, 1:-1]).ravel() for field in train
        ])

    result = least_squares(residual, (30., .02), bounds=(0., np.inf), x_scale=(50., .03))
    if not result.success:
        raise RuntimeError(result.message)
    return (0, 2), np.array([result.x[0], 0., -result.x[1], 0.])


def evaluate_real(experiments):
    results = []
    for material, dx in SPACING.items():
        for held_out in SETTINGS:
            available = [setting for setting in SETTINGS if setting != held_out]
            train = [experiments[f"{s}{material}{r}.csv"] for s in available for r in (1, 2)]
            validation = [experiments[f"{s}{material}3.csv"] for s in available]
            tests = [experiments[f"{held_out}{material}{r}.csv"] for r in (1, 2, 3)]
            for method in METHODS:
                if method == "Linear boundary":
                    errors = [rms(np.linspace(field[1:, 0], field[1:, -1], 6, axis=1)[:, 1:-1]
                                  - field[1:, 1:-1]) for field in tests]
                    model = {"support": None, "coefficients": None, "consistent": None}
                else:
                    if method == "Fixed heat + loss":
                        support, coef = fixed_heat_loss(train + validation, dx, .5)
                    else:
                        support, coef = select(train, validation, dx, .5, method)
                    valid = consistent(coef, dx, 4)
                    errors = [rollout_error(coef, dx, .5, field) for field in tests] if valid else None
                    model = {"support": [TERMS[i] for i in support], "coefficients": coef.tolist(),
                             "consistent": valid}
                results.append({"material": material, "test_setting": held_out, "method": method,
                                **model, "repeat_rmse_c": errors,
                                "rmse_c": rms(errors) if errors is not None else None})
    return results


def synthetic_field(coef, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0., 120.5, .5)
    x = np.linspace(0., 50., 21)
    left = 5 + rng.uniform(5, 20) * (1 - np.exp(-t / rng.uniform(8, 25)))
    right = 3 + rng.uniform(1, 9) * np.sin(t / rng.uniform(15, 30))
    profile = np.linspace(left[0], right[0], len(x)) + rng.uniform(2, 12) * np.sin(np.pi * x / 50)
    interior = rollout(coef, 2.5, .5, profile[1:-1], np.column_stack((left, right)))
    return np.column_stack((left, interior, right))[:, ::4]


def evaluate_synthetic():
    cases = {"diffusion": [55., 0., 0., 0.], "diffusion + loss": [55., 0., -.06, 0.],
             "drift + diffusion": [55., 2., 0., 0.]}
    results = []
    for name, coefficients in cases.items():
        clean = [synthetic_field(coefficients, 500 + i) for i in range(9)]
        truth = tuple(np.flatnonzero(coefficients))
        for sigma in (0., .25):
            for seed in range(1 if sigma == 0 else 5):
                rng = np.random.default_rng(seed)
                measured = [field + rng.normal(0, sigma, field.shape) for field in clean[:6]]
                for method in METHODS[:3]:
                    support, coef = select(measured[:4], measured[4:], 10., .5, method)
                    valid = consistent(coef, 10., 4)
                    errors = [rollout_error(coef, 10., .5, field) for field in clean[6:]] if valid else None
                    results.append({"case": name, "noise_c": sigma, "seed": seed, "method": method,
                                    "truth": [TERMS[i] for i in truth], "support": [TERMS[i] for i in support],
                                    "exact_support": support == truth, "consistent": valid,
                                    "rmse_c": rms(errors) if errors is not None else None})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark.json")
    args = parser.parse_args()
    experiments = load_experiments(args.data_dir)
    results = {"real": evaluate_real(experiments), "synthetic": evaluate_synthetic()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method in METHODS:
        real = [row["rmse_c"] for row in results["real"] if row["method"] == method]
        synthetic = [row["exact_support"] for row in results["synthetic"] if row["method"] == method]
        valid = [score for score in real if score is not None]
        print(f"{method:20} real {rms(valid):.3f} C ({len(valid)}/{len(real)} stable), "
              f"synthetic exact support {sum(synthetic)}/{len(synthetic)}")


if __name__ == "__main__":
    main()

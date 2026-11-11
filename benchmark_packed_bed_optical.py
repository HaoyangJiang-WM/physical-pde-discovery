"""Test transfer of optical boundary calibration to independent cooling runs."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import lsq_linear

from benchmark_packed_bed_pde import REGIME, REPEATS, RESULTS, design, fit_coefficients, read_run, rollout

CAMERA = re.compile(r"Intervall(\d+)s_Date(\d{8})_Time(\d{6})\.irb(?:\|(\d+))?$")
WINDOWS = [(start, start + 2) for start in range(7)]
PLANS = [(a, b) for a, b in combinations(WINDOWS, 2) if not set(a) & set(b)]
FIXED = ((0, 2), (5, 7))
SOURCE = RESULTS / "source_manifest.json"
TERMS = (0, 1, 2)


def clock_metadata(workbook):
    from openpyxl import load_workbook

    document = load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = list(document["OverviewIRData"].iter_rows(
            min_row=4, max_row=6, min_col=3, max_col=11, values_only=True))
    finally:
        document.close()
    metadata = {}
    for index, (date, camera, labview) in enumerate(zip(*rows), 1):
        # Run 9 omits the year in the spreadsheet. Its uniquely matching
        # camera name independently supplies November 21, 2024.
        if index == 9 and isinstance(date, str) and date.strip() == "21. Nov.":
            date = datetime(2024, 11, 21)
        started = datetime.combine(date, camera)
        offset = (started - datetime.combine(date, labview)).total_seconds()
        metadata[index] = (started.strftime("%Y%m%d%H%M%S"), offset)
    return metadata


def camera_series(path, stamp, offset):
    frames, values, skipped, intervals = [], [], Counter(), set()
    with path.open(encoding="latin1") as handle:
        rows = csv.reader(handle, delimiter="\t")
        header = next(row for row in rows if row[0] == "Name")
        average = header.index("Packed Bed Average")
        for row in rows:
            if not row or not row[0]:
                continue
            match = CAMERA.search(row[0])
            if match is None:
                raise ValueError(f"Unrecognized camera frame name in {path}: {row[0]}")
            cadence, date, time, frame = match.groups()
            if date + time != stamp or frame is None:
                skipped[date + time] += 1
                continue
            frames.append(int(frame))
            values.append(float(row[average].replace(",", ".")))
            intervals.add(int(cadence))
    if len(intervals) != 1 or not np.all(np.diff(frames) > 0) or frames[0] < 1:
        raise ValueError(f"Ambiguous camera frames in {path}")
    interval = intervals.pop()
    time = offset + (np.array(frames) - 1) * interval
    return time, np.array(values), {"accepted": len(frames), "excluded": dict(skipped),
                                    "nominal_interval_s": interval,
                                    "observed_spacing_s": sorted(set(np.diff(time).astype(int).tolist()))}


def basis(temperature):
    """Integrated linear slope with derivative endpoints at 0 and 500 C."""
    scaled = np.asarray(temperature) / 500
    return np.stack([500 * (scaled - scaled**2 / 2), 250 * scaled**2], axis=-1)


def fit_plan(optical, reference, plan):
    endpoints = np.array(plan)
    features = basis(optical[endpoints])
    x = features[:, 1] - features[:, 0]
    y = reference[endpoints[:, 1]] - reference[endpoints[:, 0]]
    slopes = lsq_linear(x, y, bounds=(.05, 3), tol=1e-10).x
    offset = np.mean(reference[endpoints] - features @ slopes)
    return slopes, float(offset)


def evaluate(calibration, target, optical, coefficients):
    sample = (calibration["grid"] - 600) % 300 == 0
    x = calibration["optical"][sample]
    y = calibration["field"][sample, -1]
    chosen = max(PLANS, key=lambda pair: abs(np.linalg.det(
        basis(x[np.array(pair)[:, 1]]) - basis(x[np.array(pair)[:, 0]]))))

    def score(values):
        field = target["field"].copy()
        field[:, -1] = values
        return {"boundary_rmse_K": float(np.sqrt(np.mean((values - target["field"][:, -1])**2))),
                "interior_rmse_K": rollout(target, TERMS, coefficients, boundary_field=field)}

    results = {"native_IR": score(optical), "contact_boundary": score(target["field"][:, -1])}
    scores = {}
    for pair in PLANS:
        slopes, offset = fit_plan(x, y, pair)
        scores[pair] = score(basis(optical) @ slopes + offset)
    fixed_endpoints = np.array(FIXED).ravel()
    results["fixed_offset_only"] = score(optical + np.mean(y[fixed_endpoints] - x[fixed_endpoints]))
    results["fixed"] = scores[FIXED]
    results["spread"] = scores[chosen]
    results["spread_plan_min"] = [[10 + 5 * a, 10 + 5 * b] for a, b in chosen]
    results["random_median"] = {metric: float(np.median([v[metric] for v in scores.values()]))
                                for metric in scores[FIXED]}
    return results


def checked(source, name, manifest):
    path = source / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS / "optical_transfer_centered.json")
    args = parser.parse_args()
    manifest = json.loads(SOURCE.read_text())["files"]
    metadata = clock_metadata(checked(args.source, "0_GeneralInformation/StartingTimes.xlsx", manifest))
    runs, audits = {}, {}
    for name in manifest:
        if not name.endswith("/MeasurementData.CSV"):
            continue
        run_id = int(name.split("_", 1)[0])
        run = design(read_run(checked(args.source, name, manifest)), "area")
        ir_name = name.replace("MeasurementData.CSV", "Average_PackedBed_SurfaceTemp.asc")
        time, values, audits[run_id] = camera_series(checked(args.source, ir_name, manifest),
                                                    *metadata[run_id])
        # The original camera export is in degrees Celsius, as are the CSV probes.
        time -= run["shutoff_min"] * 60
        grid = run["grid"]
        use = (grid >= 600) & (grid <= 3000)
        run["grid"], run["field"] = grid[use], run["field"][use]
        runs[run_id] = (run, time, values)
    comparisons = {}
    for shift in (0, -60, 60):
        aligned = {}
        for i, (run, time, values) in runs.items():
            optical = np.interp(run["grid"], time + shift, values, left=np.nan, right=np.nan)
            if not np.all(np.isfinite(optical)):
                raise ValueError(f"Run {i} needs optical extrapolation")
            aligned[i] = {**run, "optical": optical}
        comparisons[shift] = {}
        for i, target in aligned.items():
            sources = [j for j in REPEATS[REGIME[i]] if j != i]
            coefficients = fit_coefficients([aligned[j] for j in sources], TERMS)
            comparisons[shift][i] = {j: evaluate(aligned[j], target, target["optical"], coefficients)
                                     for j in sources}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source_audit": audits, "plans": len(PLANS),
                                    "clock_shift_s": comparisons}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

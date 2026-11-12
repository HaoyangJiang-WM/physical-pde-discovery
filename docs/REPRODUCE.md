# Reproduction guide

## Setup

Use Python 3.12 or 3.13 in a fresh environment. The exact environment used to package and check this release is recorded in `artifacts/package_environment.json`; the publication benchmark environment is also recorded in `benchmark/environment.json`.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Core dependencies are NumPy, SciPy, pandas, numba, Matplotlib and Pillow. No PySINDy download is required: the main inference engine and a transparently named STLSQ control are included. This is not a substitute for an official-package comparison.

## First commands

```bash
python run.py check
python run.py demo
python run.py summary
python run.py figures
python -m unittest discover -s tests -v
```

- `check`: verifies links, main frozen source hashes, archived aggregates, image frames and the noiseless controls. It does not rerun thousands of systems.
- `demo`: **actually runs** both selectors on a single enzyme-reaction world, with shared observations and priors. Writes a full benchmark-style path, a stopped path, and an executable selected equation to `outputs/demo/`. Cache compilation is expected on first use.
- `summary`: recomputes the main and stopping tables from archived task records.
- `figures`: replays archived fits and outcomes, reconstructs sampled observations, and regenerates five GIFs plus static plots. It does not claim new benchmark evidence. Rendering uses local DejaVu Sans when available, otherwise Matplotlib's font discovery. No fonts are distributed.

The README success examples were selected by the original rule: lowest seed initially wrong and uncertain, ultimately correct. Both policies are shown on the same enzyme world. This selection explains a mechanism and is not a random performance sample.

## Full benchmark (substantially larger than the demo)

```bash
python run.py full --workers 4
```

This executes the archived protocol: 11 cases, two noise levels, 192 seeds per cell, 120-point quadrature per active dimension, 32 predictive Sobol samples per support and four policies. Results go to `outputs/full/`, never overwrite the reference archive.

To invoke the original scripts directly:

```bash
python run.py unpack
cd benchmark
OPENBLAS_NUM_THREADS=1 python experiment.py --n 192 --order 120 --power 5 --seed 61000000 --workers 4 --out ../outputs/full
```

For Windows, set thread environment variables in the shell or let `experiment.py` set its defaults. Do not rerun `fixed_plan.py` before evaluating the archived protocol: it changes the fixed baseline. The stored plan was chosen on independent prior draws.

## Data layout

```text
README.md                       English project entry point
assets/                         five GIFs and static figures
docs/                           method, results, controls, roadmap, provenance
run.py                          safe entry points
scripts/                        rendering, validation, publication helpers
tests/                          algebra, inference and data-integrity checks
benchmark/                      preserved source and numerical archives
  replication/trials.csv.gz     main 4,224-task records
  stress/                       separate exploratory / failure archives
  examples/                     reconstructed observations and example histories
artifacts/                      visual records, manifests, package checks
outputs/                        new runs (ignored by git)
```

Large CSV records are gzip-compressed without changing their decompressed bytes. `python run.py unpack` restores the original paths for historical analysis scripts. Observations not stored explicitly can be reconstructed using `experiment.obs` and each recorded case, truth, seed, action, stage and noise condition. The examples additionally include raw NPZ arrays.

## Published repository

The project is published at https://github.com/HaoyangJiang-WM/physical-pde-discovery . Its earlier heat-transfer files are retained in Git history, not in the current tree.

This release rebuilds numerical records and media from frozen source, plans and recorded seeds. See [publication provenance](PROVENANCE.md). Main reference tables remain in `artifacts/original_*.csv`; reconstructed task counts and accuracies are checked against them. Regeneration is not another independent benchmark population. Gzip archives contain the reconstructed records, not a claim of byte identity to earlier runtime-dependent CSVs.

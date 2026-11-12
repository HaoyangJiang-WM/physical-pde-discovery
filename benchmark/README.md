# Fixed-duration active sparse equation discovery: two selectors, many systems

This is a synthetic numerical benchmark, not new SOTA, hardware validation, or an official UQ-SINDy reproduction. Read RESULTS.md for all outcomes and limits.

## Reproduce

Python 3.12/3.13, numpy, scipy, pandas>=2.2, numba, matplotlib. Use one BLAS thread per process. The tensor-quadrature caches are intentionally excluded from the ZIP and regenerate on the first run. Preserve fixed_plans.json: it was trained using independent prior draws, before validation.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python experiment.py --n 192 --order 120 --power 5 --seed 61000000 --workers 4 --out replication
OPENBLAS_NUM_THREADS=1 python experiment.py --cases reaction3 --n 96 --order 48 --power 5 --sigmas .03,.10 --seed 82000000 --workers 1 --out stress/reaction3
OPENBLAS_NUM_THREADS=1 python run_stress.py
OPENBLAS_NUM_THREADS=1 python noise_aware.py
python analyze.py
python analyze_extra.py
python audit_solvers.py
python audit_inference.py
python audit_reaction3.py
python export_examples.py
python make_figures.py
python make_report.py
```

`fixed_plan.py` documents the prior-only fixed-design search; do not rerun it before replaying the stored validation. To rebuild it intentionally, save the original JSON first. For a smoke test use `python experiment.py --cases reaction,enzyme --n 2 --order 80 --power 4 --out smoke`.

## Files and scope

- models.py: 11 two-unknown-mechanism dynamics, plus 3-mechanism reaction.
- experiment.py: one shared spike-and-slab posterior, independent Gaussian observation likelihood, prospective Bayesian structural-risk integration, full-library D-optimality, and identical stopping.
- replication/trials.csv: 4,224 independent tasks; four policies and four rounds are paired rows, not additional independent systems.
- stress/reaction3: 192 additional tasks with eight supports.
- stress/robustness: 672 deliberate stress tasks.
- stress/noise_aware: reuses the same 192 correlated-noise worlds with the true covariance supplied; not another 192 independent tasks.
- audits: timestep, independent solver, spatial, quadrature and predictive-sampling checks.
- examples: archived illustrative observations, all old model coefficients and retained curves.

Raw observations are deterministically reconstructed by `obs` using the recorded case, truth, seed, action, stage, noise and scenario. Examples additionally store raw readings in NPZ. Future observations do not enter the acquisition objective. The final trajectories were simulated to full budget for fair accuracy comparisons; reported stopping uses the first crossing, without future data.

The ordinary library has only two optional terms, known signs, fully observed state or known sufficient Fourier/radial projections, and known reset conditions. The additional library has three optional terms. This is not an arbitrary PDE library or a noise-robust universal discovery algorithm.

The default stopping rule is model-based 95% posterior confidence or budget exhaustion. One-step low-value stopping was intentionally removed because it can overlook complementary experiments. The nonlinear PDE is a 32-site periodic method-of-lines system, with separate grid checks and 64-site truth stress tests.

The fixed-duration design matches acquisition count, duration and sample count, not reagent, reset energy, preparation time or money. Do not describe it as measured lab cost reduction.

"""Generate factual, small supporting docs from the supplied archives."""
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
b=ROOT/'benchmark'
labels={'reaction':'First-/second-order reaction','chemostat':'Buffered co-reactant','reversible':'Reversible reaction','enzyme':'Linear/saturating kinetics','oscillator':'Amplitude dynamics','duffing':'Duffing stiffness','forced_damping':'Forced nonlinear damping','predator':'Two-species interaction','diffusion_pde':'Second-/fourth-order diffusion','advection_diffusion':'Advection–diffusion','allen_cahn':'Nonlinear reaction–diffusion'}
r=pd.read_csv(b/'case_results.csv')
text='''# Results: what is supported, and what is not

## Main synthetic benchmark

**4,224 independent tasks** = 11 families × 2 noise levels × 192 draws. Four policies evaluated on one task are paired alternatives, not four independent tasks. Each starts with one common acquisition and receives at most three new runs. Every acquisition has duration 1.5 and 16 observation times per channel.

The main metric is exact recovery of the optional term set after the full budget. Initial states, signs and specified base mechanisms are known. Two optional terms yield four supports. About 35.9% of tasks have neither optional term; nonempty-support results are reported separately below.

| Family | Random | Prior-optimized fixed | A: D-optimal | B: Structure risk |
|---|---:|---:|---:|---:|
'''
for case,label in labels.items():
    v=r[r.case==case].set_index('policy').accuracy
    text+=f'| {label} | {v["RANDOM"]*100:.2f}% | {v["FIXED"]*100:.2f}% | {v["D_OPT"]*100:.2f}% | {v["STRUCT_RISK"]*100:.2f}% |\n'
text+='''
![All-case results](../assets/benchmark_results.png)

Overall: A = **97.7509%**, B = **97.9167%**, prior-optimized fixed = **97.7983%**, random = **95.0284%**. B−A is +0.1657 percentage points, paired bootstrap 95% interval [−0.1657, +0.4972]. None of the 11 family-specific A/B comparisons passes the original Holm-corrected threshold. There is no established general winner and no claimed improvement over all existing active-SINDy algorithms.

For **2,709 nonempty-support tasks**, A/B = 96.53%/96.75%. The forced nonlinear-damping case remains difficult (~87%); easier tasks must not hide it. The advection–diffusion case already has ~95.57% initial accuracy, so its final 100% is not a large active-learning gain.

## Early stopping is a different endpoint

All policies share the rule: stop at maximum structure posterior ≥95%; otherwise continue to the budget and label the result unresolved. Full paths are archived so full-budget accuracy can be compared separately. Early stopping never uses future correctness.

| Policy | Mean added runs | Single-structure declaration fraction | Accuracy among declarations | Unresolved |
|---|---:|---:|---:|---:|
'''
s=pd.read_csv(b/'overall_stopping.csv').set_index('policy')
for p,label in [('RANDOM','Random'),('FIXED','Prior-optimized fixed'),('D_OPT','A: D-optimal'),('STRUCT_RISK','B: Structure risk')]:
    v=s.loc[p];text+=f'| {label} | {v.mean_runs:.3f} | {v.confident_fraction*100:.2f}% | {v.confident_accuracy*100:.2f}% | {v.unresolved*100:.2f}% |\n'
text+='''
Conditional declaration accuracy is **not** overall recovery. Forcing a MAP answer even on unresolved tasks gives early-stop accuracy 97.49%/97.56% for A/B. This is not the same as 97.75%/97.92% at full budget.

## Known limitations

A two-term, sign-constrained library with matched truth/prior/noise is not open-ended symbolic discovery. Some modes or radial projections are known. The nonlinear reaction–diffusion test is a 32-site periodic method-of-lines model, not an arbitrary PDE solver benchmark. Noise levels are dimensionless toy choices, not equal real-world difficulty. Experimental time and sample counts match; reagent, preparation, energy and money do not.

The posterior is conditional on the candidate library. Missing physics may lead to confident wrong in-library conclusions. Unknown correlated noise and near-zero effects excluded by the prior also invalidate naive interpretations. The main result table is not a claim of coverage under these shifts.

## Exploratory archives, not completed extensions

Three-or-more mechanisms and trustworthy unknown noise modeling are **future work**. Their preliminary pilot/stress records are preserved rather than hidden: `benchmark/stress/reaction3/`, `benchmark/stress/robustness/`, and `benchmark/stress/noise_aware/`. The last reuses the same 192 correlated-noise worlds with the true covariance supplied; it is an oracle diagnostic, not a noise-learning method.

The historical evaluation count including these separate studies is 5,088. The public main benchmark remains 4,224; exploratory tests are not pooled into its success rate. See [the roadmap](ROADMAP.md).

## Numerical evidence and reproducibility

Original checks include independent integration, sensitivity, grid, quadrature and predictive-integration refinement. Main posterior quadrature refinement (263 fixed datasets, order120→240) changed no sampled MAP decision or 95% stopping decision. The Bayesian acquisition estimator can still change actions on refinement; it is not an exact expected-risk oracle.

This packaging turn preserved the original numerical archive and recomputed its summary tables. It did **not** rerun 4,224 tasks or re-establish original performance. The demo is run afresh as a small executable check; the three STLSQ controls are new deterministic diagnostics, not added benchmark trials.

[Paired comparisons](../benchmark/paired_comparisons.csv) · [Original detailed report](../benchmark/RESULTS.md) · [Package checks](../artifacts/package_validation.json) · [Main records (gzip)](../benchmark/replication/trials.csv.gz)
'''
(ROOT/'docs/RESULTS.md').write_text(text)
cs=json.loads((ROOT/'artifacts/failure_controls.json').read_text())
text='''# Passive fitting can be non-identifiable

These figures are deliberately simple counterexamples. The algorithm receives a chosen dictionary and exact derivatives of noise-free states. That removes differentiation errors and shows an information limit. A different sparse optimizer may return a different equally good answer; it cannot prove uniqueness from a rank-deficient experiment.

The implementation is a transparent unregularized **STLSQ** loop with threshold 0.05 and at most 20 iterations, in `scripts/prepare_visual_data.py`. It is not the official PySINDy package. No parameters were tuned on a noisy benchmark to make a competitor look worse. The main Bayesian benchmark does not use these derivative labels.

Each old condition has rank 1 in a two-column library. Adding the new condition gives rank 2. Both state data and the derivative labels satisfy the declared generating equation exactly. Curves extend to t=3; observation points end at 1.5.

'''
for name,c in cs.items():
    text+=f'''## {c['title']}

`{c['initial_equation']}`

{c['explanation']} {c['intervention']}

| Quantity | Value |
|---|---|
| True coefficients ({', '.join(c['coef_names'])}) | `{np.round(c['true_coef'],8).tolist()}` |
| STLSQ on old condition | `{np.round(c['old_coef'],8).tolist()}` |
| STLSQ after combining both conditions | `{np.round(c['new_coef'],8).tolist()}` |
| Old derivative residual norm | {c['before_residual']:.3g} |
| Old-model RMSE on the new condition, t∈[0,3] | {c['new_condition_rmse']:.5f} |

![{c['title']}](../assets/failure_{name}.png)

'''
text+='''The RMSE values above are descriptive for these constructed examples, including the displayed extrapolation interval. They are not estimates of expected performance or the main benchmark metric.

## How this differs from the main success animations

The enzyme, Duffing and reaction–diffusion animations replay **archived Bayesian sparse-library fits from noisy data**. They must not be presented as official STLSQ failures. Both sets of illustrations support the same motivation, but use different estimators and evidential roles.

[Exact control records](../artifacts/failure_controls.json) · [Replay provenance](../artifacts/visual_provenance.json)
'''
(ROOT/'docs/FAILURE_CASES.md').write_text(text)
source_hash=json.loads((ROOT/'artifacts/source_archive.json').read_text())['sha256']
p='''# Provenance and evidence boundaries

## Source artifact

This distribution repackages the supplied `active_discovery_two_schemes_v21.zip`. Its source SHA-256 is:

```
'''+source_hash+'''
```

Original numerical source, fixed plans, protocols, result tables, example observations and audits are preserved under `benchmark/`. `benchmark/REPLICATION_LOCK.json` identifies the frozen main protocol. `benchmark/ARTIFACT_SHA256.json` records original file hashes; `python run.py check` checks all archived files and transparently decompresses CSV bytes for hashing.

## New work in this distribution

Readable English and Chinese READMEs, supporting documentation, five data-driven GIFs, three separate STLSQ ambiguity controls, a working two-selector demo, validators and unit tests. The static/GIF success stories replay original fits and reconstruct original observations. No new large benchmark run is claimed. The noiseless STLSQ controls isolate identifiability and are not included in the reported 4,224 tasks.

## Animation rules

Old coefficients remain frozen. Each model is simulated on the same displayed condition. Each old experiment has its own panel; when a new condition is shown, the old equation is simulated with the new input, not secretly re-fitted. Every curve extends to t=3, while measurements stop at t=1.5. There are no invented late measurements. The visible selected action and scores come from the original task log. Successful examples follow the original declared selection rule; they are not representative accuracy estimates.

## What was and was not validated here

Package checks recompute main results, inspect source hashes, validate links and GIF files, and compare replay coefficients/scores against task logs. Unit tests verify the ambiguity examples, deterministic observation generation and small Bayesian updates. A new enzyme demo runs both selectors and respects the actual stopping condition. These checks do not rerun the archived population study or prove global identifiability.

## External material

No third-party experimental dataset, images, font files, tokens, personal emails, CVs or company-internal data are included. All numerical media are generated from the supplied synthetic project and explicit analytic controls. External papers are linked as references; their figures/text are not copied. Core library versions are recorded in `artifacts/package_environment.json`.

## Repository status

This is a standalone repository-ready package. No existing GitHub repository has been overwritten, and no new remote URL is claimed. The provided publishing helper requires the owner's authenticated GitHub CLI and an explicit visibility choice. A license has not been selected on the owner's behalf; choose an appropriate license before offering reuse permissions.
'''
(ROOT/'docs/PROVENANCE.md').write_text(p)
import scipy,numba,matplotlib,PIL
(ROOT/'artifacts/package_environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,numba=numba.__version__,matplotlib=matplotlib.__version__,pillow=PIL.__version__,packaging_date='2026-11-12'),indent=2)+'\n')
print('Wrote results, failure-control and provenance documents.')

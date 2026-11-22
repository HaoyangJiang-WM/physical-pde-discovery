# Method, protocols and code locations

This file describes everything the code does. Numbers and conclusions are in [RESULTS.md](RESULTS.md). Paths are relative to the repository root, and code is referred to by function or class name.

## 1. Problem

| | |
|---|---|
| Goal | from the current sparse data and SINDy result, choose the next experiment so that the next equation is better |
| Input | noisy measurements; a candidate library Θ; a pool of candidate experiments |
| Output | support and coefficients of every equation; which candidate to run next |
| Not needed | the true equation, the noise level σ, the outcome of any candidate before it is run |

The true equation is used only outside the algorithm, in two places:

- the evaluation metric, which compares the recovered support with the true one;
- the diagnostics that check the noise formula.

## 2. Starting point: two kinds of error

| Error | Cause | Fixed by |
|---|---|---|
| extra terms | standard errors are wrong (wrong noise model), and the threshold looks only at coefficient size | identification: whitening + significance test |
| missing terms | too little information about the term, so its t statistic is small | acquisition: measure where the information is |

Both parts use the same noise model.

## 3. Noise model

A regression row is `r = b − a·ξ`. The features `a` and the target `b` are computed from measurements `m = m_true + σ ε`. First-order error propagation gives

```
Cov(r) = σ² J Jᵀ,      J = ∂r/∂m = ∂b/∂m − (∂a/∂m) ξ
```

`J` is evaluated at the **noisy measurements and the current estimate ξ̂**. `σ` is a common factor and cancels in whitening. Rows of one experiment share measurements, so `JJᵀ` has off-diagonal entries.

| Setting | J |
|---|---|
| points: state noisy, derivative exact | `−∇f(x̃)`, one row; `Var(r) = σ²‖∇f‖²` |
| trajectories, integral form (one row per pair of neighbouring samples) | ±1 at the two ends minus quadrature weight × `∇f`; rows of one trajectory are correlated and whitened as a block |
| PDE, noise on the target only | constant; whitening has no effect |
| graph snapshot, node i uses its neighbours' states | 10×10, nonzero along the edges; one snapshot is one block |

Code: `whiten` in `src/noise_aware.py`. A numerical floor of mean noise variance × 10⁻³ only prevents division by zero.

## 4. Identification

Each equation is identified separately (`identify`):

```
input: rows (A, b), feature derivatives dA, target derivatives dB, rows per experiment `block`
A_w, b_w ← A, b
repeat 3 times:
    (S, ξ̂, ensemble) ← E-SINDy(A_w, b_w)
    J ← dB − dA[:, S] · ξ̂
    for each experiment's block of rows:  L Lᵀ ← J Jᵀ + εI;  A_w ← L⁻¹A;  b_w ← L⁻¹b
output: S, ξ̂, ensemble, A_w
```

E-SINDy (`esindy`) keeps the double bagging of Fasel et al. 2022:

1. **Library bagging.** Run 100 times, each on a random 90% of the library terms. Terms kept in at least 40% of the runs go on.
2. **Data bagging.** Run 100 bootstrap fits to get the coefficient ensemble.
3. **Support and coefficients.** The support is the terms that are nonzero in at least 60% of the ensemble. The coefficients are the ensemble mean on that support.

The only change is inside the sparse regression (`stlsq`):

```
S ← all terms
loop:
    ξ ← least squares(A_w[:, S], b_w)
    if some |ξ_j| < λ: drop them, continue                      # original rule, λ = 0.2
    t_j² = ξ_j² / (s² [(AᵀA)⁻¹]_jj),  s² = RSS / (n − |S|)
    if min t_j² < log n: drop that term, continue               # new: significance test (BIC)
    stop
```

The test is valid only after whitening, because it assumes every row has the same noise.

## 5. Acquisition

| Setting | Criterion | Why |
|---|---|---|
| points, PDE, domain models, graphs: the experiment chooses the state or grid point directly | I-optimal (below) | the candidate's rows are known without prediction |
| trajectories: the experiment chooses only the initial state | ensemble variance (Active E-SINDy's score) | I-optimal needs the states a candidate will visit, predicted by the current model. With the true model's prediction it matches ensemble variance; with the current model it trails by 3–11 points. Four fixes were tried and none closed the gap (RESULTS.md) |

### 5.1 I-optimal

A candidate adds `m` rows: `m = 1` for a point, 10 for a graph snapshot. The score is `prediction_score`:

```
for each equation:
    A_a   ← the candidate's rows
    A_a,w ← whitened with the candidate's own J Jᵀ (J at the current ξ̂ and the candidate's location)
    G     ← A_wᵀ A_w, standardised by its diagonal (scale-invariant; 10⁻⁶ ridge only for numerics)
    P     ← mean of a_w a_wᵀ over the candidate pool               # stands for the whole feasible range
    score += tr(P [G⁻¹ − (G + A_a,wᵀ A_a,w)⁻¹])                     # drop in average prediction variance
choose the highest score
```

Meaning: how much this experiment lowers the model's average prediction variance over the whole feasible range.

- Two groups of terms that predict the same everywhere in the range cannot be told apart, so no budget is spent on them. Wherever their predictions differ, the criterion goes there.
- The score is invariant to column scaling, so no prior is needed.
- The candidate pool is the feasible range, so when the range is widened the criterion itself decides whether, and how far, to leave the initial range.
- Candidates whose predicted rows diverge are excluded.

I-optimality is a classical criterion (Box & Draper 1959; Atkinson & Donev). The contributions here are:

- whitening the candidate rows with the noise model;
- its role in the two-error framework (recovering missing terms);
- letting it choose the experimental range.

Other criteria tested (15 configurations, 24 seeds, mean recovery):

| Criterion | Mean | Mean gap to the best criterion per configuration | Configurations within 5 of the best |
|---|---|---|---|
| ensemble variance (Active E-SINDy) | 49.3 | 20.6 | 4 / 15 |
| D-optimal with prior diag(G)/n | 53.8 | 16.1 | 5 |
| D-optimal without prior | 58.1 | 11.8 | 6 |
| A-optimal | 56.1 | 13.8 | 2 |
| E-optimal | 48.8 | 21.1 | 1 |
| **I-optimal** | **62.9** | **7.0** | **10** |
| Ds-optimal (only the ensemble's undecided terms) | 59.9 | 10.0 | 5 |

D-optimal with the data-scaled prior fails on saturating libraries (Hill, Michaelis–Menten). The prior "fills in" the directions between nearly parallel columns, so sampling piles up in the saturated region. I-optimal needs no prior.

Competitor scores:

| Name | Meaning | Code |
|---|---|---|
| `variance` | Active E-SINDy: standard deviation of the ensemble prediction | `variance_score` |
| `dopt` | *How Low Can You Go?*: D-optimal, raw features, noise-blind, unit-information prior | `dopt_score` |
| `random` | reference only, not an active method | `run_trial` |

## 6. Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| λ | 0.2 | Active E-SINDy code |
| library / data bagging runs | 100 / 100 | same |
| library fraction, the two inclusion thresholds | 0.9, 0.4, 0.6 | same |
| whitening passes | 3 | standard iteratively reweighted least squares |
| significance threshold | log n | BIC, nothing to tune |
| whitening floor | mean noise variance × 10⁻³ | numerics only |

There are no new tunable hyperparameters relative to E-SINDy.

## 7. Arm names

An arm is written `identification+acquisition` and parsed by `run_trial` in `src/experiments.py`.

| Part | Value | Meaning |
|---|---|---|
| identification | `esindy` | original E-SINDy |
| | `esindy_w` / `esindy_t` / `esindy_wt` | + whitening / + significance test / both (ours) |
| | `esindy_wb` | whitening + ARD inside the ensemble (variant for collinear libraries) |
| | `sbl` / `wsbl` | Bayesian SINDy (sparse Bayesian learning) / with our whitening |
| | `wsindy` / `msindy` | Weak SINDy / Modified SINDy (trajectories only) |
| acquisition | `variance` / `dopt` / `iopt` / `random` | section 5; `iopt` is ours |

- `esindy+variance` is the published Active E-SINDy, and `esindy+dopt` is the D-optimal of *How Low Can You Go?*.
- Our method:
  - `esindy_wt+iopt` for points, domain models, the widened range and graphs;
  - `esindy_t+iopt` for PDE, where the weights are 1;
  - `esindy_wt+variance` for trajectories.
- `esindy_wt+info` in older result files is an earlier D-optimal-type acquisition with noise whitening. Its code was removed, and it is kept only for comparison.

**Metric.** Exact-support recovery: the fraction of runs whose recovered terms match the true equation exactly, averaged over the checkpoints of the acquisition curve. Within a seed, all arms share the system, the initial data and the candidate pools (the `env` random stream). Only the algorithms' own random numbers differ, so arms can be compared paired by seed.

## 8. Protocols

### 8.1 Point observations (Active E-SINDy protocol)

Code: `Points` and `SYSTEMS` in `src/experiments.py`.

| | |
|---|---|
| initial data | 20 random states in the sampling box |
| each step | measure 1 of 200 fresh random candidate states |
| budget | 80 steps, 100 points |
| measurement | Gaussian noise on the state, exact derivative (official protocol) |
| noise | 5, 10, 20% × RMS of the state on a reference trajectory |
| library | monomials of degree 1–3, no constant |
| seeds | 64 |

| System | Equations | Box |
|---|---|---|
| Lorenz | ẋ = 10(y − x), ẏ = 28x − y − xz, ż = xy − 8z/3 | x, y ∈ [−25, 25], z ∈ [0, 50] |
| Chen | ẋ = 35(y − x), ẏ = −7x + 28y − xz, ż = xy − 3z | as Lorenz |
| Van der Pol | ẋ = y, ẏ = −x + 2y − 2x²y | x ∈ [−3, 3], y ∈ [−4, 4] |
| Duffing | ẋ = y, ẏ = x − 0.5y − x³ | [−2, 2]² |
| Hopf | ẋ = x − y − x³ − xy², ẏ = x + y − x²y − y³ | [−2, 2]² |
| Lotka–Volterra | ẋ = x − xy, ẏ = −y + xy | [0, 3]² |

Chen and Duffing were added after the method was designed.

### 8.2 Trajectories (*How Low Can You Go?* ODE protocol)

Code: `Trajectories` in `src/experiments.py`.

| | |
|---|---|
| one experiment | choose an initial state, record a 10-sample noisy trajectory |
| initial data / budget | 1 random trajectory; 9 more, chosen from 100 Latin-hypercube initial states |
| rows | integral form; window 1 when whitened, window 5 otherwise |
| candidate rows | simulated forward from the candidate initial state with the current model |
| noise | 5, 10% |
| systems | Lorenz, Van der Pol, Hopf, Lotka–Volterra |
| seeds | 64 |

### 8.3 PDE (*How Low Can You Go?* PDE protocol)

Code: `PDEPoints` and `etdrk4` in `src/experiments.py`.

| | |
|---|---|
| candidates | space-time grid points, from a pool of 5000, without repetition |
| measurement | noisy `u_t` at that point; features from the noise-free field (original protocol) |
| initial data / budget | 16 points; 34 steps (Burgers), 84 steps (KS) |
| library | {1, u, u²} × {u_x, u_xx, u_xxx, u_xxxx} plus u, u²; 14 terms, columns and target normalised |
| noise | 50, 100% of the target RMS |
| systems | Burgers u_t = −u u_x + 0.1 u_xx; Kuramoto–Sivashinsky u_t = −u u_x − u_xx − u_xxxx |
| seeds | 64 |

The paper's own PDE score uses the candidate's true label, so it was left out for fairness.

### 8.4 Biology, chemistry and materials models

Code in `src/domains.py`:

- `FuncLibrary` accepts any list of columns and takes gradients by central differences.
- `MODELS` holds the 11 models.
- `DomainPoints` reuses the point protocol: 20 initial points, 200 candidates per step, up to 100 points.

Each model gets a library containing its true terms and distractors. Columns and target are scaled to unit RMS over the box, so that λ = 0.2 means the same for every model and every arm. The noise is 5% and 10%, with 32 seeds.

| Field | Model | Library |
|---|---|---|
| biology | SIR epidemic | polynomials of degree 0–2 (3 states) |
| biology | FitzHugh–Nagumo neuron | degree 0–3 |
| biology | Selkov glycolysis (a = 0.3, b = 0.6) | degree 0–3 |
| biology | genetic toggle switch | constant, degree 1–2, 1/(1+u²), 1/(1+v²), u/(1+u²), v/(1+v²) |
| chemistry | Michaelis–Menten | constant, degree 1–2, S/(K+S) for K = 0.1, 0.5, 2 |
| chemistry | Brusselator | degree 0–3 |
| chemistry | reversible binding A+B⇌C | degree 0–2 (3 states) |
| chemistry | exothermic Arrhenius reaction | degree 0–2, e(T), c·e(T) |
| materials | Allen–Cahn order parameter | degree 0–5 (1 state) |
| materials | grain-growth phase field | degree 1–3 |
| materials | Landau ferroelectric (damping 1, P⁵ coefficient −1) | degree 1–5 |

With their original parameters, Selkov, Brusselator and Landau had a true term whose normalised coefficient was below λ, so no method could find it. Their parameters or boxes were changed so that every true term is at least 0.3. `python src/domains.py check` prints the smallest normalised true coefficient and noise-free recovery for each model.

### 8.5 Experiments may leave the initial range

Code: `src/expand.py`. `wide_box` is the widening rule. `WideDomain` and `WideODE` reuse 8.4 and 8.1 and change only where candidates are drawn.

| | |
|---|---|
| initial 20 points | in the original box |
| 200 candidates per step | from the wider feasible box |
| widening rule (same for all models and methods) | extend each side by half the box width, except at a physical bound of 0 (concentrations, populations) |
| unchanged | library normalisation, absolute sensor noise |
| models | 6 polynomial systems × 10, 20%; 11 domain models × 5, 10% |
| seeds | 32; the fraction of points taken outside the original box is recorded |

Example: Lorenz [−25, 25]² × [0, 50] becomes [−50, 50]² × [0, 75].

### 8.6 Network dynamics

Code in `src/graphs.py`:

- `GraphLibrary` gives each node's 8 features and their exact derivatives with respect to every node's state.
- `GraphSnapshots` is the protocol, registered as `PROTOCOLS["graph"]`.

| | |
|---|---|
| systems | dx_i/dt = F(x_i) + Σ_j A_ij G(x_i, x_j): SIS, bistable diffusion, competitive Lotka–Volterra |
| graph | 10 nodes, Erdős–Rényi p = 0.3, no isolated node, one graph per seed; topology known |
| library | x, x², x³, Σ A·x_j, Σ A·x_j², x·Σ A·x_j, degree·x, degree·x² |
| one experiment | set the network to one state and measure every node (a snapshot of 10 rows) |
| initial data / budget | 3 snapshots, then 10 more, each chosen from 200 candidate states |
| noise | 5, 10, 20% on the measured states |
| noise model | J is 10×10 and sparse along the edges; each snapshot is whitened as one block (same code as trajectories) |
| seeds | 32 |

`python src/graphs.py check` verifies two things:

- noise-free recovery from 13 snapshots;
- measured residual noise against the formula (ratio 1.00–1.15).

## 9. Scope

- The noise formula is first order. At 20% noise the measured and predicted noise differ by 13–27%. Where the Jacobian is near zero, the second-order term dominates and the formula underestimates.
- Only the noise variance is corrected, not the attenuation bias from noisy features. Debiasing was tested and gave no net gain at these data sizes.
- Correlations between different equations are ignored; each equation is whitened separately.
- The library must contain the true equation.
- The E-SINDy reimplementation follows the paper and the official main script.
- *How Low Can You Go?* released no code, so its unspecified settings follow the Active E-SINDy code or were set here.

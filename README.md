# Active Equation Discovery

Choose the next experiment by asking where the equations that still fit will make different predictions. This repository benchmarks that loop on synthetic ODEs and PDEs.

[Method](docs/METHOD.md) · [Results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCE.md)

## A PDE in one experiment loop

At low amplitude, diffusion alone fits the observations. The next experiment starts at higher amplitude, where the cubic reaction becomes visible.

![Reaction–diffusion replay. A higher-amplitude initial field reveals the cubic reaction. Dots mark observed samples, the dashed curve shows the frozen diffusion fit, and the solid curve shows the updated model.](assets/pde_loop.gif)

One of eight observed sites is shown. Measurements end at $t=1.5$. [Replay data](artifacts/allen_cahn_struct_risk_replay.json)

## Why change the experiment?

| Case | What the first experiment hides | What separates the terms |
| --- | --- | --- |
| Reaction pathway | A fixed co-reactant makes two rate terms proportional | Change the co-reactant concentration |
| Diffusion order | One Fourier mode makes second- and fourth-order diffusion proportional | Excite another mode |
| Oscillator recovery | Growth and saturation cancel on the limit cycle | Start off the cycle |

![Three noise-free controls. The original condition fits multiple equations, and a second condition separates them.](assets/why_intervene.gif)

[Equations and exact-derivative controls](docs/FAILURE_CASES.md)

## Benchmark

**4,224 tasks · 11 system families · 2 noise levels · up to 3 added experiments**

### Exact equation-structure recovery (%)

| Case | Random | Fixed Design | D-optimal | Bayesian |
| --- | ---: | ---: | ---: | ---: |
| First-/second-order reaction | 92.19 | 95.57 | 96.09 | 95.57 |
| Co-reactant kinetics | 97.40 | 99.22 | 99.48 | 99.48 |
| Reversible reaction | 97.92 | 98.96 | 98.70 | 98.96 |
| Saturating kinetics | 91.93 | 97.40 | 97.66 | 97.66 |
| Nonlinear oscillator | 95.83 | 99.22 | 99.22 | 99.48 |
| Duffing oscillator | 93.75 | 99.74 | 99.74 | 99.74 |
| Forced nonlinear damping | 83.07 | 87.50 | 86.98 | 87.24 |
| Predator–prey coupling | 98.70 | 100 | 100 | 100 |
| Second-/fourth-order diffusion | 98.44 | 99.48 | 99.22 | 99.74 |
| Advection–diffusion | 100 | 100 | 100 | 100 |
| Nonlinear reaction–diffusion | 96.09 | 98.70 | 98.18 | 99.22 |
| **Overall** | **95.03** | **97.80** | **97.75** | **97.92** |

Both adaptive selectors use the same Bayesian sparse model. D-optimality targets coefficient information. Structure risk targets errors in term selection. Against optimized fixed, structure risk gained **0.12 percentage points** (paired 95% interval **−0.19 to +0.40**). [Method and assumptions](docs/METHOD.md) · [Paired comparisons](benchmark/paired_comparisons.csv) · [Early stopping results](docs/RESULTS.md#early-stopping-is-a-different-endpoint)

<details>
<summary>More experiment replays</summary>

**Reaction kinetics**

![Bayesian structure-risk replay for saturating reaction kinetics.](assets/bayesian_loop.gif)

![D-optimal replay on the same reaction example.](assets/doptimal_loop.gif)

**Nonlinear stiffness**

![Duffing replay. An amplitude change reveals nonlinear stiffness.](assets/duffing_loop.gif)

</details>

## Run locally

```bash
python -m pip install -r requirements.txt
python run.py demo
python -X utf8 run.py check
```

[Full reproduction guide](docs/REPRODUCE.md) · [Data and code provenance](docs/PROVENANCE.md) · [Roadmap](docs/ROADMAP.md)

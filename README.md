# Active Equation Discovery

## Motivation

Scientific equation discovery aims to recover interpretable governing laws from observations. Methods such as SINDy can identify sparse dynamical equations that accurately explain measured trajectories. However, **a good fit does not guarantee that the underlying physical law is identifiable.**

### Two equations can fit the same experiments

Consider second- and fourth-order diffusion on a periodic domain, each with an unknown linear loss term.

$$
M_1:\quad u_t=D u_{xx}-ru,
\qquad
M_2:\quad u_t=-K u_{xxxx}-\ell u.
$$

Only the initial condition changes. **Each model uses one shared coefficient pair across all runs and modes.** For $u(x,0)=\cos(kx)$, the mode evolves as $u(x,t)=e^{-\lambda_k t}\cos(kx)$. The two decay rates are

$$
\lambda_k^{(1)}=Dk^2+r,
\qquad
\lambda_k^{(2)}=Kk^4+\ell.
$$

**Two modes give two constraints.** The first run excites mode 1 and reveals one decay rate. If the second run excites only mode 2, its rate provides a second constraint. The fourth-order model can match both rates from the second-order model exactly by fitting its two unknown coefficients.

$$
K=\frac{D}{5},
\qquad
\ell=r+\frac{4D}{5}.
$$

**The third mode tests the fit.** Excite modes 2 and 3 together in the second run with $u(x,0)=\cos(2x)+\cos(3x)$. The two modes evolve independently, so we get three rates in just two runs. Matching modes 1 and 2 already fixes both coefficients of the fourth-order model. Its mode-3 rate is $17D+r$, while the second-order model gives $9D+r$. The gap of $8D$ distinguishes the models in noiseless data when $D>0$. At $D=0$, both can describe the same pure decay.

![Mode amplitude over time. The two candidate curves overlap exactly for modes 1 and 2 and separate for mode 3.](assets/fourier_mode_refit.png)

The dashed curve uses the same refitted coefficients in all three panels. The second-order model uses $D=0.10$ and $r=0.15$.

**How can we use the candidate equations that still fit to choose the next experiment?**

## Method

**1. Fit several structures.** Start with a predefined library of terms. Fit model trajectories to the noisy observations $\mathcal D$ to estimate the posterior probability $p(S\mid\mathcal D)$ of each active term set $S$. Within each structure, coefficients are shared across runs.

**2. Choose the next condition.** Both selectors score the same menu of initial conditions or inputs before seeing any new measurements.

- **D-optimal** picks the condition with the largest gain in local coefficient information, measured by $\log\det(G+I_e)-\log\det G$. Here $G$ is the current information matrix and $I_e$ is the contribution estimated for condition $e$ from trajectory sensitivities.
- **Bayesian** picks the condition with the largest expected drop in the probability of choosing the wrong term set. It averages over candidate structures, their coefficients and the noisy measurements $Y_e$ they predict.

$$
\begin{aligned}
R(\mathcal D)&=1-\max_S p(S\mid\mathcal D),\\
U(e)&=R(\mathcal D)-\mathbb E_{Y_e\mid\mathcal D,e}\big[R(\mathcal D\cup\{(e,Y_e)\})\big].
\end{aligned}
$$

**3. Measure and refit.** Add the selected run and update the posterior using all runs. In the early-stopping evaluation, declare a structure once its posterior reaches 95%, or report the task as unresolved at the budget limit. [Method details](docs/METHOD.md)

### Worked example · Reaction–diffusion

In this synthetic replay, the initial amplitude of $0.2$ favors a diffusion-only fit. The Bayesian selector raises the amplitude to $1.0$ at the same spatial frequency, where the cubic term has a larger effect. Refitting both runs brings the cubic term into the selected equation.

$$
\text{Initial fit: } u_t=0.04u_{xx},\qquad
\text{Updated fit: } u_t=0.04u_{xx}-0.3540u^3.
$$

![Synthetic reaction–diffusion replay showing candidate scoring, acquisition and refitting.](assets/pde_loop.gif)

The GIF shows one of eight observed sites. Dots mark samples through $t=1.5$, and the dashed curve uses the initial fit. [Replay data](artifacts/allen_cahn_struct_risk_replay.json)

## Results

The benchmark covers **4,224 synthetic tasks across 11 ODE and PDE families at two noise levels**. Each policy starts with the same initial run and adds three runs. Fixed Design uses a schedule optimized before seeing the task's data.

### Exact equation-structure recovery (%)

Recovery means selecting exactly the optional terms in the generating equation at the end of the three-run budget. The diffusion benchmark tests a library containing both diffusion orders. The Fourier derivation above is a separate illustration.

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

**Both adaptive selectors recovered the correct structure on more tasks than random.** The Bayesian selector gained **2.89 percentage points** over random (paired 95% interval **+2.30 to +3.55**) and **0.12 points** over Fixed Design (interval **−0.19 to +0.40**). The latter comparison leaves the benefit of adapting to each task unresolved. [Paired comparisons](benchmark/paired_comparisons.csv) · [Early stopping and benchmark limits](docs/RESULTS.md#early-stopping-is-a-different-endpoint)

## Run locally

```bash
python -m pip install -r requirements.txt
python run.py demo
python -X utf8 run.py check
```

[Full reproduction guide](docs/REPRODUCE.md) · [Data and code provenance](docs/PROVENANCE.md) · [Roadmap](docs/ROADMAP.md)

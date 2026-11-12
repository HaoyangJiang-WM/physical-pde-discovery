# Active Equation Discovery

Fit several candidate equations to the data, choose an experiment that helps tell them apart, then refit on all acquired runs. The benchmark compares two ways to choose that next experiment on synthetic ODEs and PDEs.

[Motivation](#motivation) · [Method](#method) · [Results](#results) · [Reproduce](docs/REPRODUCE.md)

## Motivation

**Can a perfect fit tell second-order diffusion from fourth-order diffusion?** Consider $u_t=a u_{xx}-b u_{xxxx}$. A sparse regression uses $u_{xx}$ and $-u_{xxxx}$ as its two feature columns. For a single Fourier mode $u(x,t)=A(t)\cos(kx)$,

$$
u_{xx}=-k^2u,\qquad -u_{xxxx}=-k^4u,\qquad
A'(t)=-\underbrace{(ak^2+bk^4)}_{\lambda_k}A(t).
$$

At $k=1$, both columns are exactly $-u$ at every sampled position and time. The data determine only $a+b$. For example, $(a,b)=(0.15,0)$ and $(0.075,0.075)$ both give $\lambda_1=0.15$ and exactly the same trajectory. More samples from that mode keep the same rank-one feature matrix.

![Top row: the two regression features overlap on mode 1 and have a 1:4 amplitude ratio on mode 2. Bottom row: the two candidate equations predict the same trajectory on mode 1 and different decay rates on mode 2.](assets/fourier_mode_identifiability.png)

A run at $k=2$ gives $\lambda_2=4a+16b$. The same two candidates now predict decay rates of $0.60$ and $1.50$. Each mode gives one constraint. The feature ratio changes from $1:1$ to $1:4$, so the two runs together identify both coefficients. This is the motivation for experiment selection: **choose a condition that separates mechanisms the current data leave indistinguishable.** [Exact-derivative controls](docs/FAILURE_CASES.md#one-spatial-mode-hides-the-diffusion-order)

## Method

1. **Fit candidate equations.** From the same predefined library, estimate the probability $p(S\mid D)$ of each term set $S$ given the noisy observations $D$. A term can be exactly zero. All experiments share the same physical coefficients.
2. **Score candidate experiments.** A condition $e$ changes the initial state or input. Both selectors use the same model and candidate experiments.

   - **D-optimality** maximizes the increase $\log\det(G_D+I_e)-\log\det G_D$, where $G_D$ is the current local information about the coefficients and $I_e$ is the estimated contribution from experiment $e$.
   - **Bayesian structure risk** maximizes the expected reduction in the probability of choosing the wrong term set. With $Y_e$ denoting the possible observations from experiment $e$,

     $$
     R(D)=1-\max_S p(S\mid D),\qquad
     U(e)=R(D)-\mathbb{E}_{Y_e\mid D}[R(D\cup\{e,Y_e\})].
     $$

3. **Measure and refit.** Add the new observations and update the term probabilities. For early stopping, report a structure at 95% posterior probability or mark it unresolved if the budget runs out.

### Worked example · Reaction–diffusion

At low amplitude the fit selects diffusion alone. The structure-risk selector chooses a higher-amplitude initial field, and refitting both runs adds the cubic reaction:

$$
\text{Initial fit: } u_t=0.04u_{xx},\qquad
\text{Updated fit: } u_t=0.04u_{xx}-0.3540u^3.
$$

![Synthetic reaction–diffusion replay showing candidate scoring, acquisition and refitting.](assets/pde_loop.gif)

One of eight observed sites is shown. Dots are observed samples through $t=1.5$. The dashed curve keeps the initial fit frozen. [Replay data](artifacts/allen_cahn_struct_risk_replay.json)

## Results

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

**Both adaptive selectors beat random.** Structure risk gained **0.12 percentage points** over optimized fixed (paired 95% interval **−0.19 to +0.40**), leaving the adaptive advantage over fixed unresolved. [Method and assumptions](docs/METHOD.md) · [Paired comparisons](benchmark/paired_comparisons.csv) · [Early stopping results](docs/RESULTS.md#early-stopping-is-a-different-endpoint)

## Run locally

```bash
python -m pip install -r requirements.txt
python run.py demo
python -X utf8 run.py check
```

[Full reproduction guide](docs/REPRODUCE.md) · [Data and code provenance](docs/PROVENANCE.md) · [Roadmap](docs/ROADMAP.md)

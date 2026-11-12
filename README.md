# Active Equation Discovery

### When several equations fit, change the experiment—not just the fit.

**Can a partially identified equation help choose the next experiment that identifies it?** This repository makes that question executable: fit a sparse dynamical model, choose a new experimental condition, collect new observations, and update the equation.

[How it works](docs/METHOD.md) · [Results and limits](docs/RESULTS.md) · [Reproduce](docs/REPRODUCE.md)

> **Scope:** a reproducible synthetic benchmark, not a new SINDy algorithm or a claim of state-of-the-art performance. Two established experiment-selection principles share the same Bayesian sparse-library inference engine.

## Why fitting the data is not enough

Suppose a reaction obeys

$$\dot x=-ax-bcx,$$

where the buffered co-reactant concentration $c$ stays constant within each run. If every run uses $c=1$, the data reveal only $a+b$. A direct reaction, a co-reactant-dependent reaction, and a mixture can produce exactly the same trajectory. **Watching that condition longer does not separate the mechanisms.**

Changing $c$ introduces a different constraint. The same physical coefficients must explain both experiments. The goal is therefore not simply a better curve fit:

> **Use the equations still compatible with the data to choose a condition under which they can no longer give the same explanation.**

This matters when experiments are limited: a useful model should help decide what to measure next—and when the current scientific question is sufficiently resolved.

## Watch the complete loop

The animation below replays a noisy reaction example. The initial linear model looks plausible at low concentration. The selected higher-concentration run reveals saturating kinetics, and the updated structure posterior passes the declared stopping threshold.

![Full Bayesian loop: observed points, current equation, prospective experiment scores, new measurements, refitting and stopping.](assets/bayesian_loop.gif)

**Read the animation:** dots are acquired synthetic measurements; dashed lines keep the **old coefficients frozen**; solid lines are updated predictions. Every run has the same **16 observation times and duration $T=1.5$**. Curves extend to $t=3$ for comparison, but everything after $1.5$ is **prediction only**. The old experiment remains visible after the new one is added.

This is a selected illustrative success, not a performance average. [Data and provenance](artifacts/enzyme_struct_risk_replay.json).

## Three ways passive SINDy can remain ambiguous

These small controls use a transparent STLSQ implementation—the sequential thresholded least-squares idea used in SINDy. We deliberately supply **noise-free states and exact derivatives** to isolate missing information, rather than blame numerical differentiation. They are not runs of the official PySINDy package and are not included in the benchmark percentages.

| Case | Why the old experiment is insufficient | What breaks the ambiguity |
|---|---|---|
| **Reaction pathway** | At fixed $c=1$, the library columns $x$ and $cx$ coincide. | Change the co-reactant concentration. |
| **Diffusion order** | On one Fourier mode, $u_{xx}$ and $u_{xxxx}$ are proportional. | Excite another spatial mode. |
| **Radial recovery** | On a limit cycle, growth and saturation can cancel exactly. | Start away from the cycle. |

![Three STLSQ identifiability controls: a perfect old-condition fit can fail after an intervention; shared-parameter refitting resolves these examples.](assets/why_intervene.gif)

[Reaction figure](assets/failure_co_reactant.png) · [PDE figure](assets/failure_pde_mode.png) · [Oscillator figure](assets/failure_limit_cycle.png) · [Equations, coefficients and checks](docs/FAILURE_CASES.md)

**These are limits of the available experiments, not a claim that SINDy is generally unreliable.** No estimator can uniquely recover two mechanisms whose effects are identical under every allowed measurement.

## One discovery loop, two experiment selectors

```text
Existing measurements
        ↓
Sparse equation structures + uncertain coefficients
        ↓
A. D-optimal information     OR     B. Bayesian structure risk
        ↓
Change concentration / initial state / forcing / spatial excitation
        ↓
Acquire new data → refit all runs with shared coefficients → repeat or stop
```

| | A · D-optimal | B · Bayesian structure risk |
|---|---|---|
| Question | Which experiment adds independent parameter information? | Which experiment most reduces the risk of choosing the wrong equation structure? |
| Computation | Local sensitivities of observable trajectories; all candidate terms remain available. | Average over uncertain structures, continuous coefficients, and possible new observations. |
| Shared components | Same library, prior, likelihood, candidate actions, observations, and stopping rule. | Same library, prior, likelihood, candidate actions, observations, and stopping rule. |

The main engine uses a **spike-and-slab prior**: a term is either exactly absent or has an uncertain continuous coefficient. This is **Bayesian SINDy-style sparse-library discovery**, not an unchanged PySINDy/STLSQ or UQ-SINDy implementation. The current main benchmark has **two optional mechanisms per family**. More about [the implementation and classical references](docs/METHOD.md).

<details>
<summary><strong>Watch selector A on the same reaction data</strong></summary>

![D-optimal replay with actual archived action scores and retained old predictions.](assets/doptimal_loop.gif)

In this example, both selectors choose the same next condition. Their scores have different meanings and must not be compared numerically across methods.

</details>

<details>
<summary><strong>More animated cases: nonlinear stiffness and a nonlinear PDE</strong></summary>

![Duffing stiffness discovery after an amplitude change; old predictions are retained.](assets/duffing_loop.gif)

![Reaction–diffusion example: change the initial field, discover a nonlinear reaction term.](assets/pde_loop.gif)

The PDE panel shows one of eight observed spatial sites; inference uses all eight. These are archived synthetic examples, not hardware measurements.

</details>

## What the benchmark shows

**4,224 independent tasks:** 11 dynamical families × 2 noise levels × 192 draws. Each task starts with one experiment and allows three additional runs. Experiment duration and samples are fixed; truth coefficients are continuous, not restricted to inference-grid nodes.

| Experiment selector | Exact term-set recovery after all 3 added runs | Mean added runs with the common stopping rule |
|---|---:|---:|
| Random conditions | 95.03% | 1.435 |
| Prior-optimized fixed design | 97.80% | 1.023 |
| **A · D-optimal** | **97.75%** | **1.035** |
| **B · Structure risk** | **97.92%** | **1.004** |

**The two columns describe different endpoints:** full-budget accuracy versus the first posterior-threshold stopping time. They do not imply that stopping is lossless.

![All 11 families, including the more difficult forced-damping case.](assets/benchmark_results.png)

B minus A is **+0.17 percentage points**, with a paired 95% bootstrap interval of **−0.17 to +0.50**. **There is no established universal winner.** Fixed design is also strong. [Full results, nonempty-support analysis and failure tests](docs/RESULTS.md).

Stop when the largest structure posterior reaches 95%; otherwise use the budget and report **unresolved**. The unresolved fractions are **7.60% for A** and **6.39% for B**. This is a model-conditional rule, not a distribution-free reliability guarantee.

## Run it

Python 3.12 or 3.13. No GPU, account credentials, or external dataset download is needed for the demo.

```bash
python -m pip install -r requirements.txt
python run.py check       # verify archived results, links, source hashes, GIFs
python run.py demo        # run both selectors on a small reaction campaign
python run.py figures     # regenerate the README figures and animations
python run.py summary    # summarize the archived 4,224-task benchmark
```

[Full benchmark commands, file layout and expected outputs](docs/REPRODUCE.md). The inference cache is regenerated locally; it is not distributed. The full benchmark is much larger than the demo.

## Limits and future work

The current evidence is synthetic, with small predefined libraries, known initial conditions, fixed sign restrictions, and mostly matched priors and independent Gaussian noise. Some state or Fourier-mode projections are known in advance. The nonlinear PDE uses a small periodic method-of-lines grid. Acquisition counts are matched; reagent, preparation and energy costs are not.

**Future work focuses on two problems—not additional easy cases:**

- **Three or more competing mechanisms:** scale beyond two optional terms, retain plausible alternatives, and test whether new experiments genuinely separate adjacent or compensating mechanisms.
- **Trustworthy noise and model checks:** learn correlated or heteroscedastic errors, avoid false certainty under missing physics, and validate statements made after adaptive stopping.

Initial pilot/stress results are retained in the archive, but neither problem is presented as solved. Real experimental campaigns and official PySINDy/PDE-FIND/PySR comparisons remain future validation. [Roadmap](docs/ROADMAP.md).

## References

This project builds on established ideas; it does not introduce D-optimality, spike-and-slab inference, or active SINDy.

1. Brunton, Proctor & Kutz (2016), [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113).
2. Fasel et al. (2022), [Ensemble-SINDy: robust sparse model discovery with active learning and control](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006119/).
3. Hirsh, Barajas-Solano & Kutz (2022), [Sparsifying priors for Bayesian uncertainty quantification in model discovery](https://pmc.ncbi.nlm.nih.gov/articles/PMC8864363/).
4. Huan & Marzouk (2013), [Simulation-based optimal Bayesian experimental design for nonlinear systems](https://arxiv.org/abs/1108.4146).
5. Hainy et al. (2022), [Optimal Bayesian design for model discrimination via classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC8924111/).

[Data and code provenance](docs/PROVENANCE.md). All visualizations are generated from included numerical records; no generative illustration is used as experimental evidence.

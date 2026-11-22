# Results

All numbers are exact-support recovery (%) averaged over the acquisition curve and over seeds. Settings are described in [METHOD.md](METHOD.md).

## 1. Final comparison

Tables are produced by `python src/report.py` from `results/*.json`. For each configuration, the best value is in bold.

- The bottom row gives the paired comparison of ours against each column (better / tie / worse, by more than 2 standard errors over common seeds).
- "Bayesian SINDy + active" is sparse Bayesian learning with our I-optimal acquisition (32 seeds).
- "our identification + random" is a reference, not a competitor.

### Point observations (Active E-SINDy protocol) (18 configurations)

| system | noise | Active E-SINDy | D-optimal | Bayesian SINDy + active | **ours** | our identification + random |
|---|---|---|---|---|---|---|
| lorenz | 5% | 66.0 | 65.1 | 52.4 | **91.7** | 91.5 |
| lorenz | 10% | 22.7 | 18.4 | 30.9 | **83.9** | 81.6 |
| lorenz | 20% | 2.6 | 2.1 | 19.1 | **77.1** | 58.2 |
| vanderpol | 5% | 93.1 | 98.1 | 95.1 | **100.0** | **100.0** |
| vanderpol | 10% | 69.6 | 81.9 | 62.8 | **97.4** | 96.5 |
| vanderpol | 20% | 20.7 | 31.1 | 20.1 | **88.2** | 83.0 |
| hopf | 5% | **100.0** | 99.8 | **100.0** | **100.0** | **100.0** |
| hopf | 10% | 73.6 | 78.8 | 95.1 | 99.1 | **99.3** |
| hopf | 20% | 48.8 | 58.2 | 16.0 | **92.0** | 86.5 |
| lotka | 5% | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| lotka | 10% | 91.0 | 93.4 | **99.7** | 99.5 | **99.7** |
| lotka | 20% | 33.3 | 29.0 | **97.6** | 94.6 | 91.7 |
| chen | 5% | 56.2 | 52.6 | **96.2** | 92.2 | 87.7 |
| chen | 10% | 14.8 | 11.6 | 78.5 | **87.0** | 85.2 |
| chen | 20% | 1.0 | 1.2 | 63.5 | **78.1** | 68.6 |
| duffing | 5% | 99.8 | **100.0** | **100.0** | **100.0** | **100.0** |
| duffing | 10% | 97.6 | 94.1 | 98.3 | **100.0** | **100.0** |
| duffing | 20% | 38.2 | 50.3 | 59.0 | **99.8** | **99.8** |
| **mean** | | 57.2 | 59.2 | 71.4 | 93.4 | 90.5 |
| ours: better / tie / worse | | 15/3/0 | 15/3/0 | 12/5/1 |  | 6/12/0 |

### Biology, chemistry and materials models, original range (22 configurations)

| system | noise | Active E-SINDy | D-optimal | Bayesian SINDy + active | **ours** |
|---|---|---|---|---|---|
| sir | 5% | 95.5 | 94.1 | **99.3** | 99.0 |
| sir | 10% | 53.1 | 54.5 | 90.6 | **92.7** |
| fhn | 5% | 99.0 | 98.3 | **100.0** | **100.0** |
| fhn | 10% | 87.5 | 85.8 | 99.0 | **100.0** |
| selkov | 5% | 2.1 | 3.5 | **79.2** | 75.0 |
| selkov | 10% | 0.0 | 0.3 | **28.5** | 6.6 |
| toggle | 5% | 52.1 | 35.4 | **93.4** | 46.9 |
| toggle | 10% | 18.4 | 7.3 | **75.0** | 28.5 |
| mm | 5% | 47.6 | 1.0 | 64.9 | **81.2** |
| mm | 10% | 5.6 | 0.0 | 18.1 | **31.6** |
| brusselator | 5% | 1.7 | 3.5 | **92.7** | 79.9 |
| brusselator | 10% | 0.0 | 0.0 | **47.9** | 40.6 |
| binding | 5% | 96.5 | 95.1 | **99.3** | 97.9 |
| binding | 10% | 70.5 | 68.4 | **93.8** | 90.3 |
| arrhenius | 5% | 0.0 | 0.0 | 0.0 | **12.5** |
| arrhenius | 10% | 0.0 | 0.0 | 0.0 | **2.4** |
| allencahn | 5% | 12.8 | 4.5 | 26.4 | **88.9** |
| allencahn | 10% | 1.4 | 0.7 | 11.8 | **63.2** |
| grain | 5% | **100.0** | **100.0** | **100.0** | **100.0** |
| grain | 10% | 59.4 | 48.3 | 92.4 | **100.0** |
| landau | 5% | 37.8 | **55.6** | 18.1 | 41.3 |
| landau | 10% | 35.8 | 37.5 | **50.7** | 15.3 |
| **mean** | | 39.9 | 36.1 | 62.8 | 63.4 |
| ours: better / tie / worse | | 16/5/1 | 18/3/1 | 8/8/6 |  |

### Experiments may leave the initial range (34 configurations)

| system | noise | Active E-SINDy | D-optimal | Bayesian SINDy + active | **ours** |
|---|---|---|---|---|---|
| sir | 5% | 95.8 | 97.2 | **99.7** | 99.0 |
| sir | 10% | 76.4 | 84.0 | **95.5** | 94.8 |
| fhn | 5% | 97.6 | 95.8 | 97.6 | **100.0** |
| fhn | 10% | 70.5 | 75.3 | 70.5 | **100.0** |
| selkov | 5% | 4.2 | 10.4 | 78.8 | **89.9** |
| selkov | 10% | 0.0 | 0.3 | 14.9 | **56.2** |
| toggle | 5% | 85.8 | 83.0 | **99.0** | 90.6 |
| toggle | 10% | 58.3 | 43.1 | **90.3** | 74.7 |
| mm | 5% | 77.4 | 13.9 | 60.8 | **94.4** |
| mm | 10% | 27.8 | 2.1 | 32.6 | **71.9** |
| brusselator | 5% | 5.2 | 17.7 | 91.3 | **93.1** |
| brusselator | 10% | 0.3 | 0.7 | 37.8 | **65.6** |
| binding | 5% | 96.9 | 96.9 | **99.3** | 97.9 |
| binding | 10% | 81.9 | 87.2 | **95.8** | 94.8 |
| arrhenius | 5% | 0.3 | 0.3 | 16.7 | **45.1** |
| arrhenius | 10% | 0.0 | 0.3 | 0.0 | **7.6** |
| allencahn | 5% | 87.8 | 88.5 | 95.8 | **99.3** |
| allencahn | 10% | 60.8 | 39.6 | 82.3 | **98.6** |
| grain | 5% | 94.1 | 96.9 | 96.5 | **100.0** |
| grain | 10% | 31.6 | 43.4 | 29.2 | **100.0** |
| landau | 5% | 15.3 | 11.8 | 48.6 | **91.0** |
| landau | 10% | 2.8 | 19.8 | 5.6 | **84.4** |
| lorenz | 10% | 14.6 | 13.2 | 44.1 | **86.8** |
| lorenz | 20% | 1.7 | 2.1 | 19.1 | **79.5** |
| chen | 10% | 11.8 | 10.4 | 80.6 | **90.6** |
| chen | 20% | 1.0 | 0.3 | 67.7 | **80.9** |
| vanderpol | 10% | 74.0 | 69.8 | 24.3 | **96.2** |
| vanderpol | 20% | 36.5 | 38.2 | 6.9 | **65.3** |
| duffing | 10% | 72.2 | 86.1 | 81.9 | **100.0** |
| duffing | 20% | 54.9 | 52.4 | 15.6 | **99.7** |
| hopf | 10% | 71.5 | 74.0 | 53.5 | **99.3** |
| hopf | 20% | 56.6 | 50.0 | 3.1 | **92.0** |
| lotka | 10% | 99.3 | 99.3 | **99.7** | 99.3 |
| lotka | 20% | 86.8 | 92.7 | **97.6** | 95.8 |
| **mean** | | 48.6 | 47.0 | 59.8 | 86.3 |
| ours: better / tie / worse | | 31/3/0 | 31/3/0 | 24/6/4 |  |

### Trajectories (How Low Can You Go? ODE protocol); ours uses ensemble variance here (8 configurations)

| system | noise | Active E-SINDy | Bayesian SINDy + active | Modified SINDy + active | Weak SINDy + active | **ours** | our identification + random |
|---|---|---|---|---|---|---|---|
| lorenz | 5% | 16.1 | 11.9 | 0.0 | 0.0 | **56.4** | 38.9 |
| lorenz | 10% | 1.4 | 1.2 | 0.0 | 0.0 | **25.5** | 8.4 |
| vanderpol | 5% | 64.5 | **74.4** | 56.2 | 22.5 | 72.8 | 66.1 |
| vanderpol | 10% | 47.7 | **65.9** | 44.1 | 14.1 | 64.7 | 55.2 |
| hopf | 5% | 60.8 | 65.6 | 51.9 | 0.3 | **69.5** | 55.8 |
| hopf | 10% | 48.6 | 43.8 | 27.5 | 0.0 | **59.4** | 38.6 |
| lotka | 5% | 65.9 | 64.4 | 14.7 | 33.8 | **73.0** | 53.1 |
| lotka | 10% | 56.6 | 48.4 | 4.7 | 21.9 | **64.5** | 36.6 |
| **mean** | | 45.2 | 47.0 | 24.9 | 11.6 | 60.7 | 44.1 |
| ours: better / tie / worse | | 8/0/0 | 6/2/0 | 8/0/0 | 8/0/0 |  | 8/0/0 |

### PDE (How Low Can You Go? PDE protocol); noise on the target only (4 configurations)

| system | noise | Active E-SINDy | D-optimal | Bayesian SINDy + active | **ours** | our identification + random |
|---|---|---|---|---|---|---|
| burgers | 50% | 39.6 | 37.2 | **79.5** | 57.6 | 4.5 |
| burgers | 100% | 7.8 | 3.3 | **55.9** | 29.3 | 2.3 |
| ks | 50% | 76.2 | 75.9 | 85.8 | **86.8** | 67.9 |
| ks | 100% | 42.2 | 30.6 | 69.1 | **76.9** | 58.3 |
| **mean** | | 41.4 | 36.7 | 72.6 | 62.7 | 33.2 |
| ours: better / tie / worse | | 4/0/0 | 4/0/0 | 1/1/2 |  | 4/0/0 |

### Network dynamics, topology known (9 configurations)

| system | noise | Active E-SINDy | D-optimal | Bayesian SINDy + active | **ours** | our identification + random |
|---|---|---|---|---|---|---|
| sis | 5% | 46.5 | 39.6 | 87.8 | **96.2** | 86.5 |
| sis | 10% | 4.9 | 2.8 | 55.6 | 61.8 | **63.9** |
| sis | 20% | 0.3 | 0.0 | **29.9** | 17.4 | 21.5 |
| bistable | 5% | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| bistable | 10% | 89.9 | 90.3 | 90.6 | **100.0** | **100.0** |
| bistable | 20% | 20.5 | 21.9 | 37.8 | **93.8** | 88.5 |
| competition | 5% | 42.7 | 49.0 | 85.1 | 99.0 | **99.3** |
| competition | 10% | 10.8 | 4.9 | 47.2 | **88.9** | 85.4 |
| competition | 20% | 0.3 | 0.7 | 24.7 | **54.9** | 52.4 |
| **mean** | | 35.1 | 34.3 | 62.1 | 79.1 | 77.5 |
| ours: better / tie / worse | | 8/1/0 | 8/1/0 | 6/2/1 |  | 2/7/0 |

## 2. Diagnosis of Active E-SINDy

| Observation | Cause |
|---|---|
| Errors are almost all extra terms; true terms are rarely lost (Lorenz fails mostly by adding x, y to ż) | λ compares coefficient size, not significance. An x term just above 0.2 contributes far less than the local noise and is still kept |
| Residual noise varies tens of times across the state space | state noise enters through the features, and E-SINDy treats it as constant |
| Points chosen by ensemble variance carry 1.4–2.1× the noise of random points and hug the box boundary | the ensemble varies most where the features are large, which is also where the noise is large |
| At ≥ 10% noise in the point setting, Active E-SINDy is worse than random sampling | as above: acquisition runs to the noisiest region |

## 3. The noise formula holds outside polynomial ODEs

The experiment covers 11 biology, chemistry and materials models (23 equations, including Hill, Michaelis–Menten and Arrhenius forms). With 5% state noise at 40,000 random states, it compares the measured residual sd with `σ‖∂f/∂x‖`:

| Field | Model | Measured / predicted | Noise spread within one equation |
|---|---|---|---|
| biology | SIR | 0.97–1.08 | 6–7× (third equation linear: 1×) |
| biology | FitzHugh–Nagumo | 0.98–1.04 | 5× (second linear: 1×) |
| biology | Selkov | 0.93–1.39 | 37–52× |
| biology | toggle switch (Hill) | 0.97–1.04 | 2.7× |
| chemistry | Michaelis–Menten | 0.93–1.07 | 38× |
| chemistry | Brusselator | 0.94–1.34 | 25–33× |
| chemistry | binding A+B⇌C | 0.98–1.01 | 3.9× |
| chemistry | Arrhenius | 0.91–1.32 | 13–15× |
| materials | Allen–Cahn | 1.00–2.09 | 31× |
| materials | grain growth | 0.98–1.12 | 20× |
| materials | Landau | 0.98–1.42 | 37× (first linear: 1×) |

In 20 of the 23 equations, the noise varies by more than 2× (median 13×); the other 3 are linear. The largest deviations occur where the Jacobian is near zero (Allen–Cahn), where the second-order term dominates. This check uses the true model on purpose; the algorithm does not.

## 4. Which part fixes which error

The table shows the share of runs with each kind of error at 20% noise, in the point setting:

| System | E-SINDy + variance: extra / missing | our identification + variance: extra / missing | ours: extra / missing |
|---|---|---|---|
| Lorenz | 97 / 7 | 11 / 57 | 14 / 13 |
| Chen | 99 / 0 | 16 / 21 | 19 / 7 |
| Van der Pol | 66 / 27 | 11 / 31 | 9 / 4 |
| Hopf | 30 / 34 | 8 / 26 | 8 / 2 |

- **Identification removes extra terms.** It cuts extra terms from 30–99% to 8–16%.
- **Variance sampling then loses true terms.** Paired with the original variance acquisition, missing terms rise to 21–57%, because that score samples the noisiest places.
- **Our acquisition recovers them.** With our acquisition, missing terms drop to 2–13%, and extra terms stay low.
- **Whitening and the test need each other.** On Lorenz and Chen, whitening alone does nothing and the test is needed. On Van der Pol and Hopf, the test alone makes things worse, because its standard errors are wrong, and whitening is needed.

## 5. Information limits: Arrhenius

A c-optimal design gives the best t² any sampling could reach for the constant and T terms of dT/dt. It assumes 100 points, a threshold t² = log 100 = 4.6, and that the true model is known:

| Upper temperature | 5%: best t² (const / T) | 5%: points needed | 10%: best t² | 10%: points needed |
|---|---|---|---|---|
| 1.6 (initial box) | 0.07 / 0.01 | 13,742 / 163,843 | 0.02 / 0.00 | 63,165 / 684,430 |
| 2.0 (widening rule) | 1.8 / 0.2 | 329 / 5,295 | 0.4 / 0.04 | 1,684 / 24,349 |
| 3.0 | 36.4 / 6.0 | 100 / 100 | 9.1 / 1.5 | 100 / 399 |
| 4.0 | 56.9 / 22.1 | 100 / 100 | 14.2 / 5.5 | 100 / 100 |

So on the original box no acquisition rule can identify Arrhenius. With the temperature widened to 4.0 (32 seeds, `results/arrhenius_T4.json`):

| Noise | Active E-SINDy | D-optimal | our identification + variance | ours |
|---|---|---|---|---|
| 5% | 1.0 | 1.7 | 38.9 | **51.4** |
| 10% | 0.3 | 0.0 | 2.4 | **12.5** |

The result points the way theory predicts, but stays far below the bound. The bound is for one term with a known model.

## 6. Trajectories: why I-optimal is not used there

Mean recovery over 32 seeds, with our identification:

| Acquisition | Lorenz 5% | Hopf 10% | Lotka–Volterra 10% |
|---|---|---|---|
| ensemble variance | 55.6 | 60.6 | 66.6 |
| I-optimal | 43.8 | 54.4 | 58.4 |
| I-optimal, candidate trajectories simulated with the true model (diagnostic only) | **54.4** | **60.6** | **66.9** |
| I-optimal, noise weights from the true model (diagnostic only) | 51.9 | 56.9 | 59.4 |

The gap comes from predicting which states a candidate initial condition will visit. It does not come from the criterion or from the noise weights. Fixes that were tried, none of which closed the gap:

| Fix | Lorenz 5% | Hopf 10% | Lotka–V. 10% |
|---|---|---|---|
| features at the initial state only | 46.2 | 55.0 | 54.4 |
| ensemble-mean model for the prediction | 48.4 | 57.8 | 58.4 |
| ensemble covariance in place of G⁻¹ | 51.9 | 52.5 | 61.3 |
| both | 43.4 | 55.6 | 61.6 |
| averaging I-optimal over ensemble members (pseudo-Bayesian) | tie with I-optimal | | |

**Lorenz 10% stays low because of an information limit.** The only missed term is −y in ẏ = 28x − y − xz, kept in 44% of runs; every other true term is kept 100% of the time. Even when the true structure is given, the median t² of −y is about 3, below log n ≈ 4.5. Dropping it is the correct behaviour under weak evidence. The original E-SINDy keeps it only because it also keeps about 3 extra terms.

## 7. Literature identifiers, passive and with our active selection

Each method was re-implemented from its paper (`src/baselines.py`) and run twice:

- **passive**, with random sampling, as in the paper;
- **active**, with our I-optimal acquisition.

The loop, candidates and budget are identical in both. Weak SINDy and Modified SINDy need time-ordered data, so they run only in the trajectory setting. Results are in `results/baselines.json`, 32 seeds.

| Setting | Configs | Bayesian SINDy passive / active |
|---|---|---|
| points | 18 | 74.2 / 71.4 |
| domain models, original box | 22 | 64.3 / 62.8 |
| widened range, domain models | 22 | 69.8 / 65.4 |
| widened range, polynomial ODEs | 12 | 59.9 / 49.5 |
| PDE | 4 | 37.2 / 72.6 |

| Trajectories (8 configs) | passive | active |
|---|---|---|
| Weak SINDy | 6.4 | 11.6 |
| Bayesian SINDy | 33.0 | 47.0 |
| Modified SINDy | 13.4 | 24.9 |

- Adding our acquisition improves all three literature methods on trajectories.
- Bayesian SINDy with our whitening added (`wsbl`) is never worse than plain Bayesian SINDy in 74 point-type configurations. The noise model helps another identifier too.
- Weak SINDy is poor here because each trajectory has only 10 samples, too short for its smooth test functions. This is a protocol mismatch, not a flaw of the method.

## 8. Pretrained model (passive)

Each seed gives one 100-sample noisy trajectory, the input format ODEFormer was trained on. All methods get the same data and no acquisition. ODEFormer runs with the official weights and beam size 50, with no training or tuning (`src/compare_pretrained.py`, 32 seeds, exact structure):

| System | Noise | E-SINDy | our identification | ODEFormer |
|---|---|---|---|---|
| Lorenz | 5 / 10% | 6.2 / 0.0 | **37.5** / 0.0 | 0.0 / 0.0 |
| Chen | 5 / 10% | 15.6 / 9.4 | **37.5** / 9.4 | 0.0 / 0.0 |
| Van der Pol | 5 / 10% | 100 / 81.2 | 100 / **93.8** | 12.5 / 9.4 |
| Duffing | 5 / 10% | 15.6 / 6.2 | **56.2 / 34.4** | 6.2 / 0.0 |
| Hopf | 5 / 10% | 40.6 / **12.5** | **59.4** / 3.1 | 0.0 / 0.0 |
| Lotka–Volterra | 5 / 10% | 50.0 / 46.9 | **65.6 / 50.0** | 28.1 / 25.0 |

ODEFormer does not know the answer is polynomial. In 169 of its 384 outputs it used non-polynomial terms such as sin or reciprocals. Only structure is compared here, not prediction accuracy.

## 9. Approaches tried and dropped

| Attempt | Target | Result | Decision |
|---|---|---|---|
| pseudo-Bayesian I-optimal (average over ensemble members) | trajectories, Landau 10% | 28 configs: 0 better / 28 tie / 0 worse vs I-optimal | dropped |
| model-discrimination design (T-optimal, grouped by structure) | imitating terms | vs I-optimal: 3 / 14 / 5 | dropped; useful only when candidates come from different libraries |
| unify selection: whitening + ARD (single, ensemble), Bayesian selection with a shared g-prior and Gibbs sampling over structures, with and without a non-local prior | collinear libraries | large gains on collinear libraries, but Lorenz and Chen 20% lose 14–60; with a minimum-effect prior the trade reverses | ensemble ARD kept as variant `esindy_wb` |
| second-order coefficient debiasing | attenuation bias | larger variance; coefficient error did not drop | dropped |
| A-optimal in place of D-optimal | saturating libraries | fixes Hill / Michaelis–Menten, but Lorenz 20% −45 | replaced by I-optimal |

The unification results (24 seeds, mean recovery):

| Variant | Lorenz 20% | Chen 20% | Hopf 20% | toggle 5% | M–M 5% | Allen–Cahn 10% |
|---|---|---|---|---|---|---|
| ours (E-SINDy + test) | **75.9** | **83.3** | 95.4 | 42.1 | 79.2 | 65.3 |
| whitening + single ARD | 44.9 | 63.4 | 95.4 | 96.8 | **99.5** | 47.7 |
| whitening + ensemble ARD (`esindy_wb`) | 48.1 | 69.0 | **95.8** | 97.2 | **99.5** | 53.7 |
| unified Bayesian selection (shared g, Gibbs) | 15.7 | 42.6 | 64.4 | **100** | 63.9 | **97.7** |
| same + non-local prior | **78.7** | 76.4 | 76.9 | 35.6 | 25.5 | 93.5 |

In theory, the threshold test, ARD and E-SINDy's inclusion frequency all approximate the same Bayesian model selection. In practice, no single version worked on both kinds of library:

- **Polynomial systems** need a minimum-effect constraint to block small spurious terms created by the noise approximation.
- **Collinear libraries** need that constraint relaxed, because the true coefficients themselves are unstable.

## 10. Attenuation bias

Noisy features also shrink the coefficients. In the point setting, with 100 random points and a whitened fit of the true structure, the mean ratio of estimated to true coefficient is:

| Noise | Mean ratio |
|---|---|
| 5% | 0.996–0.999 |
| 10% | 0.983–0.991 |
| 20% | 0.935–0.959 |

This does not cause structural errors. Second-order debiasing brings the ratio back to 1.00–1.01, but increases the variance, so the coefficient error does not improve. It is not part of the method.

# Method: the equation proposes the next experiment

## Problem

A dynamical model is written using a small fixed library:

$$\dot u=F_0(u,e)+\sum_j\xi_j\Theta_j(u,e).$$

The support $S=\{j:\xi_j\ne0\}$ and active coefficients are unknown. Controlled initial conditions or inputs may change; the physical coefficients are shared across experiments. The observation model is $y_e=\mu_e(\xi)+\epsilon$, where $\mu_e$ includes numerical integration and the actual observation operator.

A good fit under one condition need not identify the support. If the library columns are dependent on all visited trajectories, different coefficients have the same effect. More accurate fitting or repeated observations at that condition do not necessarily break the dependency.

## Shared Bayesian sparse-library engine

The main benchmark enumerates four supports for two optional terms. Each term is present with probability 0.4. Conditional on being present, its positive coefficient has a declared uniform prior. Exact zero is a separate atom. All old and new trajectories share the same coefficients.

$$p(S,\theta\mid D)\propto p(D\mid S,\theta)p(\theta\mid S)p(S).$$

The likelihood uses raw noisy observed trajectories, not estimated exact derivatives. Continuous coefficients are integrated by Gauss–Legendre quadrature; the generating truth is sampled continuously, independently of quadrature nodes. Predictions, selected support and conditional coefficient means yield executable equations.

This is inspired by SINDy's sparse-library representation and Bayesian model discovery. It is not the official STLSQ/PySINDy or a full UQ-SINDy reproduction. The separate failure controls use a short STLSQ implementation and deliberately exact derivatives; do not confuse their noiseless setup with this noisy benchmark.

## A. D-optimal: add independent coefficient information

At the current MAP support's conditional coefficient mean, calculate the observable-trajectory sensitivity $J_e=\partial\mu_e/\partial\xi$. Keep **all library columns**, including currently zero coefficients.

$$G_D=\Lambda+\sum_{r\in D}J_r^T R_r^{-1}J_r,$$
$$U_A(e)=\log\det(G_D+J_e^T R_e^{-1}J_e)-\log\det G_D.$$

Here $\Lambda$ uses the declared prior's second-moment approximation. It is not extra measured information or a proof of identifiability. In nonlinear dynamics this is a local plug-in approximation. The benchmark uses centered parameter finite differences for $J$ and separately audits numerical sensitivity. It does not require JAX or automatic differentiation.

Select the largest score among the permitted conditions. Only then obtain a new observation from the independent simulated system.

## B. Structure risk: reduce the chance of choosing the wrong term set

Under zero-one structure loss,

$$R(D)=1-\max_S p(S\mid D),$$
$$U_B(e)=R(D)-\mathbb E_{Y_e\mid D}R(D\cup\{e,Y_e\}).$$

The predictive expectation integrates structure, continuous coefficients and observation noise. Implementation uses stratified Sobol draws and logs approximation diagnostics. Finite predictive integration is not exact optimization; audit files include action changes on refinement.

The information goals are not identical. Full parameter information decomposes into structure information and within-structure coefficient information. D-optimality may therefore value an experiment that refines coefficients even when the support is settled. Conversely, a structure-risk objective may sacrifice some coefficient precision.

## Identical stopping for both

Report a single structure when its maximum posterior probability is at least 0.95. Otherwise continue until three added experiments and mark the task unresolved if the threshold remains unmet. There is **no low-one-step-value early-stop rule**: it could overlook complementary experiments.

For fair benchmark comparisons full three-run paths are archived; early-stop metrics use only the first threshold crossing and prior observations. This is not a universal error guarantee: model misspecification, near-zero mechanisms excluded by the prior, or an incorrect covariance can invalidate posterior interpretation.

## Why changing conditions helps

In a linearized parameterization,

$$\ker\!\left(\sum_e J_e^T R_e^{-1}J_e\right)=\bigcap_e\ker J_e.$$

Different experiments can remove parameter directions invisible to previous experiments. This is a local linear-algebra identity, not a global nonlinear support-recovery theorem. An experiment need not immediately falsify a model to be useful: it can first reduce the freedom that model has to fit later experiments.

## Classical references—not new claims

- Brunton, Proctor & Kutz, [SINDy](https://doi.org/10.1073/pnas.1517384113), PNAS, 2016.
- Fasel et al., [Ensemble-SINDy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006119/), 2022: ensembles and active learning already exist.
- Hirsh, Barajas-Solano & Kutz, [Sparsifying priors for Bayesian uncertainty quantification in model discovery](https://pmc.ncbi.nlm.nih.gov/articles/PMC8864363/), 2022.
- Huan & Marzouk, [Simulation-based optimal Bayesian experimental design for nonlinear systems](https://arxiv.org/abs/1108.4146), JCP, 2013.
- Hainy et al., [Optimal Bayesian design for model discrimination via classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC8924111/), Statistics and Computing, 2022. Relevant decision objective; this repository does not implement their classification surrogate.
- Seeger, [Bayesian inference and optimal design for the sparse linear model](https://www.jmlr.org/papers/v9/seeger08a.html), JMLR, 2008.

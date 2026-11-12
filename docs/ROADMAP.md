# Future work

The present release demonstrates a reproducible closed loop on small, known candidate libraries. The following are open development goals, **not completed capabilities**.

## 1. Three or more competing mechanisms

Move beyond two optional terms to competing reaction orders, nonlinear damping mechanisms, and larger PDE libraries. Preserve alternative supports until the data actually exclude them. Compare support recovery, parameter errors, holdout prediction, unresolved fractions and computational cost under a matched experiment budget.

An archived three-term pilot already shows the difficulty: for `x'=-a x-b x²-c x³`, aggregate full-budget support recovery was approximately 81.77%/82.29% for A/B, well below the easy two-term cases. This is preliminary evidence motivating this work, not proof of scalability. The original pilot code and results remain under `benchmark/stress/reaction3/`.

## 2. Trustworthy noise and model checks

Learn temporally correlated and heteroscedastic observation noise instead of supplying its true parameters. Separate uncertainty due to noise from uncertainty due to missing physics. Assess model adequacy before interpreting a concentrated support posterior as scientific confirmation.

Archived related diagnostics used known/oracle AR(1) covariance; that does **not** solve noise learning. A missing cubic reaction term can still cause confident selection of an incorrect in-library model. Preserve such failures and evaluate calibration and selective accuracy together, not confidence alone.

## 3. Validation and practical costs

Compare official PySINDy/STLSQ, PDE-FIND, UQ-SINDy and PySR pipelines with documented settings. Validate on independent experimental campaigns. Account for preparation time, reagents, reset reproducibility, input energy and measurement constraints—not only a count of runs. Jointly valuable experiments and stopping decisions also need dedicated evaluation.

No top-conference readiness, universal superiority, or industrial deployment is claimed by this roadmap.

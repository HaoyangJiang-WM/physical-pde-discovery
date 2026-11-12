# Provenance and publication

This repository publishes **Active Equation Discovery**, replacing the previous heat-transfer demonstration at the owner's explicit request. The previous content remains recoverable in Git history; no history was force-rewritten. The original main commit was `2d0719f11b37063f874aa0ca5a9dde7cdeed54ab`.

## Original research and this publication

The source research artifact was `active_discovery_two_schemes_v21.zip`, SHA-256 `c73dee6144d96b2c65b8a06905b4e18b4f4262afb8a06a999573a1cddab77321`. The readable repository was subsequently packaged with English/Chinese READMEs and five numerical GIFs.

The publication transfers the frozen source, fixed plans, protocols, documentation and main reference tables. The GitHub runner **regenerates** task logs, example observations, numerical audits and graphics from the recorded seeds and protocols. This is not a byte-for-byte upload of the original numerical archive, not a new independent validation population, and not hardware data. Execution timings, environment information and byte hashes can differ. Original main summary tables are retained as `artifacts/original_*.csv`. Publication requires the reconstructed task count and main support accuracies to match the supplied reference tables.

`benchmark/REPLICATION_LOCK.json` retains the original frozen source/plan checks. `benchmark/ARTIFACT_SHA256.json` hashes the **regenerated publication files**, with transparent decompression for CSVs. It is an integrity manifest for this release, not a claim that all original output bytes are unchanged. `artifacts/publication.json` records the workflow and reconstruction scope.

## Animation rules

Each old model keeps its coefficients frozen. All model curves extend to t=3; real simulated observations stop at t=1.5. Changing the experimental condition is not extending the measurement window. Historical curves remain visible. Scores and coefficients are read from task logs. The selected successful examples illustrate mechanisms and are not estimates of average performance.

## Scope and validation

The 4,224 primary worlds, 192 three-mechanism worlds, and 672 stress worlds are separate task populations. Noise-aware runs reuse 192 stress worlds and do not count as new independent tasks. No official PySINDy/UQ-SINDy reproduction, company-internal data, real chemical experiment, universal identifiability proof, or new D-optimal/Bayesian design criterion is claimed.

The three noiseless STLSQ controls isolate lack of experimental information, rather than numerical differentiation failure. They are not part of the main benchmark counts. Unit tests, source-lock checks, reconstructed statistics, numerical audits, replay consistency and local-link/image checks are run during publication.

No font files, credentials, emails or other private documents are included. An open-source license has not been chosen on the owner's behalf.

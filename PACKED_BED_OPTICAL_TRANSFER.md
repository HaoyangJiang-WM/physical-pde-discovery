# Can two measured temperature changes calibrate a transferable PDE boundary?

Nine packed-bed cooling runs pair a processed infrared surface average with contact thermocouples at several heights. The optical camera observes the exposed particles. The highest thermocouples sit at a nominal 98 mm inside the bed and measure a mixture of particle and gas temperatures. This comparison tests whether a scalar correction fitted on one run transfers to the contact coordinate and improves an interior PDE rollout on another run. [Data v2](https://doi.org/10.17632/3pp86gdvh4.2) · [Apparatus](https://doi.org/10.1016/j.dib.2025.111743)

## Protocol

- Check the input file hashes. Match the camera acquisition timestamp in each row to the clock workbook. Compute frame times from the numeric suffix in the original frame name and the cadence encoded in the acquisition name. Discard one foreign row from run 2 and three from run 9. Run 7's file records every other frame of a 30-second acquisition. Its effective sample spacing is 60 seconds.
- Align both channels to heater shutoff and use minutes 10 through 50. Read optical endpoints on a five-minute grid in one eligible same-flow training repeat. Enumerate the 16 possible pairs of ten-minute changes that use four distinct endpoints. The fixed pair uses minutes 10–20 and 35–45. The optical-only spread rule maximizes the absolute determinant of the two basis differences. This selection uses the calibration run's optical trace and chooses both pairs in advance.
- Use the two measured contact differences to estimate the derivative of a monotone quadratic map. Its slopes at 0 and 500 °C are bounded between .05 and 3. Estimate the intercept from the same four absolute readings. These absolute readings are needed to predict levels. The offset-only comparator uses the same four readings as the fixed quadratic fit. The random comparator takes the median error over all 16 schedules for each training/test combination.
- Fit the centered diffusion–advection–cooling PDE on the contact records of the other same-flow homogeneous repeats. Hold the coefficients fixed across boundary choices. Initialize with the held-out interior state at minute 10 and supply the observed lower boundary. Substitute calibrated or native IR for the upper boundary and score against the independent held-out interior measurements. Also replay with the contact upper boundary. Use every eligible calibration repeat, summarize each test run by its median across those repeats, and give every test run equal weight in the group median.

## Results

Nominal clock alignment. Each cell gives upper-boundary RMSE / interior rollout RMSE, in K.

| Upper boundary | Six homogeneous repeats | Three changed-setup runs |
| --- | ---: | ---: |
| Native IR | 10.10 / 10.44 | 4.80 / 12.95 |
| Fixed offset only | 14.19 / 10.42 | 3.63 / 13.31 |
| Fixed two changes and four level reads | 6.88 / 7.84 | 11.86 / 10.44 |
| Optical spread, same budget | 6.60 / 7.58 | 10.38 / 10.81 |
| Random schedules, same budget | 7.21 / 8.08 | 11.08 / 10.96 |
| Contact replay | 0 / 7.31 | 0 / 12.23 |

The small interior advantage of the spread rule over the fixed schedule on repeats reverses on changed setups at the nominal alignment. The changed-setup runs also show that a smaller rollout error can coincide with a less accurate boundary. The centered model's coefficient on the upper boundary at the nearest interior height is $(2D-v h_L)/(h_R(h_L+h_R))$. It is negative in all nine fits, between −.00314 and −.00116 per second. Under the advection-only upwind stencil, the upper boundary has zero weight by construction. Differences between the native and calibrated boundaries can therefore change the centered rollout through a discretization-sensitive pathway. The independent contact boundary error measures the calibration directly.

A common clock shift of ±60 s changes the absolute error and some rankings. For example, at −60 s, on changed setups native IR and the spread rule both give 11.11 K interior error, while the fixed rule gives 11.52 K. The test uses measured cooldown changes after the fact and the camera's processed temperature export. The archival runs supply neither commanded additive increments nor a collocated, independently labelled physical PDE term. [Per-run replay and clock sensitivity](results/packed_bed_pde/optical_transfer_centered.json)

## Reproduce

Use the unpacked v2 dataset with the nine thermocouple CSVs, nine surface ASC exports and the starting-time workbook. Install the project requirements before running the replay.

```bash
python benchmark_packed_bed_optical.py /path/to/unpacked/dataset
```

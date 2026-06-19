# New Metrics Correctness Report

This report covers the fifteen metrics added in this change. All examples use
small deterministic arrays and are verified by tests with strict numeric
tolerances.

## Temporal Drift

Formula: Per-timestep L2 error, then least-squares slope against timestep index,
averaged across batch.
Inputs: `predicted` and `reference`, shaped `TxD` or `BxTxD`, same shape.
Output: Scalar drift rate.
Score direction: Lower absolute drift is more stable; positive means error grows.
Units: State units per timestep index.
Edge-case behavior: One timestep returns `0.0`.
Hand-calculated test: Error sequence `[0, 1, 2, 3]`.
Expected value: `1.0`.
Implementation result: `1.0`.
Monotonicity verification: `[0, 1, 2, 3]` returns `1.0`; `[3, 2, 1, 0]`
returns `-1.0`.
Invalid input verification: Mismatched shapes raise `ValueError`.
Known limitations: Uses timestep index, not wall-clock time; resample first for
irregular timestamps.

## Action Jerk

Formula: Mean squared L2 norm of `diff(actions, n=2) / dt**2`.
Inputs: `actions` shaped `TxD` or `BxTxD`; positive `dt`.
Output: Non-negative scalar penalty.
Score direction: Lower is smoother.
Units: Squared action units per `s^4`.
Edge-case behavior: Fewer than three timesteps return `0.0`.
Hand-calculated test: Actions `[0, 0, 1, 0]` have second differences `[1, -2]`.
Expected value: `(1^2 + (-2)^2) / 2 = 2.5`.
Implementation result: `2.5`.
Monotonicity verification: Constant and linear actions return `0.0`; a sudden
jump returns a positive value.
Invalid input verification: Non-positive `dt` raises `ValueError`.
Known limitations: Treats action jerk as second finite difference of controls;
it does not infer actuator semantics.

## Control Smoothness

Formula: `1 / (1 + action_jerk(actions, dt))`.
Inputs: `actions` shaped `TxD` or `BxTxD`; positive `dt`.
Output: Bounded scalar score.
Score direction: Higher is smoother.
Units: Dimensionless.
Edge-case behavior: Constant, linear, and too-short controls return `1.0`.
Hand-calculated test: Alternating actions `[1, -1, 1, -1]` have action jerk
`16`.
Expected value: `1 / 17 = 0.058823529411764705`.
Implementation result: `0.058823529411764705`.
Monotonicity verification: Constant controls score higher than alternating
controls.
Invalid input verification: Invalid `dt` raises `ValueError`.
Known limitations: The score is scale-sensitive because larger action units
produce larger finite differences.

## Recovery Success Rate

Formula: `count(opportunities & successes) / count(opportunities)`.
Inputs: Same-shaped boolean or 0/1 opportunity and success masks.
Output: Recovery rate.
Score direction: Higher is better.
Units: Ratio.
Edge-case behavior: No opportunities returns `nan` because the denominator is
undefined.
Hand-calculated test: Opportunities `[1, 1, 1, 1]`, successes `[1, 0, 1, 1]`.
Expected value: `3 / 4 = 0.75`.
Implementation result: `0.75`.
Monotonicity verification: All failures return `0.0`; all successes return
`1.0`.
Invalid input verification: Shape mismatch or non-0/1 mask values raise
`ValueError`.
Known limitations: Successes outside opportunity timesteps are ignored.

## Failure Severity

Formula: Mean severity by default, or max severity with `aggregation="max"`.
Inputs: Numeric finite non-negative severities, or known category labels.
Output: Aggregated severity.
Score direction: Lower is better.
Units: Severity units.
Edge-case behavior: Empty failure collections return `0.0`.
Hand-calculated test: Numeric severities `[1, 2, 3]`.
Expected value: Mean `2.0`; max `3.0`.
Implementation result: Mean `2.0`; max `3.0`.
Monotonicity verification: `[3, 4]` scores higher than `[1, 2]`.
Invalid input verification: Negative severities, unknown labels, and unsupported
aggregation modes raise `ValueError`.
Known limitations: Category labels use a simple ordinal default mapping unless
the caller supplies `category_scores`.

## Near-Miss Rate

Formula: `mean((clearance < threshold) & ~collision_mask)`.
Inputs: Finite clearance distances, positive threshold, optional same-shaped
collision mask.
Output: Near-miss rate.
Score direction: Lower is safer.
Units: Ratio.
Edge-case behavior: Collision samples are excluded from near-miss counts.
Hand-calculated test: Clearances `[0.2, 0.5, 1.5, 0.1]`, threshold `1.0`,
collisions `[False, True, False, False]`.
Expected value: Near misses at indices 0 and 3, so `2 / 4 = 0.5`.
Implementation result: `0.5`.
Monotonicity verification: Raising the threshold from `1.0` to `2.0` increases
the rate from `0.5` to `0.75` for the same data.
Invalid input verification: Non-positive threshold or collision-mask shape
mismatch raises `ValueError`.
Known limitations: Uses strict `< threshold`; exactly-threshold clearances are
not near misses.

## Intervention-Free Time

Formula: Consecutive non-intervention samples form segments; segment duration is
`timestamp[last_false] - timestamp[first_false]`.
Inputs: Strictly increasing 1D timestamps and same-shaped boolean/0/1
intervention mask.
Output: Longest duration by default; mean segment duration with `mode="mean"`.
Score direction: Higher is better.
Units: Timestamp units, normally seconds.
Edge-case behavior: All intervention timesteps return `0.0`.
Hand-calculated test: Timestamps `[0, 1, 2, 3, 4, 5]`, interventions
`[False, False, True, False, False, False]`.
Expected value: Longest segment from `3` to `5`, duration `2.0`; mean duration
`(1.0 + 2.0) / 2 = 1.5`.
Implementation result: Longest `2.0`; mean `1.5`.
Monotonicity verification: All-clear timestamps `[0..5]` return `5.0`, which is
higher than the interrupted case.
Invalid input verification: Unsorted timestamps, shape mismatch, and unsupported
mode raise `ValueError`.
Known limitations: Segment duration follows sample-to-sample timestamps, not
closed interval occupancy.

## Coverage Score

Formula: `unique_occupied_bins / total_possible_bins` over a configured grid.
Inputs: Samples shaped `NxD`, bounds shaped `Dx2`, scalar or per-dimension bins.
Output: Coverage score in `[0, 1]`.
Score direction: Higher is broader coverage.
Units: Dimensionless.
Edge-case behavior: Out-of-bounds samples are ignored; duplicates do not
increase coverage.
Hand-calculated test: 2D bounds `[[0, 1], [0, 1]]`, bins `[2, 2]`, samples that
occupy two unique bins.
Expected value: `2 / 4 = 0.5`.
Implementation result: `0.5`.
Monotonicity verification: A three-bin sample set scores higher than a one-bin
sample set under the same bounds and bins.
Invalid input verification: Bad sample rank, bad bounds, reversed bounds, and
non-positive bins raise `ValueError`.
Known limitations: Grid coverage is sensitive to bounds and bin count.

## Calibration Error

Formula: Expected Calibration Error over uniform bins in `[0, 1]`.
Inputs: Confidence probabilities and same-shaped correctness labels.
Output: Non-negative calibration error.
Score direction: Lower is better.
Units: Error.
Edge-case behavior: Empty inputs are invalid; empty bins are skipped.
Hand-calculated test: Confidences `[0.25, 0.75]`, correctness `[0, 1]`,
`n_bins=2`.
Expected value: `0.5 * |0 - 0.25| + 0.5 * |1 - 0.75| = 0.25`.
Implementation result: `0.25`.
Monotonicity verification: Overconfident wrong predictions score `0.85`, higher
than the calibrated prompt example score `0.15`.
Invalid input verification: Confidence values outside `[0, 1]`, non-0/1 labels,
shape mismatch, and non-positive bins raise `ValueError`.
Known limitations: ECE depends on the selected number of bins.

## Kinematic Feasibility

Formula: `1 - violating_checks / measured_checks` over finite-difference
velocity, acceleration, and optional curvature checks.
Inputs: Positions shaped `T` or `TxD`, positive `dt` or strictly increasing
timestamps, optional non-negative limits.
Output: Feasibility score in `[0, 1]`.
Score direction: Higher is more feasible.
Units: Dimensionless score; checks use `m/s`, `m/s^2`, and `1/m` when positions
are metric.
Edge-case behavior: Unmeasurable checks on too-short trajectories are skipped;
no measurable checks return `1.0`.
Hand-calculated test: Positions `[0, 1, 4]`, `dt=1`, `max_velocity=2`,
`max_acceleration=3`.
Expected value: Velocity checks are `[1, 3]` with one violation; acceleration
check is `[2]` with no violation, so `2 / 3 = 0.6666666666666666`.
Implementation result: `0.6666666666666666`.
Monotonicity verification: `[0, 1, 2]` with max velocity `2` scores `1.0`;
`[0, 3, 6]` scores `0.0`.
Invalid input verification: Non-positive `dt`, unsorted timestamps, and
curvature requested for 1D positions raise `ValueError`.
Known limitations: Uses finite differences, not a robot-specific kinematic
model.

## Dynamic Feasibility

Formula: `1 - violating_checks / measured_checks` over max-force,
max-acceleration, max-torque, and friction-cone checks.
Inputs: Positive mass, acceleration samples, optional forces/torques/friction
inputs and non-negative limits.
Output: Dynamic feasibility score in `[0, 1]`.
Score direction: Higher is more feasible.
Units: Dimensionless score; checks use Newtonian units supplied by the caller.
Edge-case behavior: No configured constraints return `1.0` after validation.
Hand-calculated test: Mass `2 kg`, acceleration `3 m/s^2`, required force `6 N`.
Expected value: Max force `10 N` scores `1.0`; max force `5 N` scores `0.0`.
Implementation result: `1.0` and `0.0`.
Monotonicity verification: A friction case with tangential force `4 N`,
`mu=0.5`, normal force `10 N` scores `1.0`; tangential force `6 N` scores
`0.0`.
Invalid input verification: Missing torque samples for `max_torque`, incomplete
friction inputs, and invalid shapes raise `ValueError`.
Known limitations: Uses simple Newtonian checks and does not require or emulate
a physics engine.

## Physics Violation Rate

Formula: Logical OR across violation types, then mean over timesteps/events.
Inputs: Single boolean/0/1 mask, `CxT` mask array, or mapping of same-shaped
masks.
Output: Violation rate in `[0, 1]`.
Score direction: Lower is better.
Units: Ratio.
Edge-case behavior: Empty mappings and empty masks are invalid.
Hand-calculated test: Velocity mask `[False, True, False, False]` and
acceleration mask `[False, False, True, False]`.
Expected value: Any violation at two of four timesteps, so `0.5`.
Implementation result: `0.5`.
Monotonicity verification: All-false returns `0.0`; all-true returns `1.0`.
Invalid input verification: Mismatched mapping mask shapes and empty mappings
raise `ValueError`.
Known limitations: Does not weight violation severity; each timestep counts once
after logical OR.

## Long-Horizon Drift

Formula: Later-weighted mean L2 error using weights `1..T`, averaged across
batch.
Inputs: `predicted` and `reference`, shaped `TxD` or `BxTxD`, same shape.
Output: Non-negative later-weighted error.
Score direction: Lower is better.
Units: State units.
Edge-case behavior: Perfect rollout returns `0.0`.
Hand-calculated test: Early error `[3, 0, 0]` and late error `[0, 0, 3]`.
Expected value: Early `3 / 6 = 0.5`; late `9 / 6 = 1.5`.
Implementation result: Early `0.5`; late `1.5`.
Monotonicity verification: The late-error rollout scores worse than the
early-error rollout with the same mean error.
Invalid input verification: Mismatched shapes raise `ValueError`.
Known limitations: Uses linear horizon weights only.

## Compounding Error Index

Formula: `max(0, final_error - initial_error) / (mean_error + epsilon)`,
averaged across batch.
Inputs: Error curves shaped `T` or `BxT`, or predicted/reference states shaped
`TxD` or `BxTxD`.
Output: Non-negative growth index.
Score direction: Lower is better.
Units: Dimensionless.
Edge-case behavior: Perfect and constant error curves return `0.0`.
Hand-calculated test: Error curve `[1, 2, 3, 4]`.
Expected value: `(4 - 1) / 2.5 = 1.2`.
Implementation result: `1.2`.
Monotonicity verification: Growing error scores higher than constant error.
Invalid input verification: Negative explicit errors and non-positive epsilon
raise `ValueError`.
Known limitations: Measures first-to-final growth, not oscillatory error
patterns.

## Behavioral Diversity

Formula: Mean pairwise Euclidean distance between unique flattened behaviors.
Inputs: Behavior embeddings shaped `NxD`, or trajectories/actions shaped
`NxTxD`.
Output: Non-negative diversity score.
Score direction: Higher is more diverse.
Units: Behavior-vector distance units, or normalized distance if requested.
Edge-case behavior: Fewer than two unique behaviors return `0.0`; duplicates
are removed before pairwise distance calculation.
Hand-calculated test: Embeddings `[0, 0]`, `[3, 4]`, `[6, 8]`.
Expected value: Pair distances `5`, `10`, and `5`; mean `20 / 3`.
Implementation result: `6.666666666666667`.
Monotonicity verification: Farther embeddings score higher than closer
embeddings; identical trajectories score `0.0`.
Invalid input verification: Bad rank, empty input, and non-positive `max_pairs`
raise `ValueError`.
Known limitations: Deterministic pair sampling limits cost for large `N`, but
the score is still an embedding-distance summary rather than a behavior model.

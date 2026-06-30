# Research protocol

`config/replication-protocol.json` is the machine-readable specification for the main empirical
study. `config/historical-protocol.json` records the earlier June–August 2025 analysis. The JSON
files take precedence over this summary and remain byte-for-byte tied to their saved results.

## Question and evaluation unit

The study asks whether adding Kalshi probabilities improves a public weather probability
forecast on later dates, and whether contemporaneous disagreement identifies when that
contribution is useful. An evaluation case is one Central Park daily-high market at either the
12-hour or 6-hour checkpoint. Its mutually exclusive settlement bins are scored jointly.
Repeated prices and bins do not create additional outcomes.

The primary comparison is fitted blend minus public-only multiclass Brier loss, reported
separately for each checkpoint. Market-only, equal blend, and the pooled residual baseline are
reported on the same cases. Secondary results include log loss, calibration, and the direct
high-minus-low disagreement interaction. Negative paired differences favor the fitted blend.

## Data and timing

The replication requests June 1, 2025 through June 30, 2026. Development training ends December
31, 2025, model-selection validation ends March 31, 2026, and April–June 2026 forms the holdout.
Every training label must be available by the relevant fit cutoff. A date's calendar split does
not override its actual label availability.

The outcome is the Central Park settlement-day high over 05:00–05:00 UTC. The archived NDFD
MaxT predictor covers 12:00–00:00 UTC. The study therefore models the residual `settlement high
- NDFD MaxT` rather than treating the two windows as equal. Market probabilities use normalized
hourly bid and ask candle closes. Missing bins, stale observations, incompatible rules, or unknown
label timing exclude a case; missing bins are never assigned zero probability.

Historical reconstruction uses retained rule bytes and separates forecast issue, validity,
availability, market observation, receipt, outcome, label, and settlement times. These records
support the saved reconstruction but do not prove contemporaneous receipt or absence of later
archive revisions. See [sources and data boundaries](SOURCES.md).

## Forecast models

The pooled residual baseline fits one Normal residual distribution. The seasonal ridge residual
model adds an intercept, a 6-hour indicator, training-standardized MaxT, and one annual sine and
cosine pair. Its phase is `2*pi*(day_of_year-1)/365.25`; ridge penalties are 0.1, 1, and 10, with
the intercept unpenalized. All standardization and residual estimates use eligible earlier labels
only.

Monthly expanding folds start in July 2025. A prediction month uses data available by 17:00 UTC
on the preceding day. Horizon-specific predictive scales are the root mean square of earlier
out-of-fold errors, floored at 1°F, after at least 20 original error dates. Bin probabilities
integrate the fitted Normal distribution over inclusive integer boundaries using half-degree
edges.

Candidate parameters and scales are fixed at December 31, 2025 17:00 UTC. Candidate selection
uses identical validation cases whose labels are available by March 31, 2026 17:00 UTC. Ties
within `1e-12` favor the pooled model, then the larger ridge penalty. The selected model is refit
on eligible development labels at that final cutoff and remains fixed for the holdout point
predictions.

One convex market weight is learned from August 2025 through March 2026 chronological
out-of-fold probabilities. The same cases determine horizon-specific L1 disagreement medians.
Neither procedure uses in-sample public probabilities. In the saved study the selected model is
the seasonal ridge residual model with penalty 1, and the fitted market weight is 1.

## Uncertainty and conditions

The primary bootstrap makes 5,000 attempts with seed 20260914. It samples nonwrapping consecutive
calendar blocks within each split and calendar month, preserving each sampled date's multiplicity
across horizons and bins. Every full-refit draw repeats standardization, monthly folds, scale
estimation, candidate selection, final fitting, blend fitting, and disagreement thresholds.
Failed draws are counted and not replaced; intervals are withheld below 95% valid attempts.

The primary block length is seven days. Three-day and fourteen-day full-refit calculations and a
seven-day fixed-fit calculation are sensitivity checks. Central 95% intervals are reported, with
97.5% marginal intervals for the two primary checkpoint comparisons as an approximate
Bonferroni adjustment.

Month stratification holds observed month composition fixed and cuts dependence at month
boundaries. Nonwrapping blocks underweight month-edge dates, especially at fourteen days. Few
blocks, nonstationarity, missing seasonal labels, and model selection make coverage approximate;
no nominal-coverage theorem is claimed.

High disagreement means strictly above the earlier out-of-fold median. A condition interval
requires both groups to contain at least 20 dates and five occupied, nonoverlapping seven-day
calendar blocks. These guards limit unsupported estimates but do not establish statistical
power. A significant result in one subgroup and an insignificant result in another is not itself
evidence that the subgroups differ; the direct interaction is the relevant contrast.

## Sensitivities and interpretation

The development-label sensitivity substitutes the June 2 NCEI value for the reported settlement
temperature without changing the winning bin. The normalization sensitivity projects raw candle
midpoints onto the probability simplex. Both are point sensitivities and do not replace the main
analysis.

All forecast methods are compared on identical retained cases. Missing numeric labels are not
recovered by the bootstrap. Candle probabilities do not establish executable fills, depth, fees,
or profitability. A lower market loss on this cohort does not establish general market
superiority, and an interval containing zero does not establish equivalence.

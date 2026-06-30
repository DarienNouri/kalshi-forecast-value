# Kalshi information quality lab

Do prediction-market prices add information beyond public weather forecasts? This report compares archived Kalshi New York daily-high markets with archived NOAA NDFD forecasts at 12 and 6 hours before the settlement day begins.

## Data and methods

The outcome is the Central Park settlement-day high, divided into market bins. Market probabilities use normalized hourly bid/ask candle closes, which are historical proxies rather than executable prices. The public predictor is the NOAA NDFD MaxT grid forecast, calibrated to settlement bins using earlier labels.

The multiclass Brier score for one case sums squared errors across settlement bins,
with no division by the bin count:

$$BS_i = \sum_{k=1}^{K} \left(p_{i,k} - \mathbb{1}[k = k_i^{*}]\right)^2$$

Here $p_{i,k}$ is the forecast probability assigned to bin $k$ for case $i$, $k_i^{*}$ is the
realized settlement bin, and $K$ is the number of bins in that contract. Reported scores average
$BS_i$ over the $N$ cases in a split:

$$\overline{BS} = \frac{1}{N}\sum_{i=1}^{N} BS_i$$

Log loss floors the winning-bin probability at $\epsilon = 10^{-6}$ and does not renormalize the
probability vector:

$$LL_i = -\log\left(\max(p_{i,k_i^{*}},\ \epsilon)\right), \qquad \overline{LL} = \frac{1}{N}\sum_{i=1}^{N} LL_i$$

The fitted blend is a fixed convex combination of the market and public probability vectors,
with the weight fit out-of-fold on pre-holdout dates and then held fixed for scoring:

$$p_{\text{blend}} = w \, p_{\text{market}} + (1 - w) \, p_{\text{public}}, \qquad w \in [0, 1]$$

The study covers 2025-06-01 through 2026-06-30. Of 395 requested dates, 326 have usable numeric settlement labels and enter the cohort; 69 are excluded. The retained cohort has 326 dates and 652 scored checkpoint cases across the two horizons.

## Main finding

In this retained sample, the market had lower average Brier loss than the public forecast at each evaluated horizon. The fitted blend weight was 1.0000 (trained blend equals market only; no mixing benefit shown). The disagreement analysis did not identify in advance when the market's advantage would be larger (the 97.5% interval included zero or group support was too small).

## Holdout scores

Mean multiclass Brier loss on the same held-out cases; lower is better.

| checkpoint | public only | market only | equal blend | trained blend | pooled residual baseline | dates | cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12h | 0.7678 | 0.6939 | 0.7116 | 0.6939 | 0.7759 | 90 | 90 |
| 6h | 0.7617 | 0.6815 | 0.6950 | 0.6815 | 0.7728 | 90 | 90 |

![average multiclass brier loss by method and horizon on the same holdout dates; lower is better.](assets/holdout-losses.png)

Model selection chose the seasonal ridge residual model, λ = 1. Out-of-fold error folds run from 2025-07-01 through 2026-03-01. The blend weight was fit on 348 eligible checkpoint predictions across 174 dates, 2025-08-01 through 2026-03-30. Its market weight is 1.0000: trained blend equals market only; no mixing benefit shown.

## Uncertainty

Fitted blend minus public forecast Brier loss; negative values favor the blend. The nominal 97.5% intervals use a Bonferroni adjustment for the two horizon comparisons.

| checkpoint | fit treatment | difference | 95% interval | 97.5% interval | valid draws | interval status |
| --- | --- | --- | --- | --- | --- | --- |
| 12h | full refit, 7-day blocks | -0.0739 | [-0.1049, -0.0256] | [-0.1103, -0.0191] | 5000 / 5000 | available |
| 6h | full refit, 7-day blocks | -0.0802 | [-0.1182, -0.0152] | [-0.1262, -0.0074] | 5000 / 5000 | available |
| 12h | full refit, 3-day blocks | -0.0739 | [-0.1150, -0.0222] | [-0.1214, -0.0161] | 5000 / 5000 | available |
| 6h | full refit, 3-day blocks | -0.0802 | [-0.1312, -0.0166] | [-0.1398, -0.0092] | 5000 / 5000 | available |
| 12h | full refit, 14-day blocks | -0.0739 | [-0.0961, -0.0371] | [-0.0996, -0.0330] | 5000 / 5000 | available |
| 6h | full refit, 14-day blocks | -0.0802 | [-0.1059, -0.0287] | [-0.1120, -0.0236] | 5000 / 5000 | available |
| 12h | fixed fit, 7-day blocks | -0.0739 | [-0.1030, -0.0326] | [-0.1078, -0.0274] | 5000 / 5000 | available |
| 6h | fixed fit, 7-day blocks | -0.0802 | [-0.1165, -0.0270] | [-0.1230, -0.0217] | 5000 / 5000 | available |

![trained blend minus public-only brier loss, with 95% and nominal 97.5% full-refit intervals using seven-day blocks; negative values favor the blend.](assets/paired-differences.png)

Each full-refit draw repeats model selection, calibration, blend fitting, and condition assignment. Fixed-fit draws hold the fitted pipeline constant. Blocks stay within calendar months; this underweights dates near month edges, especially with 14-day blocks.

## Sensitivities

These point sensitivities have no separate bootstrap intervals.

| sensitivity | checkpoint | difference from primary |
| --- | --- | --- |
| development-label substitution | 12h | 0.0000 |
| development-label substitution | 6h | 0.0000 |
| simplex normalization | 12h | -0.0020 |
| simplex normalization | 6h | 0.0003 |

## Can disagreement identify the market's advantage?

**12h disagreement subgroup.** high disagreement: 68 dates in 13 calendar blocks. low disagreement: 22 dates in 9 calendar blocks. The saved support thresholds were met. The direct high-minus-low interaction is -0.0551, 95% interval [-0.1582, -0.0047] (excludes zero), 97.5% interval [-0.1677, 0.0052] (includes zero).

**6h disagreement subgroup.** high disagreement: 73 dates in 13 calendar blocks. low disagreement: 17 dates in 8 calendar blocks. Interval unavailable: too few dates in one disagreement group

These subgroup interactions are exploratory and do not establish a reliable pre-outcome selection rule.

## Data coverage

Missing numeric settlement labels are concentrated in autumn and winter. June 23, 2026 is also absent from the holdout for this reason. This selection may bias seasonal and holdout comparisons.

Excluded dates lacked usable numeric settlement labels.

| month | requested | usable | excluded |
| --- | --- | --- | --- |
| 2025-06 | 30 | 30 | 0 |
| 2025-07 | 31 | 31 | 0 |
| 2025-08 | 31 | 31 | 0 |
| 2025-09 | 30 | 30 | 0 |
| 2025-10 | 31 | 30 | 1 |
| 2025-11 | 30 | 4 | 26 |
| 2025-12 | 31 | 2 | 29 |
| 2026-01 | 31 | 19 | 12 |
| 2026-02 | 28 | 28 | 0 |
| 2026-03 | 31 | 31 | 0 |
| 2026-04 | 30 | 30 | 0 |
| 2026-05 | 31 | 31 | 0 |
| 2026-06 | 30 | 29 | 1 |

![usable and excluded dates by month for the historical study.](assets/monthly-coverage.png)

## Limitations

NDFD MaxT covers 12:00–00:00 UTC, while the Central Park settlement day runs from 05:00 UTC to 05:00 UTC. Calibration does not recover the missing overnight information.

The bootstrap repeats model selection and fitting but cannot recover missing labels or unknown archive revisions. Month-stratified, nonwrapping blocks also underweight dates near month boundaries.

Hourly bid/ask candle closes do not establish fills, depth, fees, or profitability.

## Reproducibility

This page is built from the retained empirical result. The [evidence manifest](portfolio-evidence.json) hashes the source result, scientific inputs, report, page, and exact aggregate data behind each figure.

The [replication notebook](../notebooks/06_historical_replication.ipynb), [historical study notebook](../notebooks/05_historical_study.ipynb), [NDFD extraction notebook](../notebooks/04_noaa_grid_validation.ipynb), source code, and replay scripts are tracked here. Raw inputs and fitted runs stay outside Git; see [sources and data boundaries](../docs/SOURCES.md).

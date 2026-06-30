# Sources and data boundaries

This document records the sources that define the study, how retained responses are replayed,
and the limits on interpreting or redistributing them.

## Kalshi

| Source | Use in the study |
| --- | --- |
| [Historical data guide](https://docs.kalshi.com/getting_started/historical_data) | Live and historical routing and archive cutoffs |
| [Market schema](https://docs.kalshi.com/api-reference/market/get-market) and [series schema](https://docs.kalshi.com/api-reference/market/get-series) | Identifiers, rules, settlement sources, and contract fields |
| [Live candles](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) and [historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks) | Hourly bid, ask, trade, and quantity fields |
| [Weather-market guidance](https://help.kalshi.com/en/articles/13823837-weather-markets) | Climate-report and local-standard-time settlement conventions |
| [Fixed-point guide](https://docs.kalshi.com/getting_started/fixed_point_migration) | Decimal price fields |
| [Data terms](https://kalshi-public-docs.s3.amazonaws.com/kalshi-data-terms-of-service.pdf) and [developer agreement](https://assets.kalshi.com/Kalshi-Developer-Agreement.pdf) | Conditions on access, storage, use, and publication |

The study preserves the exact market-level rule records it inspected. Current series pages do
not establish the historical rule text or archive completeness. The tracked repository excludes
the raw response archive, credentials, authentication headers, and account records. Permission
to redistribute market data or market-derived research artifacts has not been established.
Anyone redistributing code, notebooks, figures, findings, or data artifacts must review the
current provider terms for that use; the project license does not grant rights to third-party
data.

## NOAA and NWS

| Source | Use in the study |
| --- | --- |
| [NCEI NDFD archive](https://www.ncei.noaa.gov/products/weather-climate-models/national-digital-forecast-database) | Archived forecast source |
| [NOAA NDFD registry on AWS](https://registry.opendata.aws/noaa-ndfd/) | Object organization, update cadence, and scan-pattern warning |
| [NDFD metadata](https://www.weather.gov/gis/ndfd_metadata.html) | MaxT definition and 12-hour daytime validity window |
| [NWS API documentation](https://www.weather.gov/documentation/services-web-api) | Point-to-grid concepts and current forecast interface |
| [NCEI daily summaries](https://www.ncei.noaa.gov/access/search/data-search/daily-summaries) | Independent settlement-temperature reconciliation |

The archive route does not guarantee that every selected vintage exists or prove when a file
first became public. Each retained object is identified by its key, source timestamps, and
SHA-256 hash.

## Historical reconstruction

Forecast issue time, valid interval, reconstructed availability, market observation time, local
receipt time, outcome window, label availability, and settlement time remain separate fields.
NOAA object modification times, bulletin times plus the declared delay, decoded reference times,
and candle interval endpoints bound availability under the saved policy. They do not prove
contemporaneous receipt or rule out later archive revision.

NDFD MaxT covers a daytime interval rather than the full settlement day. It is therefore used as
a predictor, with residual calibration based only on earlier available labels. Current metadata
is never substituted for a retained historical market definition. An NCEI observation is used
only in the explicitly labeled label sensitivity and does not replace the primary settlement.

## Replay and retained inputs

Raw responses and fitted run directories are intentionally outside Git. A full replay requires
the retained caches under `outputs/historical-study/` and `outputs/historical-replication/`.
Cache manifests bind raw bytes to content hashes, and the replication result binds its panel and
protocol by semantic SHA-256. Missing or changed bytes fail validation. A newly downloaded
response belongs in a new versioned run rather than overwriting the frozen input.

The portable [evidence manifest](../reports/portfolio-evidence.json) records the empirical result
hash, scientific input hashes, rendered report hashes, and exact aggregate data behind each
figure. `scripts/verify_portfolio.py` checks those identities when the private source result is
available and still checks the portable report and figure integrity when it is not.

## Methods references

| Reference | Role |
| --- | --- |
| [Gneiting and Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | Proper probability scoring |
| [Paparoditis and Politis, 2002](https://numdam.org/articles/10.1016/S1631-073X(02)02578-5/) | Nonstationary time-series resampling context; the month-stratified bootstrap does not claim this paper's coverage guarantee |
| [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Descriptive reliability terminology |

These references motivate the metrics and dependence checks. They do not validate this study's
finite-sample interval coverage or establish performance outside the retained cohort.

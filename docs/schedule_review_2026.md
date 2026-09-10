# Schedule-frequency audit, 2026-09-06

## Scope and question

The committed study covers 7,300 trading days from 1996-01-02 to 2024-12-31,
using the legacy CRSP tape and the repository's documented point-in-time
constituent proxy. This audit reaggregates 140 existing variant rows. It does
not rerun strategies, select another rule or access a new holdout.

The diagnostic was chosen after inspecting the published schedule table:
does its pooled dispersion isolate the choice of calendar anchor? For each
of five SMA lengths, retain the existing one daily, five weekly and 21 monthly
anchors. Exclude the separate semi-monthly headline from this decomposition.
Report within-frequency ranges and partition the sum of squared CAGR deviations
into within- and between-frequency components. The identity is descriptive,
weighted by the existing number of anchors. It is neither a causal attribution
nor an ANOVA significance test: these portfolios share returns and exposures.

## Result

For SMA-200 the pooled range is 3.86415 percentage points, versus 0.71593 within
weekly and 1.34535 within monthly schedules. Between-frequency differences
account for 79.5327% of the squared dispersion; the five-length range is
79.5327% to 85.3024%. Thus the interpretation as pure calendar timing fails.
This does not rescue the strategy: the semi-monthly SMA-200 CAGR remains
8.395669%, and none of the five best-anchor CAGRs beats the index benchmark.

`python run_schedule_audit.py --check` reproduces all five original headline
summary rows from `reports/timing_luck_variants.csv`, then checks the 15-row
`timing_luck_by_frequency.csv` and five-row
`timing_luck_frequency_decomposition.csv`. No raw CRSP records are exported.
The full sweep remains exploratory; this decomposition supplies no new
multiple-testing-adjusted claim of positive performance.

## Primary literature and implementation sources

- [Hoffstein, Sibears and Faber, Rebalance Timing Luck](https://www.thinknewfound.com/rebalance-timing-luck):
  the authors' research page defines timing luck using identically managed
  portfolios rebalanced on different dates, and reports a fixed-mix example
  and mitigation through equal exposure to subindices. Its accessible abstract
  supplies no exact empirical date span, so none is inferred here. It concerns
  portfolio rebalance-date dispersion, not an intraday forecast horizon.
  Changing daily to monthly signal evaluation changes more than the anchor;
  this is the reason for the separate frequency tables.
- [Yin, Miki, Lesnichenko and Gural (2026), Implementation Risk in Portfolio Backtesting](https://arxiv.org/html/2603.20319v1):
  15 strategies, five retained engines, four cost regimes and 30 six-stock
  buckets drawn from 180 S&P 500 stocks. Daily adjusted-close data, 2018-2019
  warm-up and 2020-2024 evaluation (1,258 trading days). The authors report
  agreement at zero cost and material divergence for high-turnover strategies
  under costs. Their fixed complete-history universe does not establish
  survivorship freedom, despite that claim in Section 4.2. Their engineering
  comparison motivates explicit accounting conventions, not a replacement for
  this repository's constituent history or an expected improvement in returns.
- [WRDS Python client source](https://github.com/wharton/wrds/blob/main/wrds/sql.py):
  implementation reference, no empirical sample or forecast horizon. The
  installed client's automatic connection routine can retry and prompt after
  failure. The shared loader now checks `WRDS_DUO_READY=1` before optional
  imports, constructs a disconnected client and opens at most one SQLAlchemy
  connection without pooling. Offline mocks verify no prompt, retry or driver
  payload disclosure. This audit made no WRDS connection.

Sources checked through 2026-09-06. This is a targeted update for the audit,
not an exhaustive new review of all trend-following literature. A useful next
experiment is a predeclared staggered portfolio at a fixed frequency, evaluated
on a future period with unchanged costs and universe rules. The existing
sample should not be reused to choose its allocation.

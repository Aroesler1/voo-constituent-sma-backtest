# Audit history

Earlier results and the corrections that changed them, kept for the record. The current figures
are in [methodology.md](methodology.md). Where a number below is not in a committed file, it came
from a local `output/` run that needs a WRDS entitlement to regenerate (`output/` is gitignored).

## Correction, 2026-10-23: rerun after the code review

A code review found bugs that changed published results. All were fixed, each with a regression
test in `tests/test_results_fixes.py` that fails on the earlier code, and every study was rerun
offline from the same local CRSP cache.

| fix | what was wrong |
|---|---|
| SMA warm-up | the panel started on 1996-01-02 with no earlier prices, so every rule sat in cash for its first L sessions (the 200-day rule until 1996-10-15) while SPY was invested; signals now use prices from 1993 |
| rebalance band | the 25 bp no-trade band was absolute and also blocked full entries and exits: with more than 400 active names nothing new was bought, and a held name under 0.25% was never sold when its signal turned off |
| band renormalisation | when weights summed above 100%, every holding was rescaled without being charged; now only the names that trade are scaled and every executed change is charged |
| membership by PERMNO | members listed under a later ticker (WBA for WAG, HPQ for HWP) or a hyphenated class (BRK-B) never matched CRSP and were dropped; intervals are now resolved to PERMNOs from CRSP's name history |
| holiday membership events | adds and removes dated on a weekend or market holiday were silently dropped, so removed names stayed members |
| $5 filter and ADV | both used the total-return price level, not the traded price |
| price-path gap reset | a trading gap of more than 7 days restarted the total-return path at the raw price, creating a fake jump after splits |
| VOO proxy | VOO returns were computed before cutting at the ETF's inception, splicing in the 1998-2003 owner of the ticker; the buy-and-hold column flagged below as implausible (2.86%) now reads 10.01% |
| held-out control metrics | the first held-out day's return was left out of CAGR and drawdown |
| smaller metric fixes | Sortino used the dispersion of negative returns rather than downside deviation; `run_pbo.py` printed a Deflated Sharpe on raw returns with 5 trials instead of `main.py`'s excess returns and 9 trials; DGS3MO accrued on ACT/360 instead of 365; the contamination audit's clean signal spanned PERMNO changes |

The fixes were applied together, so the changes below are not attributed one by one.

| result | before | after | file |
|---|---|---|---|
| strategy CAGR, legacy / CIZ | 8.280% / 8.298% | 8.848% / 8.879% | `tape_comparison_corrected.csv` |
| CIZ minus legacy strategy CAGR | 1.81 bps | 3.04 bps | same |
| strategy Sharpe, legacy | 0.373 | 0.419 | same |
| strategy max drawdown | 54.4% | 52.1% | same |
| annual turnover / trades | 4.54 / 21,835 | 5.71 / 36,667 | same |
| annualised cost | 67.9 bps | 85.1 bps | same |
| comparable constituent-days | 5,424,523 | 5,425,928 | `tape_return_differences_corrected.csv` |
| ordinary-session days differing by >1 bp | 1 of 5,423,806 | 38 of 5,425,294 | `tape_return_differences_by_session_corrected.csv` |
| SMA-200 CAGR range over 27 schedules | 3.86 pp (5.73% to 9.60%) | 4.21 pp (5.87% to 10.08%) | `timing_luck_summary.csv` |
| SMA-200 between-frequency share | 79.5% | 71.7% | `timing_luck_frequency_decomposition.csv` |
| anchor variants above SPY's CAGR | 0 of 135 | 5 of 135 | `timing_luck_variants.csv` |
| index-level SMA-200, CAGR / Sharpe | 7.43% / 0.421 | 7.60% / 0.451 | `index_vs_stock.csv` |
| constituent daily SMA-200, CAGR / Sharpe | 5.73% / 0.212 | 5.87% / 0.239 | same |
| SMA-200 shortfall / cost component | 1.70 / 1.63 pp | 1.73 / 1.85 pp | `index_vs_stock_decomposition.csv` |
| SPY monthly overlay, net vs buy-and-hold Sharpe | 0.741 vs 0.764 | 0.740 vs 0.761 | `vol_managed_control_corrected.csv` |
| SMA-200 overlay average exposure | 0.0037 | 1.07 | same |
| Romano-Wolf, SMA lengths beating SPY | 0 of 5 | 0 of 5 | `romano_wolf_sweep.csv` |
| Deflated Sharpe, SMA-200 | 0.985 | 0.993 | `sma_sweep.csv` |
| PBO, median OOS rank of the winner | 0.613, 3.0 of 5 | 0.867, 2.0 of 5 | `pbo_sweep.csv` |

The conclusion is unchanged: the strategy earns less than SPY and nothing in the search survives a
multiple-testing check. Two statements made before the rerun no longer hold: that the strategy
loses to SPY on Sharpe as well (it is now level, 0.419 against 0.414), and that every tested
calendar loses on CAGR (five of 135 anchor variants end above SPY's CAGR). The volatility-control
SMA-200 leg is no longer degenerate; its near-zero exposure came from the cash warm-up.

**What the offline rerun could not do.** It used the local CRSP cache, the local name-history
extract and CIZ caches already corrected with the verified delisting extract, with no WRDS
connection.

- 329 of 4,550 membership segments (76,753 of 3,730,986 member business-days, 2.1%) resolve to no
  CRSP security: bankruptcy "Q" symbols CRSP never listed, and short gaps at name-record boundaries
  (284 segments are five days or shorter; 39 members lose a year or more).
- 348 resolved stretches across 169 members had no cached price rows, because the cache holds
  prices by the ticker requested at the time and the earlier tickers (HWP, WAG, YHOO and others)
  were never requested. Filling them needs a WRDS pull of those tickers.
- Without the extract file, terminal sessions on the CIZ side were marked from the corrected caches;
  this may account for some of the rise from 1 to 38 ordinary-session differences.

## Summary that previously opened the README

**Sample:** 1996-01-02 to 2024-12-31, 7,300 daily observations on the legacy
CRSP tape, with a proxy for point-in-time S&P 500 membership. The calendar audit
in [methodology.md](methodology.md#holding-the-frequency-fixed) recomputes existing aggregates from committed data without credentials.

**Latest corrected result:** on the pinned 1996-2024 comparison, strategy CAGR
is **8.280% on legacy CRSP and 8.298% on CIZ**, against index CAGR of 9.977%
and 10.039%, respectively. The earlier 8.40% result below is retained as
history, not the current corrected headline. The tape comparison and the daily-return verifier showed the source
and arithmetic (both since rerun; see the correction above). The corrected volatility-managed control also
fails to improve net held-out Sharpe.

## Earlier results, preserved before the terminal-return repair

This historical rerun followed the initial data-integrity audit, before the later terminal-return correction. It used 1,133 CRSP-resolved constituents, 7,300 trading days and 21,836 trades. (`reports/tape_comparison_historical.csv` records `n_tickers` = 1,148 for the same run, CAGR 8.3957%; the two ticker counts are not reconciled in the committed files.) The tables in this section preserve that earlier legacy `crsp.dsf` run with `config.END_DATE` pinned to 2024-12-31; the corrected comparison is in [methodology.md](methodology.md#which-crsp-tape).

| metric | Strategy | S&P 500 total return |
|---|---|---|
| CAGR | 8.40% | **9.98%** |
| Annualised volatility | 16.85% | 19.27% |
| Sharpe | 0.363 | **0.399** |
| Max drawdown | 54.4% | 55.2% |
| Annual turnover | 4.54 | 0.03 |
| Annualised cost | **68 bps** | 0 |

The corrected comparison subtracts a flat 2% cash rate in its Sharpe ratios (`metric_policy` in
`reports/tape_strategy_daily_returns.manifest.json`). This earlier table used a different effective
cash rate, which is why the index Sharpe reads 0.399 here and 0.4140 there at the same 9.977% CAGR.

**The strategy underperforms simply holding the index, and does not reduce drawdown to compensate** (54.4% against 55.2%). Costs are about half the mechanism: 14.9 bps per trade at 4.5x annual turnover annualises to 68 bps, and rerunning the identical configuration with costs switched off recovers a 9.13% CAGR, so the measured drag is **74 bps against a 158 bp gap to the benchmark**. Costs explain 47% of the shortfall. The other half is the rule.

An earlier version of this table read 758 bps, and that figure was wrong. `metrics.py` averaged the cost rate over the days that traded and then annualised by 252, which prices a semi-monthly rebalance as if it rebalanced daily; this configuration trades on 9% of sessions, so the drag came out roughly eleven times too large. It is now annualised over every trading day and agrees with the costs-off rerun to within the compounding difference. `tests/test_quant_review.py` pins the behaviour. What the correction changes is the story, not the conclusion: the strategy still loses to the index, but it is no longer true that costs alone account for the loss.

## Family-wise error control across the sweep

The Deflated Sharpe asks whether the *selected* configuration beats the expected best of N noise strategies. Romano-Wolf stepwise testing asks a different question: across the whole sweep, which configurations beat the benchmark, holding the probability of **any** false rejection at 5%?

| null hypothesis | configurations significant |
|---|---|
| beats cash | **5 of 5** |
| beats the S&P 500 | **0 of 5** |

That contrast is the point. Tested against cash, every SMA length looks significant in this sample, but that does not establish an advantage over holding equities. A long-equity strategy can lose to cash. Tested against actually holding the index, none of them clear the bar, and every configuration has a *negative* mean excess return (t between -0.88 and -1.35).

Choosing the weak null would have produced five significant results and a much better-looking repository. The benchmark-relative test is the one reported.

The Deflated Sharpe over the 9-configuration sweep is 0.985, meaning the *selected configuration* is unlikely to be the best of nine noise strategies. That is worth stating precisely: it says the configuration search did not manufacture the result. It does not say the result is good, and here it is not.

## Probability of Backtest Overfitting

The Deflated Sharpe asks whether the selected configuration's Sharpe survives the fact that several were tried. The **Probability of Backtest Overfitting** asks a sharper question about the selection procedure itself: pick the best configuration in sample, and how often does it land in the bottom half out of sample? Bailey, Borwein, López de Prado and Zhu, ["The Probability of Backtest Overfitting"](https://doi.org/10.2139/ssrn.2326253) (*Journal of Computational Finance* 20(4), 2017).

`statistics_mt.probability_of_backtest_overfitting` implements combinatorially symmetric cross-validation: the daily return panel is cut into 16 contiguous blocks, all C(16,8) = 12,870 ways of using half for training and the complement for testing are enumerated, and PBO is the share of those splits in which the in-sample winner ranked in the bottom half out of sample. CSCV is used rather than a single train/test cut because one cut point is arbitrary, and because the symmetric design means every block trains exactly as often as it tests.

It is computed over the **five SMA lengths that already exist** (150/175/200/225/250), on the 7,300-day panel from 1996-01-02 to 2024-12-31 (7,296 observations enter the statistic: the four trailing days that cannot fill a 16th equal block are dropped rather than folded into the last one, which would break the symmetry CSCV is named for). Widening the sweep would make PBO look more interesting without making it more informative.

| statistic | value |
|---|---|
| PBO | **0.613** |
| median out-of-sample rank of the in-sample winner | **3.0 of 5** |
| best in-sample configuration (full sample) | `sma_150` |
| symmetric splits enumerated | 12,870 |

**Read 0.613 against 0.60, not against 0.50.** With five configurations the out-of-sample rank takes five values, and "bottom half" means rank 3 or worse, so a ranking that carried *no* information at all would score 3/5 = 0.60. The observed 0.613 is that number. Picking the best SMA length in sample tells you nothing whatsoever about which length will do best out of sample, and the median out-of-sample rank of the in-sample winner is exactly the median rank.

That sits beside the Deflated Sharpe rather than contradicting it, because the two measure different things:

| | question | answer |
|---|---|---|
| Deflated Sharpe 0.985 | does the *selected* configuration's Sharpe survive the fact that nine were tried? | yes, the search did not manufacture it |
| PBO 0.613 | does the in-sample *ranking* predict the out-of-sample ranking? | no, it is indistinguishable from noise |

The configuration search did not invent the result, and the configuration choice is also worthless. Both are consistent with the headline: there is no edge here to select over. It is also why the 200-day being pre-specified matters more than its rank: at an annualised Sharpe of 0.5630 it places 4th of the 5 lengths (150: 0.5944, 175: 0.5854, 250: 0.5755, 200: 0.5630, 225: 0.5625), and PBO says that ordering would not repeat. Those sweep figures are a different Sharpe from the 0.363 in the results table above and are not comparable to it: the table reports `metrics.py`'s (CAGR minus cash rate) / annualised volatility, while the sweep ranks configurations on the per-period mean over standard deviation of `output/sma_sweep_returns.csv` annualised by root-252, an arithmetic mean with no cash rate subtracted, which is why it reads higher.

Reproduce. `main.py` writes the per-configuration daily returns as `output/sma_sweep_returns.csv` (the full pipeline takes about nine minutes and needs a WRDS entitlement); `run_pbo.py` then needs nothing else:

```bash
python main.py
```

```bash
python run_pbo.py
```

This is the expected outcome for a 200-day SMA on index constituents, and it is reported rather than buried. The contribution of this repository is the infrastructure and the audit, not the strategy.

## What the literature says

This result is not news, and presenting it as this repository's finding would overstate it. Zakamulin has made the same case twice on far longer samples: ["The Real-Life Performance of Market Timing with Moving Average and Time-Series Momentum Rules"](https://doi.org/10.2139/ssrn.2242795) (*Journal of Asset Management* 15, 2014) argues that the published performance of moving-average timing rules contains considerable data-mining bias and ignores market frictions, and that the advantage largely disappears in out-of-sample tests carrying realistic transaction costs; ["A Comprehensive Look at the Empirical Performance of Moving Average Trading Strategies"](https://doi.org/10.2139/ssrn.2677212) (2015) reaches the same conclusion over **155 years** of data, finding no single optimal lookback and no reliable out-of-sample edge. What is contributed here is not the conclusion but the audit trail behind it: a point-in-time CRSP universe with the ticker-to-PERMNO collisions actually resolved, a cost model built from a published spread estimator rather than a flat assumption, and the multiple-testing statistics reported above.

## Caveats on these numbers

- **The sample ends 2024-12-31 because that is where the end date is configured.** `config.END_DATE` is pinned to the last date the legacy CRSP tape carries; extending past it means moving to the CIZ tape (`crsp_v2.py`), not moving the string. Override it with `BACKTEST_END_DATE` if you have a reason to. It previously resolved to `"today"`, which produced two artefacts now gone: every current constituent was flagged as needing a "recent tail" this tape does not have, and the run was not reproducible, because the snapshot gate rejects any cached snapshot whose requested end predates the request, so the benchmarks refetched from WRDS at every date rollover. An earlier draft read the first artefact as a 32% coverage gap; it was not one. Under the pin the coverage report is clean, **1,133** tickers resolve and 1,064 actually trade, with trades running 1996-10-15 to 2024-12-16. **48** tickers never appear in CRSP's name history at all, mostly delisted shells (`AAMRQ`, `ABKFQ`) rather than live constituents.
- **EODHD was removed entirely (2026-09).** The loader previously fell back to a
  second vendor for tickers CRSP could not resolve, and spliced a vendor tail onto
  CRSP history for current constituents. Both are gone: unresolved tickers are now
  reported and excluded, so the universe is a single consistent PERMNO-keyed panel
  and the coverage gate sees the true shortfall rather than a patched-over one.
- **Dual-class tickers needed a fix.** Vendor and index files write `BRK-B`; CRSP writes `BRK` with the class in a separate `shrcls` field, so the hyphenated form matches nothing and every dual-class name was silently lost. `permno_resolution.split_share_class` now splits the suffix and matches on ticker plus share class: `BRK-B` resolves to permno 83443, `BF-B` to 29946. Status: the CRSP loaders (`data_loader.py`, `crsp_v2.py`) do not call this resolver yet; they match the hyphenated ticker against CRSP's ticker field, so by the reasoning above `BRK-B` and `BF-B` are still unresolved in the published runs.
- **The buy-and-hold column in the raw output is unreliable.** It reports a 2.86% CAGR, which is implausible for the period; the ETF benchmark has CRSP coverage only from late 2014 and the contaminated cache for it was quarantined. Compare against the S&P 500 total-return column instead.
- Pre-2019 constituent history remains proxy-based rather than a licensed point-in-time master.

## The universe audit that prompted the rerun

An audit of the cached CRSP universe found that **437 of 1,153 tickers (38%) resolve to more than one PERMNO**. CRSP tickers are reused across companies and collide across share classes, and this loader resolved ticker to PERMNO without a point-in-time constraint, so a single "ticker" series could splice unrelated securities together.

Two distinct failure modes, both confirmed:

- **Sequential reuse: 327 tickers (28.4%) whose segments are separated by more than a year.** `SOLV` has a **26.7-year gap** between its two securities. The cached `VOO` series spliced **Vornado Operating Co** (permno 86379, common stock, 1998-2003) onto the **Vanguard S&P 500 ETF** (permno 12305, share code 73, 2014-2024). A 200-day SMA spanning such a boundary averages two different companies' prices, so the signal is corrupted, not merely one return.
- **Simultaneous share classes: 23 tickers with duplicate dates** (TAP, BIO, MKC, STZ, LEN, CBS, CNP). These raise inside `_extract_adjusted_series`; the exception is swallowed by a bare `except Exception` logged at DEBUG only, so the ticker is **silently dropped from the universe** - invisible selection bias.

Quantified with `quantify_contamination.py`: 288,281 of 6,198,577 rows (4.7%) fail point-in-time filtering, 27,474 returns are fabricated across splices, and 2,486 of those exceed 50% in a single day. The 200-day signal itself disagrees on only 0.01% of ticker-days, but returns above 200% still reach the P&L matrix.

Reproduce both:

```bash
python audit_universe.py --cache-dir data_cache
python quantify_contamination.py
```

Both scripts read the legacy flat cache files `data_cache/crsp_*.parquet` (written only when
`FREEZE_SNAPSHOTS=False`), and `quantify_contamination.py` also needs a licensed `crsp.dsenames`
extract at `data/universe/dsenames.parquet` (gitignored). They document the pre-fix audit and are
not part of the current pipeline.

## Results before the 2026-10-23 correction

The sections below are the methodology write-up as it stood before the 2026-10-23 rerun, kept
verbatim except for heading levels and links. Their numbers were superseded for the reasons in
[the correction](#correction-2026-10-23-rerun-after-the-code-review); the current figures are in
[methodology.md](methodology.md).

### Which CRSP tape

In January 2025 CRSP shipped the last release of Flat File Format 1.0 (SIZ) and now updates only Format 2.0 (CIZ). Schwarz, Walter and Weiss, ["Rewriting CRSP's History: Impact of Altered Monthly Returns on Asset Pricing"](https://doi.org/10.2139/ssrn.5074864) (*Journal of Financial and Quantitative Analysis*, 24 February 2026), measure the consequence: the transition "rewrites 9.62% of monthly returns by more than 1 basis point, primarily due to a change in the dividend reinvestment assumption", payouts reinvesting on the ex-date under CIZ against month-end under SIZ, with a 22 bp mean absolute difference among the altered returns and 11.43% of monthly long-short returns moving by more than 10 bps.

The same paper contains a prediction this repository is well placed to test. The reinvestment change is a *monthly* artefact, daily returns "did not change materially", and the authors rebuild CIZ monthly returns by compounding SIZ daily ones. A strategy evaluated on daily data should therefore be close to tape-invariant. `crsp_v2.py` re-pulls the whole constituent panel from `crsp.dsf_v2` and `run_tape_compare.py` reruns the headline configuration on both, holding the ticker-to-PERMNO resolution and the entire normalisation path fixed so that only CRSP's numbers change.

The corrected comparison pins the common end date to 2024-12-31 and the common membership universe to 1,133 resolved tickers. `reports/ciz_source_validation.csv` records the verified source: `crsp.dsf_v2` daily returns and `crsp.stkdelists` terminal metadata, with `DelRet` already included in CIZ `DlyRet` on `DelDlyDt`. Missing post-delist CIZ rows are restored on that date, never compounded into the prior row. The licensed extract remains outside the repository and is hash checked before use.

| corrected result | legacy `crsp.dsf` | CIZ `crsp.dsf_v2` |
|---|---|---|
| tickers resolved | 1,133 | 1,133 |
| covered constituent-days | 5,435,483 | 5,425,235 |
| terminal sessions | 736 | 712 |
| trading days | 7,300 | 7,299 |
| sample | 1996-01-02 to 2024-12-31 | 1996-01-02 to 2024-12-31 |
| strategy CAGR | 8.280% | **8.298%** |
| strategy Sharpe | 0.3727 | **0.3739** |
| strategy max drawdown | 54.36% | 54.26% |
| annual turnover | 4.541 | 4.543 |
| trades | 21,835 | 21,914 |
| annualised cost | 67.9 bps | 68.0 bps |
| index CAGR | 9.977% | **10.039%** |
| index Sharpe | 0.4140 | **0.4172** |

| constituent-day comparison at 1 bp | ordinary sessions | terminal sessions |
|---|---|---|
| comparable ticker-days | 5,423,806 | 717 |
| differ by more than 1 bp | **1** (0.000018%) | **665** (92.75%) |
| median absolute difference among altered days | 15.35 bps | 92.95 bps |
| mean absolute difference among altered days | 15.35 bps | 521.35 bps |
| covered by legacy only | 10,176 | 18 |
| covered by CIZ only | 0 | 712 |
| identifier mismatches | 0 | 0 |

The source correction changes the interpretation. Daily returns do not differ materially on ordinary sessions: only one comparable ordinary ticker-day moves by more than 1 bp. Nearly every altered comparable observation is a terminal session, where legacy and CIZ place the outcome on different dates. The 712 CIZ-only terminal observations are the restored post-delist rows. This is a convention difference that must be preserved rather than removed by double counting.

The strategy remains close across tapes after holding identifiers, membership, costs, and endpoint fixed. CIZ raises CAGR by 1.81 bps and Sharpe by 0.0012. Across 7,299 common strategy days, 895 differ by more than 1 bp; the mean absolute gap is 0.468 bp and the maximum is 191.9 bps. The historical pre-correction comparison, which had 1,148 tickers and did not restore the CIZ post-delist rows, remains in `reports/tape_comparison.csv` and `reports/tape_comparison_historical.csv`; the pinned result is `reports/tape_comparison_corrected.csv`. Constituent coverage and segmented differences trace to `reports/tape_coverage_corrected.csv` and `reports/tape_return_differences_by_session_corrected.csv`.

The tapes' benchmark calendars still differ by one day. Legacy carries SPY on 1997-03-31 with zero volume and CIZ does not, so the CIZ strategy panel has 7,299 sessions rather than 7,300.

**Which tape each table is on.** Every other table in these docs is on the legacy `crsp.dsf` tape, including the earlier headline, Romano-Wolf and PBO results in this page, the timing-luck sweep, the index-versus-stock decomposition and the volatility-managed control. The two-tape table in this section is the only one carrying `crsp.dsf_v2` numbers. Since the legacy tape stopped being updated in January 2025 and the sample ends 2024-12-31, that choice costs nothing in coverage; it will have to change for any extension past 2024.

```bash
BACKTEST_OFFLINE=1 \
CIZ_TERMINAL_RETURNS_PATH=/external/versioned/voo_ciz_terminal_returns.csv \
CIZ_TERMINAL_METADATA_PATH=/external/versioned/voo_ciz_terminal_returns.metadata.json \
python run_tape_compare.py
```

### How much of a moving-average result is the calendar

The sweep below uses the pre-repair legacy run whose semi-monthly SMA-200 CAGR is 8.40% (see this page); the tape headline above was 8.280%. A headline reports one signal-evaluation schedule. Hoffstein, Sibears and Faber, ["Rebalance Timing Luck: The Difference between Hired and Fired"](https://doi.org/10.2139/ssrn.3319045) (*The Journal of Index Investing* 10(1), 2019, 27-36), give the reason to distrust that: two portfolios running an identical rule and differing only in which day inside the period they rebalance on are different portfolios, and the spread between them is *rebalance timing luck*. Their follow-up with Braun, ["Rebalance Timing Luck: The (Dumb) Luck of Smart Beta"](https://doi.org/10.2139/ssrn.3673910) (2020), measures it above 100 bps annualised for long-only factor indices, with one S&P index showing calendar-year return differences above 40% from the rebalance schedule alone.

`timing_luck.py` adds an evaluation-schedule option and runs all five SMA lengths against all 27 schedules: daily, five weekly anchors, and twenty-one monthly anchors. Anchors are trading-day ordinals rather than calendar weekdays. Within each frequency the sweep changes the day; the pooled set also changes how often the rule trades. Its total range therefore measures frequency and anchor sensitivity together.

The headline schedule is carried alongside them as a 28th labelled row. It is not a 28th anchor: semi-monthly fires twice a month, so it is a frequency the anchor sweep does not contain. It is excluded from every dispersion statistic below and reported only to locate the headline inside the sweep.

| SMA | daily rule | **headline (semi-monthly)** | worst of 27 | best of 27 | CAGR range | CAGR s.d. | Sharpe range | gap, daily to index | range / daily gap | gap, headline to index | range / headline gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 | 4.93% | 9.07% | 4.93% | 9.82% | **4.89 pp** | 1.06 pp | 0.278 | 5.04 pp | **0.97** | 0.90 pp | **5.41** |
| 175 | 5.53% | 8.86% | 5.53% | 9.75% | 4.22 pp | 0.91 pp | 0.241 | 4.45 pp | 0.95 | 1.11 pp | 3.79 |
| 200 | 5.73% | **8.40%** | 5.73% | 9.60% | 3.86 pp | 0.84 pp | 0.221 | 4.24 pp | 0.91 | **1.58 pp** | **2.44** |
| 225 | 5.49% | 8.36% | 5.49% | 9.47% | 3.98 pp | 0.82 pp | 0.228 | 4.49 pp | 0.89 | 1.62 pp | 2.46 |
| 250 | 5.31% | 8.58% | 5.31% | 9.44% | 4.13 pp | 0.83 pp | 0.240 | 4.67 pp | 0.88 | 1.39 pp | 2.97 |

(The box plot that accompanied this table was regenerated with the corrected sweep; the earlier image is in the git history.)

**Where the headline sits.** The 8.40% pre-repair headline is the 200-day rule evaluated **semi-monthly**, which is `config.REBALANCE_DEFAULT` and is not one of the 27 anchors in the sweep. Placed among them it ranks **9th lowest of 27**: nineteen anchor schedules would have produced a higher CAGR for the same rule on the same data, and the best of them reaches 9.60% against the headline's 8.40%. It is not the unluckiest calendar available, which the daily rule is, but it sits in the bottom third of them.

The gap from 8.40% to the index's 9.98% is **1.58 pp**. The pooled 3.86 pp range is **2.44 times that gap**, but combines changes of frequency and anchor. It is not an error bar on the semi-monthly headline. No anchor sweep at the semi-monthly frequency has been run.

Three things fall out of that table.

**The pooled dispersion is large.** At every length the CAGR range across frequencies and anchors is 88% to 97% of the gap between the daily rule and buy-and-hold. That comparison does not identify how much is due to the anchor alone.

**Daily evaluation is the worst of the 27 at every single length.** Not near the bottom, the bottom: rank 1 of 27 for 150, 175, 200, 225 and 250 alike. The best variant is a monthly anchor in all five cases. That is not a coincidence of anchor choice, it is turnover. **At the 200-day length** the daily rule turns over 16.0 times a year against 3.1 for the monthly anchors, which at roughly 15 bps a trade is 240 bps of annual cost against 46 bps. That cost gap accounts for 42% to 50% of the spread at each length, so it is the largest single driver and not the whole of it; the rest is the rule acting on a different price.

**None of it rescues the strategy.** The best tested frequency, anchor and length reaches 9.82% against the index's 9.98%. Every tested configuration underperforms in CAGR. Neither the size of the pooled range nor that ranking establishes future performance.

#### Holding the frequency fixed

Same 1996-2024 sample, same five lengths, same 140 committed rows. The question
here is whether the pooled dispersion remains as large when only the anchor varies.
This is an exploratory diagnostic of existing results, with no new strategy
selection, holdout access or significance claim.

| SMA | original pooled CAGR range | weekly-only range (5 anchors) | monthly-only range (21 anchors) | between-frequency share of pooled squared dispersion |
|---|---|---|---|---|
| 150 | 4.89 pp | 1.65 pp | 1.14 pp | 83.8% |
| 175 | 4.22 pp | 1.30 pp | 1.18 pp | 82.4% |
| 200 | **3.86 pp** | **0.72 pp** | **1.35 pp** | **79.5%** |
| 225 | 3.98 pp | 0.56 pp | 1.44 pp | 80.7% |
| 250 | 4.13 pp | 0.44 pp | 1.37 pp | 85.3% |

Sources: `reports/timing_luck_by_frequency.csv` and
`reports/timing_luck_frequency_decomposition.csv`. The last column uses the
identity total sum of squares = within-frequency + between-frequency, with one
vote per existing anchor. The groups have unequal sizes and correlated returns;
this is a descriptive split, not a causal estimate or an ANOVA test. The daily
group has one observation, so its anchor standard deviation is undefined.

**The pure-calendar interpretation fails:** at 200 days, the monthly anchor
range is 1.35 pp, while the previously emphasized 3.86 pp also changes trading
frequency. Anchor choice still matters. The audit does not rescue the strategy
or estimate the benefit of averaging sub-portfolios. Primary literature and
remaining checks are in [the review notes](schedule_review_2026.md).

Quick check, entirely offline:

```bash
python run_schedule_audit.py --check
```

This verifies all five existing headline summaries from the 140 committed rows,
including the 200-day semi-monthly CAGR of 8.395669%, and both new tables.

Reproduce. This section and the two that follow share one panel load, which is why they share an entrypoint (needs WRDS data or a warm cache):

```bash
python run_timing_luck.py
```

### Index versus stock

Zakamulin's evidence against moving-average timing (cited in [audit_history.md](#what-the-literature-says)), reported on 155 years of data, is entirely *index-level*: both papers time a market index, not its constituents. This repository times the constituents. Those are different strategies, and no Zakamulin paper compares them directly, so the comparison below is this repository's rather than a replication: the same five rules applied to the S&P 500 total-return series, same cost model, same daily evaluation.

The single-instrument spread comes from the EDGE estimator run on the ETF's own OHLC. SPY is the instrument that tracks this series and CRSP covers it from 1993, so SPY's estimate is used throughout, averaging 21.2 bps. VOO is carried only as a cross-check: CRSP's first usable VOO estimate is 2011-03-09, and over the 3,477 overlapping days the two ETFs' EDGE spreads differ by 4.5 bps on average, with VOO the tighter at 18.4 bps. Using the more liquid instrument is the assumption that favours the index-level rule, which is the right direction for a test whose finding is that the index-level rule still loses.

| SMA | index CAGR | index Sharpe | index max DD | constituent CAGR | constituent Sharpe | constituent max DD |
|---|---|---|---|---|---|---|
| 150 | 5.69% | 0.282 | 36.3% | 4.93% | 0.162 | 68.0% |
| 175 | 6.76% | 0.367 | 29.4% | 5.53% | 0.199 | 66.6% |
| 200 | **7.43%** | **0.421** | 25.3% | 5.73% | 0.212 | 65.0% |
| 225 | 7.26% | 0.404 | 26.0% | 5.49% | 0.197 | 64.9% |
| 250 | 7.14% | 0.385 | 25.8% | 5.31% | 0.186 | 62.6% |
| *buy and hold* | *9.98%* | *0.399* | *55.2%* | | | |

The index-level rule beats the constituent-level rule at every length, and at the pre-specified 200 days it posts a **higher Sharpe than buy-and-hold** (0.421 against 0.399) on less than half the drawdown (25.3% against 55.2%), while still giving up 2.55 pp of CAGR. That is the classic shape of index-level trend following, and it does not survive at the stock level.

Where the constituent version loses, it loses on cost, not on signal. Rerunning both with costs switched off splits the shortfall exactly, since `shortfall = (index - constituent) with costs on` decomposes into `(index - constituent) with costs off` plus `(constituent cost drag - index cost drag)`:

| SMA | shortfall | whipsaw component | cost component |
|---|---|---|---|
| 150 | 0.76 pp | **-0.95 pp** | +1.71 pp |
| 175 | 1.24 pp | **-0.44 pp** | +1.67 pp |
| 200 | 1.70 pp | +0.06 pp | +1.63 pp |
| 225 | 1.77 pp | +0.33 pp | +1.44 pp |
| 250 | 1.83 pp | +0.31 pp | +1.52 pp |

The cost component is 1.4 to 1.7 pp at every length and swamps the other term. The whipsaw component is small, and at 150 and 175 days it is **negative**: gross of costs, running the rule name by name *beat* running it on the index. Applying a moving average to 500 stocks instead of to their index is not a worse signal here. It is a signal that costs two to three times as much to run: **at the 150-day length** the constituent version turns over 19.5 times a year against the index version's 9.5, and it wears per-name spreads instead of an ETF's.

Sources: [`reports/index_vs_stock.csv`](../reports/index_vs_stock.csv) and [`reports/index_vs_stock_decomposition.csv`](../reports/index_vs_stock_decomposition.csv).

### Volatility-managed control

The control is Moreira and Muir, ["Volatility-Managed Portfolios"](https://doi.org/10.1111/jofi.12513) (*The Journal of Finance* 72(4), 2017, 1611-1644): scale exposure by `c` over trailing realised variance, with exposure capped at 1.5. The first half of the sample now calibrates `c`; every reported performance statistic uses only the frozen second half, 2010-07-02 through 2024-12-31.

The corrected simulator trades from drifted risky and cash holdings, charges the resulting exposure trade, holds the monthly target between month starts, and compounds the annual cash yield over elapsed calendar days. The historical implementation treated target changes as turnover, accrued one trading day's cash over weekends, and included the calibration half in reported performance. Its aggregate files remain in `reports/vol_managed_control.csv` and `reports/vol_managed_control_historical.csv`; the held-out reproduction is `reports/vol_managed_control_corrected.csv`.

| leg | variant | held-out CAGR | vol | Sharpe | Sharpe (arith.) | max DD | turnover |
|---|---|---|---|---|---|---|---|
| S&P 500 TR | buy and hold | 14.93% | 16.93% | 0.764 | 0.787 | 33.7% | 0.0 |
| S&P 500 TR | monthly, gross | 13.80% | 15.11% | 0.781 | 0.800 | 21.9% | 3.9 |
| S&P 500 TR | monthly, net | 13.20% | 15.10% | 0.741 | 0.765 | 22.0% | 3.9 |
| S&P 500 TR | daily, net | 11.65% | 13.61% | 0.709 | 0.731 | 15.5% | 8.0 |
| equal-weight constituents | buy and hold | 13.82% | 18.27% | 0.647 | 0.689 | 39.6% | 0.0 |
| equal-weight constituents | monthly, net | 8.65% | 14.67% | 0.453 | 0.503 | 28.4% | 4.3 |
| equal-weight constituents | daily, net | 6.63% | 13.26% | 0.349 | 0.400 | 21.7% | 9.1 |

The historical full-sample result said the index monthly net overlay raised geometric Sharpe from 0.399 to 0.488. The corrected held-out result does not reproduce that claim. The gross monthly overlay raises geometric Sharpe only from 0.764 to 0.781, and estimated costs reverse it to 0.741. The gross daily overlay reaches 0.799, but falls to 0.709 net. The equal-weight constituent overlay is negative even before costs.

Romano-Wolf testing over all six net overlays rejects none at 5% family-wise error. Every mean excess return is negative, adjusted p-values run from 0.940 to 1.000, and the result is recorded in `reports/vol_managed_romano_wolf_corrected.csv`. This is a failed positive-control hypothesis, not evidence that volatility management improves executable profit in this sample.

The 15 held-out buy-and-hold, gross, and net daily portfolio series behind the
control table are committed in
`reports/vol_managed_control_daily_returns.csv.gz`, with source labels and
metric conventions in its adjacent manifest. `python
verify_corrected_reports.py` independently recomputes every corrected control
CAGR, volatility, geometric and arithmetic Sharpe, and max drawdown. (That file and the
tape file were removed from the repository on 2026-10-23 because their SPY columns are CRSP's
own returns; full copies are now written to `output/evidence/` and only constituent-portfolio
series are committed.)

The SMA-200 leg remains a documented degeneracy. Its held-out average exposure is 0.0037 because near-cash training returns make inverse realised variance dominate the calibration. Inverse-variance scaling is not useful on a return series with long near-flat stretches. The longest such stretch is the SMA's own 200-session warm-up at the start of the training half, when the rule can only hold cash; calibrating `c` from the first invested session would remove it, and would change the four SMA-200 overlay rows and the Romano-Wolf table.

Cederburg, O'Doherty, Wang and Yan, ["On the performance of volatility-managed portfolios"](https://doi.org/10.1016/j.jfineco.2020.04.015) (*Journal of Financial Economics* 138(1), 2020, 95-117), find that the strategy does not survive real-time implementation for most factors. The corrected result is consistent with that caution.

### Known limits

- Pre-2019 constituent history is still proxy-based, not a licensed S&P point-in-time master.
- **The SMA warm-up sits inside the sample.** The panel starts on 1996-01-02 with no earlier prices, so each rule holds cash for its first L sessions (the 200-day rule first trades on 1996-10-15) while buy-and-hold is invested. Every strategy-versus-index gap in these docs includes that stretch; loading pre-1996 history and rerunning would remove it.
- Adjusted-open execution on daily data is an approximation, even with QA repair.
- Corporate-event outliers in crisis periods can still exist in constituent data and should be reviewed before live deployment.
- **Frequency and anchor sensitivity differ.** The pooled 27-schedule range mixes both. The within-frequency tables isolate anchor dispersion; neither table supplies an uncertainty interval for the semi-monthly headline.
- **The pre-2010 index-level spread is SPY's, not the fund the strategy would have held.** CRSP's first usable VOO estimate is 2011; SPY's EDGE spread is used for the whole sample and averages 21.2 bps against VOO's 18.4 bps over the overlap.
- **CIZ delisting returns are bridged through the CIZ delisting table.** 712 CIZ-only terminal rows are restored. 665 of 717 comparable terminal observations still differ from the legacy tape by more than 1 bp because the two tapes place the delisting return on different dates; that is a convention difference and is preserved rather than removed. Ordinary days agree on all but one of 5,423,806.
- **The volatility-managed control is only a control.** Calibrated on the first half and evaluated on 2010-07-02 to 2024-12-31, the monthly overlay lowers volatility and drawdown but does not raise net geometric Sharpe above buy-and-hold (0.741 against 0.764), and none of the six net overlays passes the Romano-Wolf test. The earlier full-sample version that appeared to beat buy-and-hold used evaluation-period information in its scaling constant and is withdrawn.
- The cash sleeve borrows at the three-month bill rate when the volatility overlay levers up to its 1.5 cap. A retail account pays more, so that leg is flattered.
- **WRDS defaults to disabled.** Both CRSP loaders refuse a connection unless `WRDS_DUO_READY=1` is set for the current session, which triggers a Duo push. Warm snapshots still work with the gate closed. Unresolved tickers still lack negative-cache entries, so a caller with the gate open may attempt a new resolution. Each connection call makes at most one login attempt and never prompts or retries. Batch queries into one session, and do not enable the gate in CI or background jobs.

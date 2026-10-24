# VOO Constituent SMA Backtest

A point-in-time backtest of a 200-day moving-average rule applied stock by stock to the S&P 500
(VOO) constituents from 1996 to 2024, built on CRSP daily data with modeled trading costs.

Trend rules often look good in backtests that use today's index members, fill at the close and ignore
costs. Each of those shortcuts can change the answer. I wanted to know whether holding only the
constituents above their 200-day average beats simply holding the index once membership, ticker
reuse, delistings and costs are handled carefully. It does not beat it on return, and its slightly
better risk-adjusted ratio does not survive multiple-testing checks; most of the repository is the
data and validation work needed to say that with confidence.

## Results

Sample: 1996-01-02 to 2024-12-31, 7,300 trading days, 1,133 constituent tickers resolved in CRSP.
The benchmark is SPY's total return from CRSP. Sharpe is (CAGR minus 2% cash) over annualised
volatility. All figures are from the rerun of 2026-10-23.

- The strategy (SMA-200, rebalanced twice a month) returned 8.85% a year against 9.98% for SPY. It
  ran at lower volatility (16.4% against 19.3%) and a smaller worst drawdown (52.1% against 55.2%),
  so its Sharpe is about the same (0.419 against 0.414). It turned over 5.7 times a year at an
  estimated 85 bps of annual cost.
  Sources: [`tape_comparison_corrected.csv`](reports/tape_comparison_corrected.csv),
  [`results_summary.csv`](reports/results_summary.csv).
- No SMA length from 150 to 250 days beats SPY in a Romano-Wolf test at 5% family-wise error (0 of
  5; every mean daily excess return is negative), and choosing the length with the best in-sample
  Sharpe has a probability of backtest overfitting of 0.867.
  Sources: [`romano_wolf_sweep.csv`](reports/romano_wolf_sweep.csv),
  [`pbo_sweep.csv`](reports/pbo_sweep.csv).
- I reran the identical backtest on CRSP's legacy tape and its new CIZ format and compared 5,425,928
  constituent-days. On ordinary sessions 38 of 5,425,294 returns differ by more than 1 bp; most
  differences sit on delisting days, where the two formats date the final return differently. The
  strategy's CAGR moves by 3.0 bps (8.848% to 8.879%).
  Sources: [`tape_return_differences_by_session_corrected.csv`](reports/tape_return_differences_by_session_corrected.csv),
  [`tape_comparison_corrected.csv`](reports/tape_comparison_corrected.csv).
- Across 140 SMA-length and rebalance-schedule configurations, the SMA-200 CAGR ranges from 5.87%
  (daily evaluation) to 10.08% (best monthly anchor). Differences between rebalance frequencies,
  rather than the day chosen within a frequency, account for 71.7% of the squared dispersion (a
  descriptive split).
  Sources: [`timing_luck_summary.csv`](reports/timing_luck_summary.csv),
  [`timing_luck_frequency_decomposition.csv`](reports/timing_luck_frequency_decomposition.csv).
- The same rule applied to SPY itself (SMA-200, daily evaluation) earns 7.60% with a 0.451 Sharpe,
  against 5.87% and 0.239 stock by stock. Costs explain more than the whole 1.73 pp gap (1.85 pp).
  Sources: [`index_vs_stock.csv`](reports/index_vs_stock.csv),
  [`index_vs_stock_decomposition.csv`](reports/index_vs_stock_decomposition.csv).

## How it works

1. Build daily index membership from a public S&P 500 membership history before October 2019 and
   from VOO holdings in SEC filings after it, applied with a 45-business-day lag.
2. Resolve each membership interval to the CRSP security (PERMNO) that was the member, using CRSP's
   name history, so a renamed company or a dual-class ticker is priced from the right security and
   a reused ticker never splices two companies together.
3. Build a total-return price path per security from CRSP daily returns, including the delisting
   return on the day a stock leaves (legacy `dlret` or CIZ `DelRet`). Prices from 1993 serve as
   signal history, so the 200-day average is defined on the first sample day.
4. On each rebalance date, hold equal weights in every member trading above its 200-day average and
   put the rest in cash. Signals use data through the prior close; trades fill at the next open.
   Small reweights inside a 25 bp band are skipped; entries and exits always trade.
5. Charge costs per trade: EDGE effective spreads (Ardia, Guidotti and Kroencke, 2024), 3 bps of
   opening-auction slippage, square-root participation impact and the FINRA sell-side fee. The $5
   price filter and dollar volume use traded prices.
6. Stress the result: rerun on both CRSP formats, sweep SMA lengths and rebalance schedules, compare
   with the index-level rule and a volatility-managed control, and apply multiple-testing checks
   (Deflated Sharpe, Romano-Wolf, probability of backtest overfitting).
7. Commit aggregate tables and the daily returns of the portfolios, and recompute the headline
   metrics from them offline with `verify_corrected_reports.py`.

## Limitations

- Membership before October 2019 comes from a public proxy, not a licensed S&P point-in-time file.
  Members listed under bankruptcy "Q" symbols that CRSP never carried stay unresolved, and the
  offline rerun had no cached prices for some resolved members' earlier tickers; both need a WRDS
  pull ([details](docs/audit_history.md#correction-2026-10-23-rerun-after-the-code-review)).
- The CIZ side of the tape comparison was rerun from caches already corrected with the verified
  delisting extract, without the extract itself, so terminal sessions are marked from those caches.
- Cash earns a flat 2% a year; the 3-month T-bill series is supported but was not used.
- Execution uses daily open prices, so fills and costs are modeled, not observed.
- Rebuilding the results needs a CRSP licence through WRDS; no CRSP record is committed.

## Repository layout

| path | what it holds |
|---|---|
| `config.py` | all settings: dates, costs, schedules, cash rate |
| `data_loader.py`, `crsp_v2.py`, `panel.py` | CRSP loading (legacy and CIZ), membership resolution, panel assembly |
| `permno_resolution.py` | ticker and membership-interval resolution to PERMNOs |
| `preprocessing.py`, `spread_edge.py` | membership matrix, liquidity and EDGE spread features |
| `strategy.py`, `backtest_engine.py`, `metrics.py`, `reporting.py` | signals, portfolio simulation, metrics, report and charts |
| `statistics_mt.py`, `timing_luck.py`, `vol_managed.py` | multiple testing, schedule sweep, volatility control |
| `main.py`, `run_*.py` | entry points (order below) |
| `verify_corrected_reports.py`, `finalize_corrected_reports.py`, `report_evidence.py` | offline verification and publishing of `reports/` |
| `audit_universe.py`, `quantify_contamination.py` | the ticker-reuse audit that led to the PERMNO work |
| `reports/` | committed result tables and portfolio daily returns |
| `data/universe/` | membership history and SEC holdings proxy |
| `tests/` | offline unit tests, run in CI |
| `docs/` | methodology, audit history, reproduction notes |

## Running it

Without any licensed data, the tests and the two verification scripts run from committed files:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests -q
python run_schedule_audit.py --check   # timing-luck tables from reports/timing_luck_variants.csv
python verify_corrected_reports.py     # strategy, tape-gap and overlay metrics from committed returns
```

With CRSP access through WRDS (credentials in `.env`, copied from `.env.example`, `WRDS_DUO_READY=1`
for the session, and `CRSP_NAMES_PATH` pointing at a CRSP name-history extract), run the pipeline in
this order:

1. `python main.py`: headline backtest, SMA sweep, Romano-Wolf, PBO, report and charts.
2. `python run_pbo.py`: the PBO table with the best in-sample configuration.
3. `python run_timing_luck.py`: schedule sweep, index versus stock, volatility-managed control.
4. `python run_schedule_audit.py`: the frequency tables built from step 3.
5. `python run_tape_compare.py`: legacy versus CIZ; needs the CIZ delisting extract
   (`CIZ_TERMINAL_RETURNS_PATH`, `CIZ_TERMINAL_METADATA_PATH`) or caches corrected with it
   (`CIZ_CORRECTED_CACHE_VERSION`).
6. `python verify_corrected_reports.py`.

`audit_universe.py` and `quantify_contamination.py` reproduce the historical ticker-reuse audit.
Environment variables, outputs and the exact rerun command are in
[docs/reproducing.md](docs/reproducing.md).

## More detail

- [docs/methodology.md](docs/methodology.md): strategy definition, cost model, and the full tables
  for the headline, multiple testing, tape comparison, timing luck, index versus stock and the
  volatility control.
- [docs/audit_history.md](docs/audit_history.md): earlier results, each correction that changed
  them (including the 2026-10-23 rerun), and the universe audit.
- [docs/schedule_review_2026.md](docs/schedule_review_2026.md): notes on the frequency audit and
  its sources.
- [docs/reproducing.md](docs/reproducing.md): setup, script order, outputs and WRDS access.
- [DATA.md](DATA.md): data sources, what is and is not committed, and licensing.

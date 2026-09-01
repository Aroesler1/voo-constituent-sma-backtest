# VOO Constituent SMA Backtest

Institutional-style Python backtest for a constituent-level VOO trend-following strategy, built to research-grade data standards rather than around a strategy claim.

Repository: `https://github.com/Aroesler1/voo-constituent-sma-backtest`

## What it does

- **Point-in-time universe construction** from CRSP, with S&P membership history and delisting returns compounded into the final return of names that exit
- **Retail-implementable cost model**: EDGE effective spreads (Ardia, Guidotti & Kroencke, JFE 2024), opening-auction slippage, participation-based impact, and FINRA regulatory fees
- **Multiple-testing-aware validation**: Deflated Sharpe Ratio over the SMA-length sweep, with the full configuration grid as the trial pool
- **A universe integrity audit** (`audit_universe.py`) that checks the CRSP ticker→PERMNO mapping is point-in-time correct, and a resolver (`permno_resolution.py`) that fixes it

That last item is what this repository is currently most useful for. The strategy is a 200-day SMA — deliberately simple. The infrastructure around it is the substance, and running the audit against it produced the finding below.

> ### Results are being regenerated after a data-integrity audit
>
> An audit of the cached CRSP universe found that **437 of 1,153 tickers (38%) resolve to more than one PERMNO**. CRSP tickers are reused across companies and collide across share classes, and this loader resolved ticker to PERMNO without a point-in-time constraint, so a single "ticker" series could splice unrelated securities together.
>
> Two distinct failure modes, both confirmed:
>
> - **Sequential reuse: 327 tickers (28.4%) whose segments are separated by more than a year.** `SOLV` has a **26.7-year gap** between its two securities. The cached `VOO` series splices PERMNO 86379 (1998-2010, an unrelated company that held the ticker) onto 12305 (the actual Vanguard ETF, from 2010). A 200-day SMA spanning such a boundary averages two different companies' prices, so the signal is corrupted, not merely one return.
> - **Simultaneous share classes: 23 tickers with duplicate dates** (TAP, BIO, MKC, STZ, LEN, CBS, CNP). These raise inside `_extract_adjusted_series`; the exception is swallowed by a bare `except Exception` logged at DEBUG only, so the ticker is **silently dropped from the universe** - invisible selection bias in a repo whose premise is point-in-time discipline.
>
> **Every performance number below predates this audit and should not be relied on.** The fix is to resolve ticker to PERMNO as of each date via `crsp.dsenames` (`namedt`/`nameendt`), treat each PERMNO as a distinct instrument, and never concatenate across PERMNOs. Regeneration is blocked only on restoring WRDS access.
>
> Reproduce the audit yourself against any cache directory:
>
> ```bash
> python audit_universe.py --cache-dir data_cache
> ```

The strategy evaluates the holdings underlying `VOO` on a point-in-time basis and holds only constituents trading above their `200-day SMA`, allocating capital equally across active names and routing the remainder to cash when breadth collapses.

## What This Project Does

- Builds a point-in-time constituent universe using:
  - SEC-based `VOO` holdings proxy post-2019
  - public S&P 500 membership history proxy pre-2019
- Fetches price history with `CRSP/WRDS` as primary and `EODHD` as validated fallback
- Simulates daily, weekly, semi-monthly, and monthly rebalance schedules
- Applies realistic implementation assumptions:
  - next-session open execution
  - dynamic cash rates via FRED `DGS3MO`
  - spread, impact, slippage, and regulatory-fee modeling
  - frozen input snapshots and run manifests for reproducibility
- Produces a full research report:
  - equity curves
  - drawdowns
  - rolling Sharpe
  - active breadth
  - cost diagnostics
  - schedule comparison
  - SMA parameter sweep

## Repository Layout

```text
config.py            Runtime configuration and environment loading
data_loader.py       Vendor, snapshot, and cash-rate ingestion
preprocessing.py     Universe construction, resampling, liquidity features
strategy.py          SMA and signal generation
backtest_engine.py   Constituent-level portfolio simulation
metrics.py           Performance and risk analytics
reporting.py         Tables, charts, and markdown report generation
main.py              End-to-end pipeline entrypoint
data/universe/       Source universe proxy datasets
requirements.txt     Python dependencies
```

## Strategy Definition

For each eligible constituent:

1. Compute the `200-day SMA` on adjusted close.
2. Mark the name `active` when price is above its SMA.
3. On rebalance dates, allocate equally across active names.
4. If no names are active, stay in cash.
5. Execute at the next session open with modeled implementation costs.

Default reporting schedule is `semi_monthly`, with full comparisons against `daily`, `weekly`, and `monthly`.

## Realism Features

- Point-in-time constituent membership
- Snapshot-backed vendor inputs
- CRSP-first price sourcing with cross-vendor validation
- Time-varying cash sleeve using `DGS3MO`
- Retail-implementable cost model:
  - opening-auction slippage
  - EDGE spread estimator (Ardia, Guidotti, Kroencke, JFE 2024) by default, Corwin-Schultz retained for comparison
  - participation-based impact
  - FINRA sell-side regulatory fee support
- Output manifests for deterministic reruns

## Statistical Honesty

- The SMA-length sweep and schedule comparison constitute a multiple-testing search, so the report includes the Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014) for the selected configuration and for every sweep entry, with the full sweep treated as the trial pool. A high raw Sharpe with a low deflated Sharpe means the configuration choice is not statistically distinguishable from picking the best of several noise strategies.
- Trade-level profit concentration (`profit_top5pct_share`, `profit_top10pct_share`) is reported because single-stock trend following concentrates most profit in a small tail of trades; averages alone hide this dependence.
- CRSP delisting returns (`dsedelist.dlret`) are compounded into the final return of names that exit, so departures do not silently leave at their last quoted price.

## Known Limits

- Pre-2019 constituent history is still proxy-based, not a licensed S&P point-in-time master.
- Adjusted-open execution on daily data is an approximation, even with QA repair.
- Corporate-event outliers in crisis periods can still exist in constituent data and should be reviewed before live deployment.

## Setup

Create a local environment with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and provide the credentials you have available:

```bash
cp .env.example .env
```

Supported credentials:

- `WRDS_USERNAME` / `WRDS_PASSWORD` for CRSP access
- `EODHD_API_KEY` for fallback price coverage
- `FRED_API_KEY` for cash-rate retrieval

## Running The Backtest

From the project root:

```bash
./.venv/bin/python main.py
```

Run the unit tests (no credentials required):

```bash
./.venv/bin/python -m pytest tests -q
```

## Generated Outputs

Successful runs write artifacts to `output/`, including:

- `results_summary.csv`
- `detailed_report.md`
- `equity_curves.png`
- `drawdowns.png`
- `rolling_sharpe.png`
- `active_breadth.png`
- `cost_diagnostics.png`
- `schedule_comparison.png`
- `schedule_risk_return.png`
- `sma_sweep.png`
- `regime_comparison.png`
- `run_manifest.json`

## Intended Use

This codebase is designed for research-grade backtesting and strategy evaluation, not for direct live trading deployment without a licensed point-in-time constituent master and broker-specific transaction-cost calibration.

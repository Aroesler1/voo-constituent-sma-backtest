# VOO Constituent SMA Backtest

Institutional-style Python backtest for a constituent-level VOO trend-following strategy.

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
  - Corwin-Schultz spread estimate
  - participation-based impact
  - FINRA sell-side regulatory fee support
- Output manifests for deterministic reruns

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

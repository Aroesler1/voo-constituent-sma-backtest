# Reproducing the results

The test suite and two verification scripts run offline from committed files. Everything else needs
CRSP data through a WRDS entitlement, or a warm local cache built from one.

## Offline checks (no credentials)

```bash
python -m pytest tests -q
python run_schedule_audit.py --check
python verify_corrected_reports.py
```

- `run_schedule_audit.py --check` rebuilds the five timing-luck summary rows and both frequency
  tables from `reports/timing_luck_variants.csv`.
- `verify_corrected_reports.py` recomputes CAGR, volatility, Sharpe, drawdown and the tape gap
  statistics from portfolio-level daily returns. With only the committed files it checks the
  strategy on both tapes, the tape gap and the constituent overlays, and reports the SPY and
  index-leg metrics as skipped; with `output/evidence/` present locally it checks those too.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # WRDS_USERNAME / WRDS_PASSWORD, optional FRED_API_KEY
```

## Environment

| variable | effect |
|---|---|
| `BACKTEST_OFFLINE=1` | cached inputs only; a missing ticker is reported, never fetched |
| `BACKTEST_CACHE_DIR` | location of the CRSP snapshot cache (default `./data_cache`) |
| `BACKTEST_END_DATE` | sample end (default and required for the tape study: 2024-12-31) |
| `CRSP_NAMES_PATH` | CRSP name-history extract; enables PERMNO resolution of membership |
| `CIZ_TERMINAL_RETURNS_PATH`, `CIZ_TERMINAL_METADATA_PATH` | verified CIZ delisting extract and sidecar |
| `CIZ_CORRECTED_CACHE_VERSION` | reuse CIZ caches already corrected with that extract, read-only |
| `WRDS_DUO_READY=1` | allows one WRDS login for the session (see [WRDS access](#wrds-access)) |
| `FRED_API_KEY` | time-varying 3-month T-bill cash rate; without it cash earns a flat 2% |

## Script order

| step | script | needs | writes |
|---|---|---|---|
| 1 | `main.py` | WRDS or warm cache | `output/` report, charts and `run_manifest.json`; summary, schedule, sweep, PBO and Romano-Wolf tables in `output/` and `reports/` |
| 2 | `run_pbo.py` | `output/sma_sweep_returns.csv` | `pbo_sweep.csv` with the best configuration, in `output/` and `reports/` |
| 3 | `run_timing_luck.py [--parts 1,2,3]` | WRDS or warm cache | timing-luck, index-vs-stock and volatility-control tables in `output/` and `reports/`, `figures/timing_luck_box.png` |
| 4 | `run_schedule_audit.py` | `reports/timing_luck_variants.csv` | the two frequency tables in `reports/` |
| 5 | `run_tape_compare.py [--threshold-bps 1.0]` | warm legacy and CIZ caches, CIZ extract or corrected caches | tape tables in `output/` and `reports/` |
| 6 | `verify_corrected_reports.py` | `reports/`, optionally `output/evidence/` | nothing |

`finalize_corrected_reports.py --ciz-metadata <sidecar>` re-copies the corrected tables from
`output/` to `reports/`, regenerates `reports/ciz_source_validation.csv` from the sidecar and runs the
verifier; steps 3 and 5 already write the `reports/` copies.

`audit_universe.py` and `quantify_contamination.py` reproduce the historical universe audit; see
[audit_history.md](audit_history.md#the-universe-audit-that-prompted-the-rerun).

## The 2026-10 rerun

Every current table in `reports/` was regenerated on 2026-10-23 from a warm local cache, offline;
the earlier-run copies and `ciz_source_validation.csv` (same extract) were left as they were:

```bash
export BACKTEST_OFFLINE=1
export BACKTEST_CACHE_DIR=/path/to/warm/data_cache
export CRSP_NAMES_PATH=/path/to/crsp_dsenames.parquet
python main.py
python run_pbo.py
python run_timing_luck.py
python run_schedule_audit.py
CIZ_CORRECTED_CACHE_VERSION=terminal-v3-7fe0edc5a632 python run_tape_compare.py
python verify_corrected_reports.py
```

What that cache could not supply is listed in
[audit_history.md](audit_history.md#correction-2026-10-23-rerun-after-the-code-review).

## Outputs

`output/` is gitignored. The aggregate tables, and daily returns of portfolios built by this code,
are committed to `reports/`. No ticker-level CRSP data and no CRSP return series is committed; the
full daily-return files, which also carry SPY's CRSP returns, are written to `output/evidence/`.

```text
reports/
  timing_luck_variants.csv                  140 rows: 5 lengths x (27 anchors + the headline schedule)
  timing_luck_summary.csv                   dispersion per length, both range-over-gap scalings
  timing_luck_by_frequency.csv              anchor dispersion within each frequency
  timing_luck_frequency_decomposition.csv   within- and between-frequency sums of squares
  index_vs_stock.csv                        index and constituent rules, costs on and off
  index_vs_stock_decomposition.csv          shortfall split into cost and whipsaw
  vol_managed_control_corrected.csv         held-out volatility-managed control
  vol_managed_romano_wolf_corrected.csv     family-wise test of the six net overlays
  tape_comparison_corrected.csv             headline on both CRSP tapes
  tape_coverage_corrected.csv               tape coverage and terminal counts
  tape_return_differences_corrected.csv     all-session constituent-day differences
  tape_return_differences_by_session_corrected.csv
                                             ordinary versus terminal-session differences
  ciz_source_validation.csv                 CIZ terminal source fields, semantics, extract hash
  tape_strategy_daily_returns.csv.gz        strategy daily returns on each tape (+ manifest)
  vol_managed_constituent_daily_returns.csv.gz
                                             held-out constituent-leg series and cash (+ manifest)
  results_summary.csv, schedule_comparison.csv
                                             headline metrics and the four rebalance schedules
  sma_sweep.csv, pbo_sweep.csv, romano_wolf_sweep.csv
                                             SMA-length sweep, PBO and Romano-Wolf
  index_spread_notes.csv                    SPY and VOO EDGE spread averages for the index rule
  *_historical.csv, tape_comparison.csv, tape_return_differences*.csv,
  vol_managed_control.csv, vol_managed_romano_wolf.csv
                                             earlier runs, kept for the audit history
```

`main.py` also writes to `output/`: `detailed_report.md`, the equity,
drawdown, rolling-Sharpe, breadth, cost, schedule, sweep and regime charts, `trade_log.csv`,
`cost_attribution.csv` and `run_manifest.json`. `run_tape_compare.py` writes
`tape_largest_differences.csv` to `output/` only, because its rows are CRSP ticker-day returns.

## WRDS access

Both CRSP loaders refuse a connection unless `WRDS_DUO_READY=1` is set for the current session,
which triggers a Duo push. Warm snapshots still work with the gate closed. Unresolved tickers have
no negative-cache entries, so a run with the gate open may try to resolve them again. Each connection call
makes at most one login attempt and never prompts or retries. Batch queries into one session, and
do not enable the gate in CI or background jobs.

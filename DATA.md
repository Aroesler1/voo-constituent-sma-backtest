# Data provenance

**Primary source:** CRSP daily stock files via WRDS (Berkeley/Haas
subscription), requested from 1993-01-29. The backtest sample starts on
1996-01-02, the first date of the membership history, and ends 2024-12-31; the
1993-1995 prices are used only as signal history, so the 200-day average is
defined on the first sample day. The headline uses legacy `crsp.dsf`. The CIZ
comparison uses entitled `crsp.dsf_v2`; terminal-event metadata comes from
`crsp.stkdelists` with `DelDlyDt` and `DelRet`.

Supplementary: SEC-derived VOO holdings proxy (post-2019) and an S&P 500
membership history for point-in-time universe construction.

## What is committed

No CRSP record is committed: no constituent price or return, no CRSP index or
ETF return series, no name-history row.

- Source code, tests, and the universe audit tool
- Membership and holdings-proxy reference files under `data/universe/` (public sources)
- Aggregate result tables under `reports/` (metrics, counts, ranges)
- Daily returns of portfolios built by this code, in
  `reports/tape_strategy_daily_returns.csv.gz` (the strategy on each tape) and
  `reports/vol_managed_constituent_daily_returns.csv.gz` (equal-weight basket,
  SMA strategy, their overlays, and the flat cash sleeve), each with a SHA-256
  manifest. Series derived from SPY alone are excluded because they would
  reproduce CRSP's SPY returns.

## What is not committed

- `data_cache/` (gitignored): per-ticker CRSP parquet extracts
- `output/` (gitignored), including `output/evidence/`: the full daily-return
  files with the SPY benchmark and index-leg series, and the
  largest-return-differences table, which carries ticker-day CRSP returns
- the CRSP name-history extract (`crsp.dsenames` columns) used to resolve
  membership to PERMNOs; point `CRSP_NAMES_PATH` at it
- the CIZ terminal-return extract and its sidecar
- `logs/` (gitignored): run logs, which have previously contained vendor API tokens in error URLs

## Reproducing

The schedule audit uses only committed aggregates: `python run_schedule_audit.py --check`.
`python verify_corrected_reports.py` recomputes the strategy, tape-gap and
constituent-overlay metrics from the committed daily returns, and the SPY and
index-leg metrics as well when `output/evidence/` is present locally.

WRDS connections are disabled unless `WRDS_DUO_READY=1` is set for the current
session. Keep it unset for offline reproduction and CI. `BACKTEST_OFFLINE=1`
disables connection setup even when a ticker is absent from cache.

Membership resolution needs a CRSP name-history extract (`CRSP_NAMES_PATH`).
Each membership interval is mapped to the PERMNO that was the member, so a
renamed company (WBA for Walgreens' WAG years) or a dual-class ticker (BRK-B)
is priced from the right security. Prices are still fetched by the ticker CRSP
carried at the time, so a rerun from a cache needs those tickers cached; the
2026-10 offline rerun lists what was missing in
[docs/audit_history.md](docs/audit_history.md).

CRSP documents `DelRet` as part of the main CIZ daily series on `DelDlyDt`,
conventionally the trading day after delisting. It must not be compounded into
the prior day. A verified external extract and hash-checked sidecar restore
post-delist daily rows omitted by legacy name-end filtering
(`CIZ_TERMINAL_RETURNS_PATH`, `CIZ_TERMINAL_METADATA_PATH`). Caches already
corrected with that extract can be reused offline by naming their versioned
directory in `CIZ_CORRECTED_CACHE_VERSION` (for example
`terminal-v3-7fe0edc5a632`, the first 12 hex digits of the extract hash in
`reports/ciz_source_validation.csv`).

Historical aggregate tape and volatility-control reports remain under their
original names and explicit `_historical.csv` copies. Corrected reproductions
use `_corrected.csv`.

## Licence and retention

CRSP is licensed through the university subscription; only aggregates and
portfolio-level series computed by this code are published. Raw extracts are
deleted at the end of the associated academic affiliation.

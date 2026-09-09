# Data provenance

**Primary source:** CRSP daily stock files via WRDS (Berkeley/Haas
subscription), 1993-2024 in the pinned study. The historical headline uses
legacy `crsp.dsf`. The verified CIZ comparison uses entitled
`crsp.dsf_v2`; terminal-event metadata comes from `crsp.stkdelists` with
`DelDlyDt` and `DelRet`.

Supplementary: SEC-derived VOO holdings proxy (post-2019) and an S&P 500 membership history for point-in-time universe construction.

## What is committed

- Source code, tests, and the universe audit tool
- Membership and holdings-proxy reference files under `data/universe/`
- Derived results and run manifests

## What is not committed

- `data_cache/` (gitignored): ~4.7 GB of per-ticker CRSP parquet extracts
- `logs/` (gitignored): run logs, which have previously contained vendor API tokens in error URLs

## Reproducing

The schedule audit uses only committed portfolio aggregates and needs no vendor
access: `python run_schedule_audit.py --check`. Its inputs are
`reports/timing_luck_variants.csv` and `reports/timing_luck_summary.csv`; the
two new frequency tables contain only aggregate ranges and a descriptive
sum-of-squares decomposition. No new source or licensed raw observations are
added.

WRDS connections are disabled unless `WRDS_DUO_READY=1` is explicitly set after
approval for the current session. Keep it unset for offline reproduction and
CI. `BACKTEST_OFFLINE=1` disables connection setup even when a ticker is absent
from cache. Historical cache files are read in place; corrected CIZ files use a
new versioned directory.

CRSP documents `DelRet` as part of the main CIZ daily series on `DelDlyDt`,
conventionally the trading day after delisting. It must not be compounded into
the prior day. A verified external extract and hash-checked sidecar restore
post-delist daily rows omitted by legacy name-end filtering. Licensed terminal
rows remain outside the repository. Set `CIZ_TERMINAL_RETURNS_PATH` and
`CIZ_TERMINAL_METADATA_PATH` only for the offline tape rebuild.

Historical aggregate tape and volatility-control reports remain under their
original names and explicit `_historical.csv` copies. Corrected reproductions
use `_corrected.csv`; `reports/ciz_source_validation.csv` records the terminal
source fields, return semantics, row counts, and external extract hash. The
largest-return-differences table stays under gitignored `output/` because it
contains licensed ticker-day observations.

Corrected headline CSVs are verified from committed portfolio-level daily
returns in `reports/tape_headline_daily_returns.csv.gz` and
`reports/vol_managed_control_daily_returns.csv.gz`. Those files contain dated
strategy and benchmark series only, with SHA-256 manifests and no tickers or
PERMNOs. `python verify_corrected_reports.py` recomputes CAGR, volatility,
Sharpe, drawdown, and tape-gap fields from those series rather than comparing
report bytes to local output copies.

With a WRDS entitlement:

```bash
python audit_universe.py --cache-dir data_cache
```

then `python main.py`. The README's regenerated results include the point-in-time
PERMNO fix; archived exploratory outputs may predate it.

## Licence and retention

CRSP is licensed through the university subscription; only derived outputs are published. Raw extracts are deleted at the end of the associated academic affiliation.

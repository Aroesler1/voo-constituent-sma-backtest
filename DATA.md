# Data provenance

**Primary source:** CRSP daily stock file via WRDS (Berkeley/Haas subscription), 1993-2026, cached per ticker.

Supplementary: SEC-derived VOO holdings proxy (post-2019) and an S&P 500 membership history for point-in-time universe construction.

## What is committed

- Source code, tests, and the universe audit tool
- Membership and holdings-proxy reference files under `data/universe/`
- Derived results and run manifests

## What is not committed

- `data_cache/` (gitignored): ~4.7 GB of per-ticker CRSP parquet extracts
- `logs/` (gitignored): run logs, which have previously contained vendor API tokens in error URLs

## Reproducing

With a WRDS entitlement:

```bash
python audit_universe.py --cache-dir data_cache
```

then `python main.py`. Note the audit finding in the README: published results predate the point-in-time PERMNO fix.

## Licence and retention

CRSP is licensed through the university subscription; only derived outputs are published. Raw extracts are deleted at the end of the associated academic affiliation.

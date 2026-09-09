"""Snapshot metadata remains usable when a cache directory is relocated."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_snapshot, resolve_snapshot


def test_relative_historical_snapshot_path_relocates_from_metadata(tmp_path):
    snap = tmp_path / "snapshots" / "sp500pub" / "MEMBERSHIP" / "frozen"
    snap.mkdir(parents=True)
    expected = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"]), "ticker": ["ABC"]}
    )
    expected.to_parquet(snap / "normalized.parquet", index=False)
    metadata = {
        "vendor": "sp500pub",
        "ticker": "MEMBERSHIP",
        "snapshot_id": "frozen",
        "schema_version": 1,
        "requested_start": "2020-01-01",
        "requested_end": "2020-12-31",
        "fetched_at_utc": "2020-12-31T23:59:00+00:00",
        "min_date": "2020-01-02",
        "max_date": "2020-01-02",
        "normalized_path": "data_cache/snapshots/sp500pub/MEMBERSHIP/"
        "frozen/normalized.parquet",
    }
    (snap / "metadata.json").write_text(json.dumps(metadata))

    selected = resolve_snapshot(
        "sp500pub",
        "MEMBERSHIP",
        "2020-01-01",
        "2020-12-31",
        tmp_path,
        "2021-01-01T00:00:00+00:00",
    )
    actual = load_snapshot(selected)
    pd.testing.assert_frame_equal(actual, expected)

"""Deterministic writers for portfolio-level daily-return evidence.

Two copies are written for each study:

- a local copy under ``output/evidence/`` (gitignored) with every series,
  including the SPY benchmark, whose daily returns are CRSP's own values and
  so stay out of the repository;
- a committed copy under ``reports/`` with only series built from the
  constituent portfolio (strategy, equal-weight basket, overlays on them) and
  the flat cash sleeve. None of these is a CRSP record.

``verify_corrected_reports.py`` recomputes every metric it can from whichever
copies are present.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LOCAL_EVIDENCE_DIR = Path("output") / "evidence"

TAPE_DAILY_NAME = "tape_headline_daily_returns.csv.gz"
TAPE_MANIFEST_NAME = "tape_headline_daily_returns.manifest.json"
TAPE_PUBLIC_NAME = "tape_strategy_daily_returns.csv.gz"
TAPE_PUBLIC_MANIFEST_NAME = "tape_strategy_daily_returns.manifest.json"

CONTROL_DAILY_NAME = "vol_managed_control_daily_returns.csv.gz"
CONTROL_MANIFEST_NAME = "vol_managed_control_daily_returns.manifest.json"
CONTROL_PUBLIC_NAME = "vol_managed_constituent_daily_returns.csv.gz"
CONTROL_PUBLIC_MANIFEST_NAME = "vol_managed_constituent_daily_returns.manifest.json"


def deterministic_gzip_csv(frame: pd.DataFrame) -> bytes:
    """Serialize a frame as stable gzip CSV bytes.

    ``pandas`` otherwise places the current time in the gzip header, which
    makes a byte-for-byte reproduction check fail even when every return is
    unchanged.
    """
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as compressed:
        compressed.write(csv_bytes)
    return output.getvalue()


def write_evidence(
    frame: pd.DataFrame,
    *,
    directory: Path,
    name: str,
    manifest_name: str,
    return_columns: dict[str, dict[str, Any]],
    manifest_fields: dict[str, Any],
    columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write daily returns and their provenance sidecar into one directory.

    Args:
        frame: Daily returns with ``date`` as the first column.
        directory: Destination directory.
        name: Artifact file name.
        manifest_name: Sidecar file name.
        return_columns: Source labels for every return column of ``frame``.
        manifest_fields: Policy fields copied into the sidecar.
        columns: Optional subset of return columns to write.

    Returns:
        The manifest that was written.
    """
    if list(frame.columns)[0] != "date":
        raise ValueError("Daily evidence must have 'date' as its first column.")
    selected = list(columns) if columns is not None else [c for c in frame.columns if c != "date"]
    unknown = set(selected) - set(frame.columns)
    if unknown:
        raise ValueError(f"Unknown evidence columns: {sorted(unknown)}")
    subset = frame.loc[:, ["date", *selected]]
    payload = deterministic_gzip_csv(subset)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(payload)
    manifest = {
        **manifest_fields,
        "artifact_file": name,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload),
        "rows": len(subset),
        "first_date": str(pd.Timestamp(subset["date"].min()).date()),
        "last_date": str(pd.Timestamp(subset["date"].max()).date()),
        "return_columns": {c: return_columns[c] for c in selected},
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (directory / manifest_name).write_bytes(text.encode("utf-8"))
    return manifest

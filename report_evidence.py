"""Deterministic writers for committed aggregate daily-return evidence."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


TAPE_DAILY_NAME = "tape_headline_daily_returns.csv.gz"
TAPE_MANIFEST_NAME = "tape_headline_daily_returns.manifest.json"
CONTROL_DAILY_NAME = "vol_managed_control_daily_returns.csv.gz"
CONTROL_MANIFEST_NAME = "vol_managed_control_daily_returns.manifest.json"


def committed_report_path(name: str) -> str:
    """Return the repository-relative path recorded in corrected reports."""
    return f"reports/{name}"


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


def write_daily_evidence(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    report_dir: Path,
    name: str,
) -> dict[str, Any]:
    """Write aggregate daily returns to generated and tracked locations."""
    if list(frame.columns)[0] != "date":
        raise ValueError("Daily evidence must have 'date' as its first column.")
    payload = deterministic_gzip_csv(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_bytes(payload)
    (report_dir / name).write_bytes(payload)
    return {
        "artifact_path": committed_report_path(name),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload),
        "rows": len(frame),
        "first_date": str(pd.Timestamp(frame["date"].min()).date()),
        "last_date": str(pd.Timestamp(frame["date"].max()).date()),
    }


def write_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    report_dir: Path,
    name: str,
) -> None:
    """Write a deterministic provenance sidecar beside daily evidence."""
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_bytes(payload)
    (report_dir / name).write_bytes(payload)

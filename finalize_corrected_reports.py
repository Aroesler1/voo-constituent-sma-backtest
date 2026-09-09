#!/usr/bin/env python3
"""Publish corrected aggregate reports beside preserved historical outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from report_evidence import (
    CONTROL_DAILY_NAME,
    CONTROL_MANIFEST_NAME,
    TAPE_DAILY_NAME,
    TAPE_MANIFEST_NAME,
)
from verify_corrected_reports import verify_corrected_reports


MAPPINGS = {
    "tape_comparison.csv": "tape_comparison_corrected.csv",
    "tape_coverage.csv": "tape_coverage_corrected.csv",
    "tape_return_differences.csv": "tape_return_differences_corrected.csv",
    "tape_return_differences_by_session.csv": (
        "tape_return_differences_by_session_corrected.csv"
    ),
    "vol_managed_control.csv": "vol_managed_control_corrected.csv",
    "vol_managed_romano_wolf.csv": "vol_managed_romano_wolf_corrected.csv",
    TAPE_DAILY_NAME: TAPE_DAILY_NAME,
    TAPE_MANIFEST_NAME: TAPE_MANIFEST_NAME,
    CONTROL_DAILY_NAME: CONTROL_DAILY_NAME,
    CONTROL_MANIFEST_NAME: CONTROL_MANIFEST_NAME,
}


def expected_bytes(source: Path) -> bytes:
    """Preserve the exact derived CSV emitted by the reproduction."""
    return source.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--ciz-metadata", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    stale = []
    for source_name, target_name in MAPPINGS.items():
        source = args.output_dir / source_name
        if not source.exists():
            missing.append(source_name)
            continue
        target = args.report_dir / target_name
        expected = expected_bytes(source)
        if args.check:
            if not target.exists() or target.read_bytes() != expected:
                stale.append(target_name)
        else:
            target.write_bytes(expected)

    if args.ciz_metadata:
        metadata = json.loads(args.ciz_metadata.read_text(encoding="utf-8"))
        selected = pd.DataFrame(
            [
                {
                    "source_product": metadata["source_product"],
                    "daily_source_table": metadata["daily_source_table"],
                    "terminal_source_table": metadata["source_table"],
                    "terminal_event_date_field": metadata[
                        "terminal_event_date_field"
                    ],
                    "terminal_return_field": metadata["terminal_return_field"],
                    "return_semantics": metadata["return_semantics"],
                    "n_terminal_rows": metadata["n_rows"],
                    "n_available_terminal_returns": metadata[
                        "n_available_returns"
                    ],
                    "n_missing_terminal_returns": metadata["n_missing_returns"],
                    "terminal_extract_sha256": metadata["extract_sha256"],
                }
            ]
        ).to_csv(index=False).encode()
        target = args.report_dir / "ciz_source_validation.csv"
        if args.check:
            if not target.exists() or target.read_bytes() != selected:
                stale.append(target.name)
        else:
            target.write_bytes(selected)
    elif not args.check:
        missing.append("--ciz-metadata")

    if missing:
        raise SystemExit(f"missing corrected inputs: {', '.join(missing)}")
    if stale:
        raise SystemExit(f"stale corrected reports: {', '.join(stale)}")
    verify_corrected_reports(args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
sighting_log.py - Append sighting records to the local sightings.log file.

This module only writes to disk.  All terminal output lives in main.py.

Format matches the existing sightings.log so historical entries stay readable:

  2026-03-31 09:52:00
  Spinus tristis (American Goldfinch)
  Spinus tristis (American Goldfinch) (99.99%)

  Top 3 predictions:
  1. Spinus tristis (American Goldfinch): 99.99%
  2. Cardellina pusilla (Wilson's Warbler): 0.00%
  3. Pitangus sulphuratus (Great Kiskadee): 0.01%

  **********************************************************************

Public API
----------
  log_sighting(label, confidence, top_3, timestamp=None)
"""

import os
from datetime import datetime
from pathlib import Path

import config


def log_sighting(
    label: str,
    confidence: float,
    top_3: list[tuple[str, float]],
    timestamp: datetime | None = None,
) -> None:
    """
    Append one sighting record to config.LOG_FILE.

    Args:
        label      : winning label string (e.g. 'Spinus tristis (American Goldfinch)')
        confidence : winning confidence (0.0 - 1.0)
        top_3      : list of up to 3 (label, confidence) tuples
        timestamp  : datetime of the sighting; defaults to now
    """
    Path(config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    ts  = timestamp or datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        ts_str,
        label,
        f"{label} ({confidence:.2%})",
        "",
        "Top 3 predictions:",
    ]
    for i, (lbl, conf) in enumerate(top_3, start=1):
        lines.append(f"{i}. {lbl}: {conf:.2%}")

    lines.append("")
    lines.append("*" * 70)

    with open(config.LOG_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")

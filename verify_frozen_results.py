#!/usr/bin/env python3
"""Verify the original archived files using SHA-256; no training imports."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "FROZEN_RESULTS_SHA256.json").read_text())
    failures = []
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Changed: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest)} original files: all SHA-256 hashes match.")


if __name__ == "__main__":
    main()

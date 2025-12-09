#!/usr/bin/env python
"""
chain_stats.py

Compute coverage, storage, and verification stats for SlideChain.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

from slidechain_core import load_ledger, verify_ledger


def count_provenance_slides(root: Path) -> int:
    """Count number of Slide*.json files under provenance/Lecture X."""
    return len(list(root.glob("Lecture */Slide*.json")))


def get_file_size(path: Path) -> int:
    """Return file size in bytes if exists, else 0."""
    if not path.exists():
        return 0
    return path.stat().st_size


def get_dir_size(root: Path) -> int:
    """Return total size (in bytes) of all files under a directory."""
    total = 0
    if not root.exists():
        return 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            total += p.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance-root",
        type=str,
        default="provenance",
        help="Root directory containing provenance JSON files.",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default="slidechain_ledger.json",
        help="Path to ledger JSON file.",
    )
    args = parser.parse_args()

    provenance_root = Path(args.provenance_root)
    ledger_path = Path(args.ledger)

    num_prov_slides = count_provenance_slides(provenance_root)
    print(f"Total provenance slides: {num_prov_slides}")

    blocks = load_ledger(ledger_path)
    num_blocks = len(blocks)
    print(f"Blocks in ledger: {num_blocks}")

    # Coverage (how many provenance slides are on-chain)
    coverage = num_blocks / max(1, num_prov_slides)
    print(f"Coverage: {coverage:.3f}")

    # Storage sizes
    prov_size = get_dir_size(provenance_root)
    ledger_size = get_file_size(ledger_path)

    print(f"Provenance size: {prov_size / (1024**2):.3f} MB")
    print(f"Ledger size:     {ledger_size / (1024**2):.6f} MB")

    overhead_ratio = ledger_size / max(1, prov_size)
    print(f"Ledger storage overhead (ledger / provenance): {overhead_ratio:.6f}")

    # Verification time
    t0 = time.perf_counter()
    summary = verify_ledger(blocks)
    t1 = time.perf_counter()
    print(f"verify_ledger time: {(t1 - t0)*1000:.2f} ms")
    print(f"Ledger OK: {summary['ok']}")
    if not summary["ok"]:
        for msg in summary["messages"]:
            print("  -", msg)


if __name__ == "__main__":
    main()

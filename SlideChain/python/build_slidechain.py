#!/usr/bin/env python
"""
build_slidechain.py

Build/extend the SlideChain ledger from provenance JSON files.

Assumes provenance directory structure:

provenance/
  Lecture 1/
    Slide1.json
    Slide2.json
    ...
  Lecture 2/
    Slide1.json
    ...

Usage:
    python build_slidechain.py --provenance-root provenance --ledger slidechain_ledger.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from slidechain_core import (
    Block,
    load_ledger,
    save_ledger,
    get_last_hash,
    compute_file_hash,
    verify_ledger,
)


def find_provenance_files(root: Path) -> List[Path]:
    """Return a sorted list of all provenance JSON files under root."""
    files = list(root.glob("Lecture */Slide*.json"))
    # Sort by lecture number then slide number
    def sort_key(p: Path):
        # Expect folder name like "Lecture 1"
        lecture_part = p.parent.name.replace("Lecture", "").strip()
        slide_part = p.stem.replace("Slide", "").strip()
        try:
            lec = int(lecture_part)
        except ValueError:
            lec = 0
        try:
            slid = int(slide_part)
        except ValueError:
            slid = 0
        return (lec, slid)

    files.sort(key=sort_key)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance-root",
        type=str,
        default="provenance",
        help="Root directory containing provenance/Lecture X/SlideY.json",
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

    if not provenance_root.exists():
        raise SystemExit(
            f"Provenance root not found: {provenance_root.absolute()}"
        )

    # Load existing ledger (if any)
    blocks = load_ledger(ledger_path)
    existing_keys = {(b.lecture, b.slide_id) for b in blocks}
    print(
        f"Loaded existing ledger with {len(blocks)} block(s) "
        f"from {ledger_path}"
    )

    files = find_provenance_files(provenance_root)
    print(f"Found {len(files)} provenance files under {provenance_root}")

    next_index = len(blocks)
    prev_hash = get_last_hash(blocks)
    new_blocks = 0

    for path in files:
        # Lecture and slide_id from path
        lecture = path.parent.name  # e.g. "Lecture 1"
        slide_id = path.stem        # e.g. "Slide1"
        key = (lecture, slide_id)

        if key in existing_keys:
            # Already committed
            continue

        data_hash = compute_file_hash(path)
        # Store provenance_path relative to provenance_root for portability
        provenance_rel = str(path.relative_to(provenance_root))

        b = Block.make(
            index=next_index,
            lecture=lecture,
            slide_id=slide_id,
            provenance_path=provenance_rel,
            data_hash=data_hash,
            prev_hash=prev_hash,
        )
        blocks.append(b)
        existing_keys.add(key)
        prev_hash = b.block_hash
        next_index += 1
        new_blocks += 1

    save_ledger(blocks, ledger_path)
    summary = verify_ledger(blocks)

    print(f"Appended {new_blocks} new block(s).")
    print(f"Total blocks: {summary['num_blocks']}")
    print(f"Ledger OK: {summary['ok']}")
    if not summary["ok"]:
        for msg in summary["messages"]:
            print("  -", msg)


if __name__ == "__main__":
    main()

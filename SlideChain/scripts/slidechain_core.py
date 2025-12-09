#!/usr/bin/env python
"""
slidechain_core.py

Core SlideChain data structures and helper functions.

Each block commits to a single slide's provenance JSON via SHA256,
and is chained via prev_hash to form an append-only ledger.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# Where to store the ledger
LEDGER_PATH = Path("slidechain_ledger.json")


def canonical_dumps(obj: Any) -> str:
    """Canonical JSON serialization (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute SHA-256 hash of an arbitrary JSON-serializable object."""
    payload = canonical_dumps(obj).encode("utf-8")
    return sha256_bytes(payload)


@dataclass
class Block:
    index: int
    timestamp: float
    lecture: str
    slide_id: str
    provenance_path: str  # path to provenance JSON (relative or absolute)
    data_hash: str        # sha256 of provenance JSON file contents
    prev_hash: str
    block_hash: str       # sha256 of block header (excluding block_hash)

    @staticmethod
    def make(
        index: int,
        lecture: str,
        slide_id: str,
        provenance_path: str,
        data_hash: str,
        prev_hash: str,
        timestamp: Optional[float] = None,
    ) -> "Block":
        """Create a new Block and compute its block_hash."""
        if timestamp is None:
            timestamp = time.time()
        header = {
            "index": index,
            "timestamp": timestamp,
            "lecture": lecture,
            "slide_id": slide_id,
            "provenance_path": provenance_path,
            "data_hash": data_hash,
            "prev_hash": prev_hash,
        }
        block_hash = sha256_json(header)
        return Block(
            index=index,
            timestamp=timestamp,
            lecture=lecture,
            slide_id=slide_id,
            provenance_path=provenance_path,
            data_hash=data_hash,
            prev_hash=prev_hash,
            block_hash=block_hash,
        )


def load_ledger(path: Path = LEDGER_PATH) -> List[Block]:
    """Load an existing ledger from disk, or return an empty list if missing."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    blocks: List[Block] = []
    for b in raw:
        blocks.append(Block(**b))
    return blocks


def save_ledger(blocks: List[Block], path: Path = LEDGER_PATH) -> None:
    """Save ledger (list of blocks) to disk as JSON."""
    serializable = [asdict(b) for b in blocks]
    path.write_text(canonical_dumps(serializable) + "\n", encoding="utf-8")


def get_last_hash(blocks: List[Block]) -> str:
    """Get the prev_hash to use for the next block."""
    if not blocks:
        # Genesis prev_hash can be all zeros
        return "0" * 64
    return blocks[-1].block_hash


def build_index(blocks: List[Block]) -> Dict[tuple, Block]:
    """
    Build a mapping from (lecture, slide_id) -> Block.
    Useful for fast lookup in experiments.
    """
    index: Dict[tuple, Block] = {}
    for b in blocks:
        key = (b.lecture, b.slide_id)
        index[key] = b
    return index


def verify_ledger(blocks: List[Block]) -> Dict[str, Any]:
    """
    Verify internal consistency of the ledger.

    Checks:
    - Chain structure (prev_hash and block_hash correctness)
    - No duplicate (lecture, slide_id) entries

    Returns a dict with:
    - ok (bool)
    - num_blocks (int)
    - num_duplicates (int)
    - chain_ok (bool)
    - messages (list of str)
    """
    messages: List[str] = []
    chain_ok = True
    seen_keys: Dict[tuple, int] = {}
    num_duplicates = 0

    for i, b in enumerate(blocks):
        # Check index order
        if b.index != i:
            chain_ok = False
            messages.append(f"Index mismatch at position {i}: block.index={b.index}")

        # Recompute block hash
        header = {
            "index": b.index,
            "timestamp": b.timestamp,
            "lecture": b.lecture,
            "slide_id": b.slide_id,
            "provenance_path": b.provenance_path,
            "data_hash": b.data_hash,
            "prev_hash": b.prev_hash,
        }
        recomputed = sha256_json(header)
        if recomputed != b.block_hash:
            chain_ok = False
            messages.append(f"Block hash mismatch at index {b.index}")

        # Check prev_hash linkage
        if i == 0:
            # Genesis prev_hash is allowed to be anything fixed; we use 0*64
            if b.prev_hash != "0" * 64:
                chain_ok = False
                messages.append("Genesis block prev_hash is not zeros")
        else:
            if b.prev_hash != blocks[i - 1].block_hash:
                chain_ok = False
                messages.append(
                    f"prev_hash mismatch at index {b.index}: "
                    f"expected blocks[{i-1}].block_hash"
                )

        # Duplicate (lecture, slide_id)
        key = (b.lecture, b.slide_id)
        if key in seen_keys:
            num_duplicates += 1
            messages.append(
                f"Duplicate entry for lecture={b.lecture}, slide={b.slide_id}"
            )
        else:
            seen_keys[key] = i

    ok = chain_ok and (num_duplicates == 0)
    return {
        "ok": ok,
        "num_blocks": len(blocks),
        "num_duplicates": num_duplicates,
        "chain_ok": chain_ok,
        "messages": messages,
    }


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of the raw file contents."""
    data = path.read_bytes()
    return sha256_bytes(data)

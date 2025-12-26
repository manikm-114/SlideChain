import hashlib
import json
from typing import Any, Dict


def canonical_json(data: Dict[str, Any]) -> str:
    """Deterministic JSON serialization (sorted keys, no spaces)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes32(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()

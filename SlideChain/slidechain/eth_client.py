from pathlib import Path
from typing import Any, Dict, Tuple

import json
from web3 import Web3
from eth_account import Account

from .config import RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS, CONTRACT_ABI_PATH
from .hashing import canonical_json, sha256_bytes32


def load_contract(w3: Web3):
    abi = json.loads(CONTRACT_ABI_PATH.read_text(encoding="utf-8"))
    return w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)


def get_w3_and_account() -> Tuple[Web3, Any]:
    if not PRIVATE_KEY:
        raise RuntimeError("SLIDECHAIN_PRIVATE_KEY not set in .env")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    acct = Account.from_key(PRIVATE_KEY)
    return w3, acct


def compute_record_hash_bytes(record: Dict[str, Any]) -> bytes:
    """Compute bytes32 hash used on-chain from canonical JSON without record_hash_hex."""
    record = dict(record)
    record["record_hash_hex"] = None
    cj = canonical_json(record)
    return sha256_bytes32(cj)

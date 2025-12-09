# eth_client.py  (FULLY FIXED VERSION)

from pathlib import Path
from web3 import Web3
import json
from typing import Any

from config import (
    RPC_URL,
    ABI_PATH,
    ADDRESS_PATH,
    ACCOUNT_INDEX,
    GAS_LIMIT,
)


class SlideChainClient:
    def __init__(self):
        # ------------------------------------------
        # 1. Connect to Hardhat local blockchain
        # ------------------------------------------
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))

        if not self.w3.is_connected():
            raise RuntimeError(f"[FATAL] Could not connect to RPC: {RPC_URL}")

        # ------------------------------------------
        # 2. Load ABI + contract address
        # ------------------------------------------
        if not ABI_PATH.exists():
            raise FileNotFoundError(f"ABI file not found at: {ABI_PATH}")

        if not ADDRESS_PATH.exists():
            raise FileNotFoundError(f"Contract address file not found: {ADDRESS_PATH}")

        abi = json.loads(ABI_PATH.read_text(encoding="utf-8"))
        address = ADDRESS_PATH.read_text(encoding="utf-8").strip()

        self.contract = self.w3.eth.contract(address=address, abi=abi)

        # ------------------------------------------
        # 3. Use first Hardhat account (unlocked)
        # ------------------------------------------
        accounts = self.w3.eth.accounts
        if len(accounts) == 0:
            raise RuntimeError("[FATAL] No unlocked accounts in Hardhat node")

        self.account = accounts[ACCOUNT_INDEX]
        print(f"[INFO] Using account: {self.account}")

    # =============================================================
    # Helper to enforce correct argument types (CRITICAL FIX)
    # =============================================================

    @staticmethod
    def _ensure_int(x: Any) -> int:
        """Force all lecture_id / slide_id to real integers."""
        if isinstance(x, int):
            return x
        if isinstance(x, str) and x.isdigit():
            return int(x)
        raise TypeError(f"Expected int for lecture/slide id, got: {x} ({type(x)})")

    # =============================================================
    # Contract wrappers (FIXED)
    # =============================================================

    def is_registered(self, lecture_id: Any, slide_id: Any) -> bool:
        """Safe version — prevents 0-byte return errors."""
        L = self._ensure_int(lecture_id)
        S = self._ensure_int(slide_id)

        try:
            result = self.contract.functions.isRegistered(L, S).call()
            return bool(result)
        except Exception as e:
            print(f"[ERROR] is_registered({L},{S}) failed: {e}")
            return False

    def get_slide(self, lecture_id: Any, slide_id: Any):
        """Return full SlideRecord tuple or None."""
        L = self._ensure_int(lecture_id)
        S = self._ensure_int(slide_id)

        try:
            rec = self.contract.functions.getSlide(L, S).call()

            # rec is a tuple:
            # (lectureId, slideId, slideHash, uri, timestamp, registrant)

            if rec[4] == 0:
                # Timestamp == 0 → never registered
                return None

            return {
                "lecture_id": rec[0],
                "slide_id": rec[1],
                "slide_hash": rec[2],
                "uri": rec[3],
                "timestamp": rec[4],
                "registrant": rec[5],
            }

        except Exception as e:
            print(f"[ERROR] get_slide({L},{S}) failed: {e}")
            return None

    # =============================================================
    # Registration
    # =============================================================

    def register_slide_simple(
        self,
        lecture_id: Any,
        slide_id: Any,
        slide_hash: str,
        uri: str,
    ) -> str:

        L = self._ensure_int(lecture_id)
        S = self._ensure_int(slide_id)

        tx_hash = self.contract.functions.registerSlide(
            L, S, slide_hash, uri
        ).transact({"from": self.account, "gas": GAS_LIMIT})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        print(
            f"[OK] Registered L{L} S{S}  "
            f"(block={receipt.blockNumber}, gas={receipt.gasUsed})"
        )

        return tx_hash.hex()

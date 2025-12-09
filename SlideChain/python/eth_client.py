from pathlib import Path
from web3 import Web3
from eth_account import Account
from typing import Tuple, Any
import json

from config import RPC_URL, ABI_PATH, ADDRESS_PATH, ACCOUNT_INDEX, GAS_LIMIT

class SlideChainClient:
    def __init__(self):
        # Connect to local node
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        if not self.w3.is_connected():
            raise RuntimeError(f"Web3 not connected to {RPC_URL}")

        # Load ABI + address
        if not ABI_PATH.exists():
            raise FileNotFoundError(f"ABI file not found: {ABI_PATH}")
        if not ADDRESS_PATH.exists():
            raise FileNotFoundError(f"Address file not found: {ADDRESS_PATH}")

        abi = json.loads(ABI_PATH.read_text(encoding="utf-8"))
        address = ADDRESS_PATH.read_text(encoding="utf-8").strip()

        self.contract = self.w3.eth.contract(address=address, abi=abi)

        # Use Hardhat unlocked accounts
        accounts = self.w3.eth.accounts
        if len(accounts) == 0:
            raise RuntimeError("No accounts available from RPC. Is Hardhat node running?")
        self.account = accounts[ACCOUNT_INDEX]
        print(f"Using account: {self.account}")

    # -------- Contract wrappers -------- #

    def is_registered(self, lecture_id: int, slide_id: int) -> bool:
        return self.contract.functions.isRegistered(lecture_id, slide_id).call()

    def get_slide(self, lecture_id: int, slide_id: int) -> Any:
        return self.contract.functions.getSlide(lecture_id, slide_id).call()

    def register_slide(
        self,
        lecture_id: int,
        slide_id: int,
        slide_hash: str,
        uri: str,
    ) -> str:
        """
        Calls registerSlide(...) and returns tx hash.
        """
        tx = self.contract.functions.registerSlide(
            lecture_id,
            slide_id,
            slide_hash,
            uri,
        ).build_transaction(
            {
                "from": self.account,
                "nonce": self.w3.eth.get_transaction_count(self.account),
                "gas": GAS_LIMIT,
            }
        )

        signed = self.w3.eth.account.sign_transaction(tx, private_key=None)
        # For Hardhat, we can send unsigned tx directly using 'from' account
        # So instead, let's send using send_transaction with 'from' only:

        # Simpler: ignore manual building; just send transact:
        # But we keep this wrapper structure. So override:
        raise NotImplementedError("Use register_slide_simple instead")

    def register_slide_simple(
        self,
        lecture_id: int,
        slide_id: int,
        slide_hash: str,
        uri: str,
    ) -> str:
        """
        Simpler version using unlocked account (Hardhat, Anvil).
        """
        tx_hash = self.contract.functions.registerSlide(
            lecture_id,
            slide_id,
            slide_hash,
            uri,
        ).transact({"from": self.account, "gas": GAS_LIMIT})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(
            f"[OK] Registered lecture {lecture_id}, slide {slide_id} "
            f"(gas used={receipt.gasUsed}, block={receipt.blockNumber})"
        )
        return tx_hash.hex()

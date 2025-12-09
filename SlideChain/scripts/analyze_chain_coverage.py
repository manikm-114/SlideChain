import csv
import json
from pathlib import Path

from eth_hash.auto import keccak
from web3 import Web3

from config import RPC_URL, ABI_PATH, ADDRESS_PATH, PROVENANCE_ROOT


def keccak256_hex(data: bytes) -> str:
    """Return 0x-prefixed keccak256 hash of raw bytes."""
    return "0x" + keccak(data).hex()


def find_all_local_slides(provenance_root: Path):
    """Scan provenance directory for all slide JSON files."""
    records = []

    for lecture_dir in sorted(provenance_root.iterdir()):
        if not lecture_dir.is_dir():
            continue

        # Expect folder names like "Lecture 1", "Lecture 2", ...
        name = lecture_dir.name
        if not name.lower().startswith("lecture"):
            continue

        try:
            lecture_id = int(name.split()[1])
        except Exception:
            continue

        for slide_path in sorted(lecture_dir.glob("Slide*.json")):
            if not slide_path.is_file():
                continue

            stem = slide_path.stem  # e.g., "Slide23"
            try:
                slide_id = int(stem.replace("Slide", ""))
            except Exception:
                continue

            # Hash MUST match what was used at registration time: raw bytes of the JSON file
            data = slide_path.read_bytes()
            local_hash = keccak256_hex(data)

            records.append((lecture_id, slide_id, slide_path, local_hash))

    return records


def analyze_chain():
    # ---------- Load ABI + address ----------
    print(f"[DEBUG] Loading ABI from: {ABI_PATH}")
    if not ABI_PATH.exists():
        raise FileNotFoundError(f"ABI file not found at: {ABI_PATH}")

    abi = json.loads(ABI_PATH.read_text(encoding="utf-8"))

    if not ADDRESS_PATH.exists():
        raise FileNotFoundError(f"Contract address file not found at: {ADDRESS_PATH}")

    contract_address = ADDRESS_PATH.read_text(encoding="utf-8").strip()

    # ---------- Connect to RPC ----------
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise RuntimeError(f"Could not connect to RPC: {RPC_URL}")

    print(f"[OK] Connected to RPC: {RPC_URL}")

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=abi,
    )

    print(f"[OK] Contract loaded at: {contract_address}")
    print(f"[INFO] Provenance root: {PROVENANCE_ROOT}")

    # ---------- Scan local provenance ----------
    print("\n[INFO] Scanning provenance directory…")
    records = find_all_local_slides(PROVENANCE_ROOT)
    print(f"[INFO] Found {len(records)} local slide JSON files.")

    # ---------- Check on-chain coverage ----------
    from pathlib import Path as _Path
    out_dir = _Path("chain_results")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "chain_coverage.csv"

    print("[INFO] Checking on-chain coverage…")

    total = 0
    missing = 0
    mismatches = 0
    matches = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lecture", "slide", "status", "local_hash", "onchain_hash"])

        for lecture_id, slide_id, slide_path, local_hash in records:
            total += 1

            try:
                # ✅ Correct way: ask the contract if this slide is registered
                onchain_registered = contract.functions.isRegistered(
                    lecture_id, slide_id
                ).call()
            except Exception as e:
                writer.writerow(
                    [lecture_id, slide_id, f"error_isRegistered: {e}", local_hash, ""]
                )
                continue

            if not onchain_registered:
                missing += 1
                writer.writerow([lecture_id, slide_id, "missing", local_hash, ""])
                continue

            # If registered, fetch full record to compare hashes
            try:
                rec = contract.functions.getSlide(lecture_id, slide_id).call()
                # struct SlideRecord {
                #   uint256 lectureId;   // index 0
                #   uint256 slideId;     // index 1
                #   string slideHash;    // index 2
                #   string uri;          // index 3
                #   uint256 timestamp;   // index 4
                #   address registrant;  // index 5
                # }
                onchain_hash = rec[2]
            except Exception as e:
                writer.writerow(
                    [lecture_id, slide_id, f"error_getSlide: {e}", local_hash, ""]
                )
                continue

            if onchain_hash == local_hash:
                matches += 1
                writer.writerow(
                    [lecture_id, slide_id, "match", local_hash, onchain_hash]
                )
            else:
                mismatches += 1
                writer.writerow(
                    [lecture_id, slide_id, "mismatch", local_hash, onchain_hash]
                )

    print(f"[OK] Wrote: {out_csv}")

    # ---------- Summary ----------
    print("\n=== Chain Coverage Summary ===")
    print("Total slides:", total)
    print("Registered:", matches)
    print("Missing:", missing)
    print("Hash mismatches:", mismatches)
    print("==============================\n")


if __name__ == "__main__":
    analyze_chain()

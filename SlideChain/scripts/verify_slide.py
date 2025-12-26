import sys
import json
from pathlib import Path

from slidechain.config import PROVENANCE_ROOT
from slidechain.eth_client import get_w3_and_account, load_contract, compute_record_hash_bytes


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/verify_slide.py \"Lecture 1\" Slide1")
        sys.exit(1)

    lecture_id = sys.argv[1]
    slide_id = sys.argv[2]

    prov_path = PROVENANCE_ROOT / lecture_id.replace(" ", "_") / f"{slide_id}.json"
    if not prov_path.exists():
        print(f"Provenance file not found: {prov_path}")
        sys.exit(1)

    record = json.loads(prov_path.read_text(encoding="utf-8"))
    record_hash_bytes = compute_record_hash_bytes(record)

    w3, _ = get_w3_and_account()
    contract = load_contract(w3)

    exists = contract.functions.hasRecord(record_hash_bytes).call()
    if not exists:
        print("❌ Slide NOT registered on SlideChain.")
        sys.exit(0)

    dataset_id, lec, slid, model_count, timestamp, submitter = contract.functions.getRecord(
        record_hash_bytes
    ).call()

    print("✅ Slide IS registered on SlideChain.")
    print(f"  dataset_id : {dataset_id}")
    print(f"  lecture_id : {lec}")
    print(f"  slide_id   : {slid}")
    print(f"  models     : {model_count}")
    print(f"  timestamp  : {timestamp}")
    print(f"  submitter  : {submitter}")
    print()
    print(f"Local provenance file: {prov_path}")


if __name__ == "__main__":
    main()

import os
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from statistics import mean, median

from web3 import Web3

# --------------------------
# CONFIG
# --------------------------
RPC_URL = "http://127.0.0.1:8545"
SCRIPT_DIR = Path(__file__).resolve().parent
ABI_PATH = SCRIPT_DIR / "SlideChain_abi.json"
ADDRESS_PATH = SCRIPT_DIR / "SlideChain_address.txt"

PROV_ROOT = SCRIPT_DIR.parent / "provenance"
OUT_DIR = SCRIPT_DIR.parent / "chain_results"
OUT_DIR.mkdir(exist_ok=True)

# --------------------------
# LOAD WEB3 + CONTRACT
# --------------------------
if not ABI_PATH.exists():
    raise FileNotFoundError(f"ABI not found: {ABI_PATH}")

if not ADDRESS_PATH.exists():
    raise FileNotFoundError(f"Contract address not found: {ADDRESS_PATH}")

ABI = json.loads(ABI_PATH.read_text())
CONTRACT_ADDRESS = ADDRESS_PATH.read_text().strip()

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 not connected")

contract = w3.eth.contract(
    address=w3.to_checksum_address(CONTRACT_ADDRESS),
    abi=ABI
)

print("[OK] Connected to chain and loaded contract.")
print("[INFO] Provenance folder:", PROV_ROOT)


# ============================================================
# B1 — TIME GAP ANALYSIS
# ============================================================
def compute_time_gaps():
    rows = []
    for lecture_dir in sorted(PROV_ROOT.iterdir()):
        if not lecture_dir.is_dir():
            continue

        lecture_id = int(lecture_dir.name.replace("Lecture ", ""))

        for file in sorted(lecture_dir.glob("Slide*.json")):
            slide_id = int(file.stem.replace("Slide", ""))

            # local file modification time
            mtime = file.stat().st_mtime

            # on-chain record
            record = contract.functions.getSlide(lecture_id, slide_id).call()
            reg_ts = record[4]  # timestamp field

            if reg_ts == 0:
                continue

            time_gap = reg_ts - mtime
            rows.append((lecture_id, slide_id, mtime, reg_ts, time_gap))

    # write csv
    out_csv = OUT_DIR / "integrity_timegap.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("lecture,slide,local_mtime,chain_timestamp,time_gap_sec\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"[OK] Time-gap CSV written → {out_csv}")

    gaps = [r[4] for r in rows]
    if gaps:
        print("\n=== Time Gap Summary ===")
        print("Min:", min(gaps))
        print("Max:", max(gaps))
        print("Mean:", mean(gaps))
        print("Median:", median(gaps))
        print("========================\n")


# ============================================================
# B2 — TAMPER DETECTION EXPERIMENT
# ============================================================
def slight_corruption(original_json: str) -> str:
    """Modifies one character → valid corruption."""
    if len(original_json) < 10:
        return original_json[::-1]  # fallback: reverse short string

    pos = len(original_json) // 2
    corrupted = (
        original_json[:pos] +
        ("X" if original_json[pos] != "X" else "Y") +
        original_json[pos+1:]
    )
    return corrupted


def run_tamper_test(sample_size: int = 10):
    print(f"[INFO] Running tamper detection test on {sample_size} slides...")

    all_slides = []
    for lecture_dir in PROV_ROOT.iterdir():
        if lecture_dir.is_dir():
            for file in lecture_dir.glob("Slide*.json"):
                all_slides.append(file)

    sample = random.sample(all_slides, min(sample_size, len(all_slides)))

    rows = []

    for file in sample:
        lecture_id = int(file.parent.name.replace("Lecture ", ""))
        slide_id = int(file.stem.replace("Slide", ""))

        # load raw bytes exactly as the registration script
        data = file.read_bytes()
        good_hash = "0x" + Web3.keccak(data).hex()[2:]

        # corrupt raw bytes
        corrupted_bytes = bytearray(data)
        pos = len(corrupted_bytes) // 2
        corrupted_bytes[pos] = (corrupted_bytes[pos] + 1) % 256  # change 1 byte

        corrupted_hash = "0x" + Web3.keccak(bytes(corrupted_bytes)).hex()[2:]

        # blockchain
        record = contract.functions.getSlide(lecture_id, slide_id).call()
        chain_hash = record[2]

        match_good = (chain_hash == good_hash)
        match_bad = (chain_hash == corrupted_hash)

        rows.append([
            lecture_id, slide_id,
            good_hash, chain_hash, match_good,
            corrupted_hash, match_bad
        ])

    out_csv = OUT_DIR / "tamper_detection.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("lecture,slide,good_hash,chain_hash,match_good,corrupted_hash,match_corrupted\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"[OK] Tamper detection CSV → {out_csv}")

    detected = sum(1 for r in rows if (not r[6]))
    print("\n=== Tamper Detection Summary ===")
    print(f"Slides tested: {len(rows)}")
    print(f"Corruptions correctly detected: {detected}/{len(rows)}")
    print("===============================\n")

# ============================================================
# B3 — HASH COLLISION PROBABILITY ESTIMATION
# ============================================================
def print_collision_probability():
    n = 1117  # slides
    bits = 256

    # birthday bound approximation
    # P ≈ n^2 / 2^(bits+1)
    approx_p = (n * n) / (2 ** (bits + 1))

    print("\n=== Hash Collision Probability (Theoretical) ===")
    print("Hash function: keccak256 (256-bit)")
    print(f"Slides hashed: {n}")
    print(f"Birthday-bound collision probability ≈ {approx_p:.3e}")
    print("This is effectively zero for all practical purposes.")
    print("===============================================\n")


# ============================================================
# MAIN
# ============================================================
def main():
    compute_time_gaps()
    run_tamper_test(sample_size=20)
    print_collision_probability()


if __name__ == "__main__":
    main()

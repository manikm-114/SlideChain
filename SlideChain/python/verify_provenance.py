import csv
from pathlib import Path
import re
from typing import Optional

from config import BY_SLIDE_ROOT
from eth_client import SlideChainClient

from eth_hash.auto import keccak

LECTURE_RE = re.compile(r"Lecture\s+(\d+)", re.IGNORECASE)
SLIDE_RE = re.compile(r"Slide(\d+)\.json$", re.IGNORECASE)


def parse_lecture_id(lecture_dir: Path) -> int:
    m = LECTURE_RE.match(lecture_dir.name)
    if not m:
        raise ValueError(f"Cannot parse lecture id from {lecture_dir}")
    return int(m.group(1))


def parse_slide_id(slide_file: Path) -> int:
    m = SLIDE_RE.match(slide_file.name)
    if not m:
        raise ValueError(f"Cannot parse slide id from {slide_file}")
    return int(m.group(1))


def keccak256_hex(data: bytes) -> str:
    return "0x" + keccak(data).hex()


def main():
    client = SlideChainClient()

    if not BY_SLIDE_ROOT.exists():
        raise FileNotFoundError(f"BY_SLIDE_ROOT does not exist: {BY_SLIDE_ROOT}")

    output_csv = BY_SLIDE_ROOT / "slidechain_provenance_report.csv"
    rows = []

    lectures = sorted(
        [p for p in BY_SLIDE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: parse_lecture_id(p),
    )

    total = 0
    ok = 0
    mismatch = 0
    missing = 0

    for lecture_dir in lectures:
        lecture_id = parse_lecture_id(lecture_dir)
        slide_files = sorted(
            [p for p in lecture_dir.glob("Slide*.json") if p.is_file()],
            key=parse_slide_id,
        )

        print(f"\n=== Checking Lecture {lecture_id} ({lecture_dir}) ===")

        for slide_file in slide_files:
            slide_id = parse_slide_id(slide_file)
            total += 1

            # Query chain
            record = client.get_slide(lecture_id, slide_id)

            onchain_ts = record[4]  # timestamp
            onchain_hash = record[2]  # slideHash
            onchain_uri = record[3]
            registrant = record[5]

            if onchain_ts == 0:
                print(f"[MISS] L{lecture_id} S{slide_id}: not registered on chain")
                missing += 1
                rows.append(
                    {
                        "lecture_id": lecture_id,
                        "slide_id": slide_id,
                        "status": "missing",
                        "onchain_hash": "",
                        "computed_hash": "",
                        "onchain_uri": "",
                        "registrant": "",
                        "timestamp": "",
                    }
                )
                continue

            # Compute local hash
            data = slide_file.read_bytes()
            local_hash = keccak256_hex(data)

            if local_hash == onchain_hash:
                print(f"[OK]   L{lecture_id} S{slide_id}: hash matches")
                ok += 1
                status = "ok"
            else:
                print(
                    f"[MISMATCH] L{lecture_id} S{slide_id}:\n"
                    f"  onchain = {onchain_hash}\n"
                    f"  local   = {local_hash}"
                )
                mismatch += 1
                status = "mismatch"

            rows.append(
                {
                    "lecture_id": lecture_id,
                    "slide_id": slide_id,
                    "status": status,
                    "onchain_hash": onchain_hash,
                    "computed_hash": local_hash,
                    "onchain_uri": onchain_uri,
                    "registrant": registrant,
                    "timestamp": onchain_ts,
                }
            )

    print("\n=== Provenance Summary ===")
    print(f"Total slides checked: {total}")
    print(f"OK hashes:            {ok}")
    print(f"Missing on-chain:     {missing}")
    print(f"Hash mismatches:      {mismatch}")

    # Write CSV
    fieldnames = [
        "lecture_id",
        "slide_id",
        "status",
        "onchain_hash",
        "computed_hash",
        "onchain_uri",
        "registrant",
        "timestamp",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {output_csv}")


if __name__ == "__main__":
    main()

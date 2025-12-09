import re
from pathlib import Path
import hashlib
from typing import Tuple

from config import BY_SLIDE_ROOT
from eth_client import SlideChainClient

LECTURE_RE = re.compile(r"Lecture\s+(\d+)", re.IGNORECASE)
SLIDE_RE = re.compile(r"Slide(\d+)\.json$", re.IGNORECASE)


def parse_lecture_id(lecture_dir: Path) -> int:
    """
    'Lecture 1' -> 1
    """
    m = LECTURE_RE.match(lecture_dir.name)
    if not m:
        raise ValueError(f"Cannot parse lecture id from {lecture_dir}")
    return int(m.group(1))


def parse_slide_id(slide_file: Path) -> int:
    """
    'Slide23.json' -> 23
    """
    m = SLIDE_RE.match(slide_file.name)
    if not m:
        raise ValueError(f"Cannot parse slide id from {slide_file}")
    return int(m.group(1))


def keccak256_hex(data: bytes) -> str:
    """
    Returns 0x-prefixed keccak256 hash as hex string.
    Web3 has keccak; here we use its underlying implementation.
    """
    from eth_hash.auto import keccak

    return "0x" + keccak(data).hex()


def main():
    if not BY_SLIDE_ROOT.exists():
        raise FileNotFoundError(f"BY_SLIDE_ROOT does not exist: {BY_SLIDE_ROOT}")

    client = SlideChainClient()

    lectures = sorted(
        [p for p in BY_SLIDE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: parse_lecture_id(p),
    )

    total = 0
    skipped = 0
    registered = 0

    for lecture_dir in lectures:
        lecture_id = parse_lecture_id(lecture_dir)
        slide_files = sorted(
            [p for p in lecture_dir.glob("Slide*.json") if p.is_file()],
            key=parse_slide_id,
        )

        print(f"\n=== Lecture {lecture_id} ({lecture_dir}) ===")

        for slide_file in slide_files:
            slide_id = parse_slide_id(slide_file)
            uri = f"{lecture_dir.name}/{slide_file.name}"

            total += 1

            if client.is_registered(lecture_id, slide_id):
                print(f"[SKIP] Lecture {lecture_id}, Slide {slide_id} already registered")
                skipped += 1
                continue

            data = slide_file.read_bytes()
            slide_hash = keccak256_hex(data)

            try:
                client.register_slide_simple(
                    lecture_id=lecture_id,
                    slide_id=slide_id,
                    slide_hash=slide_hash,
                    uri=uri,
                )
                registered += 1
            except Exception as e:
                print(f"[ERR] Failed to register L{lecture_id} S{slide_id}: {e}")

    print("\n=== Summary ===")
    print(f"Total slides seen:     {total}")
    print(f"Already registered:    {skipped}")
    print(f"Newly registered:      {registered}")


if __name__ == "__main__":
    main()

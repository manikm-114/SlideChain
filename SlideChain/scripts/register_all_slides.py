import re
from pathlib import Path
from eth_hash.auto import keccak

from config import PROVENANCE_ROOT
from eth_client import SlideChainClient


LECTURE_RE = re.compile(r"Lecture\s+(\d+)", re.IGNORECASE)
SLIDE_RE = re.compile(r"Slide(\d+)\.json$", re.IGNORECASE)


def parse_lecture_id(lecture_dir: Path) -> int:
    m = LECTURE_RE.match(lecture_dir.name)
    if not m:
        raise ValueError(f"Cannot parse lecture number from: {lecture_dir}")
    return int(m.group(1))


def parse_slide_id(slide_file: Path) -> int:
    m = SLIDE_RE.match(slide_file.name)
    if not m:
        raise ValueError(f"Cannot parse slide number from: {slide_file}")
    return int(m.group(1))


def keccak256_hex(data: bytes) -> str:
    return "0x" + keccak(data).hex()


def main():
    prov_root = Path(PROVENANCE_ROOT)
    if not prov_root.exists():
        raise FileNotFoundError(f"PROVENANCE_ROOT does not exist: {prov_root}")

    client = SlideChainClient()

    # find lecture folders
    lectures = sorted(
        [p for p in prov_root.iterdir() if p.is_dir()],
        key=lambda p: parse_lecture_id(p)
    )

    total = 0
    skipped = 0
    registered = 0

    print("\n=== REGISTERING ALL PROVENANCE SLIDES ===")

    for lecture_dir in lectures:
        lecture_id = parse_lecture_id(lecture_dir)
        slide_files = sorted(
            [p for p in lecture_dir.glob("Slide*.json") if p.is_file()],
            key=parse_slide_id
        )

        print(f"\n--- Lecture {lecture_id} ({len(slide_files)} slides) ---")

        for slide_file in slide_files:
            slide_id = parse_slide_id(slide_file)
            total += 1

            if client.is_registered(lecture_id, slide_id):
                print(f"[SKIP] L{lecture_id} S{slide_id} already registered")
                skipped += 1
                continue

            # compute hash
            data = slide_file.read_bytes()
            slide_hash = keccak256_hex(data)

            # create on-chain URI
            uri = f"{lecture_dir.name}/{slide_file.name}"

            try:
                client.register_slide_simple(
                    lecture_id=lecture_id,
                    slide_id=slide_id,
                    slide_hash=slide_hash,
                    uri=uri,
                )
                print(f"[OK] Registered L{lecture_id} S{slide_id}")
                registered += 1
            except Exception as e:
                print(f"[ERR] Failed L{lecture_id} S{slide_id}: {e}")

    print("\n=== REGISTRATION SUMMARY ===")
    print(f"Total slides found:     {total}")
    print(f"Already registered:     {skipped}")
    print(f"Newly registered:       {registered}")
    print("==============================\n")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from slidechain.config import PROVENANCE_ROOT
from slidechain.provenance import build_provenance_from_master, save_record

# 🔧 ADJUST THIS to the folder where your combined per-slide JSONs live
MASTER_ROOT = Path(r"G:\MILU23_master")  # example; change as needed


def iter_master_jsons():
    for lecture_dir in MASTER_ROOT.iterdir():
        if not lecture_dir.is_dir():
            continue
        for slide_json in sorted(lecture_dir.glob("Slide*.json")):
            yield slide_json


def main():
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)

    for master_path in iter_master_jsons():
        master_json = json.loads(master_path.read_text(encoding="utf-8"))
        record = build_provenance_from_master(master_json)
        out_path = save_record(record)
        print(f"{record.lecture_id}/{record.slide_id} -> {out_path} (hash={record.record_hash_hex})")


if __name__ == "__main__":
    main()

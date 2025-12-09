#!/usr/bin/env python
"""
tamper_experiment.py

Simulate tampering with provenance JSON and measure how often
SlideChain detects the tampering (via hash mismatch).

We:
- Load the ledger.
- For each block (or subset), load the corresponding provenance JSON.
- Create tampered versions in memory (no file changes on disk).
- Recompute the data hash and compare to ledger.data_hash.

Outputs:
- results/tamper_experiment.csv
- printed summary stats.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from slidechain_core import (
    load_ledger,
    build_index,
    sha256_json,
)


def canonical_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tamper_change_concept_category(prov: Dict[str, Any]) -> bool:
    """
    Change the category of the first concept in the first model that has one.
    Returns True if tampering was applied, False if not possible.
    """
    models = prov.get("models", {})
    for model_name, mdata in models.items():
        concepts = mdata.get("concepts", {})
        parsed = concepts.get("parsed")
        # parsed can be a dict or dict with 'concepts' list
        if isinstance(parsed, dict) and "category" in parsed:
            parsed["category"] = parsed["category"] + "_tampered"
            return True
        if isinstance(parsed, dict) and "concepts" in parsed:
            clist = parsed.get("concepts") or []
            if clist and isinstance(clist[0], dict) and "category" in clist[0]:
                clist[0]["category"] = clist[0]["category"] + "_tampered"
                return True
    return False


def tamper_drop_model(prov: Dict[str, Any]) -> bool:
    """
    Drop one model entirely from the 'models' dict.
    Returns True if tampering applied, False otherwise.
    """
    models = prov.get("models", {})
    keys = list(models.keys())
    if not keys:
        return False
    drop_name = random.choice(keys)
    del models[drop_name]
    return True


def tamper_change_text_path(prov: Dict[str, Any]) -> bool:
    """
    Modify the 'paths.text' field by appending a bogus suffix.
    """
    paths = prov.get("paths")
    if not isinstance(paths, dict) or "text" not in paths:
        return False
    paths["text"] = str(paths["text"]) + "_tampered"
    return True


TAMPER_FUNCTIONS = {
    "change_concept_category": tamper_change_concept_category,
    "drop_model": tamper_drop_model,
    "change_text_path": tamper_change_text_path,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance-root",
        type=str,
        default="provenance",
        help="Root directory of provenance/Lecture X/SlideY.json",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default="slidechain_ledger.json",
        help="Path to ledger JSON file.",
    )
    parser.add_argument(
        "--num-per-tamper",
        type=int,
        default=200,
        help="Number of (lecture, slide) samples per tamper type. "
             "If larger than available, will just use all.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/tamper_experiment.csv",
        help="Where to save the CSV with tamper results.",
    )
    args = parser.parse_args()

    provenance_root = Path(args.provenance_root)
    ledger_path = Path(args.ledger)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not ledger_path.exists():
        raise SystemExit(f"Ledger not found: {ledger_path}")

    blocks = load_ledger(ledger_path)
    index = build_index(blocks)

    # Collect all keys that have a local provenance file
    available_keys: List[tuple] = []
    for (lecture, slide_id), block in index.items():
        prov_rel = Path(block.provenance_path)
        prov_path = provenance_root / prov_rel
        if prov_path.exists():
            available_keys.append((lecture, slide_id))

    if not available_keys:
        raise SystemExit("No provenance files found corresponding to ledger blocks.")

    print(f"Tamper experiment over {len(available_keys)} slide(s).")

    rows: List[Dict[str, Any]] = []

    for tamper_name, tamper_fn in TAMPER_FUNCTIONS.items():
        # Sample keys (without replacement) for this tamper type
        n = min(args.num_per_tamper, len(available_keys))
        sampled_keys = random.sample(available_keys, n)

        detected = 0
        total = 0

        for lecture, slide_id in sampled_keys:
            block = index[(lecture, slide_id)]
            prov_rel = Path(block.provenance_path)
            prov_path = provenance_root / prov_rel
            original = canonical_load(prov_path)
            tampered = copy.deepcopy(original)

            applied = tamper_fn(tampered)
            if not applied:
                # No modification possible; skip
                continue

            total += 1

            new_hash = sha256_json(tampered)
            # Detected if hash no longer matches ledger's data_hash
            is_detected = (new_hash != block.data_hash)
            if is_detected:
                detected += 1

            rows.append(
                {
                    "lecture": lecture,
                    "slide_id": slide_id,
                    "tamper_type": tamper_name,
                    "detected": int(is_detected),
                }
            )

        print(
            f"[{tamper_name}] used {total} sample(s), "
            f"detected {detected} ({detected / max(1, total):.3f})"
        )

    # Save CSV
    fieldnames = ["lecture", "slide_id", "tamper_type", "detected"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved tamper experiment results to {output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
compute_provenance.py

Reads merged per-slide JSON files from the BY_SLIDE directory and creates
a unified provenance JSON for each slide.

Input directory structure (user provided):
    by_slide/
      Lecture 1/
         Slide1.json
         Slide2.json
         ...
      Lecture 2/
         Slide1.json
         ...

Output directory structure (created here):
    provenance/
      Lecture 1/
         Slide1.json
         Slide2.json
         ...
      Lecture 2/
         ...

Each provenance JSON contains ONLY clean model outputs and paths:

{
  "lecture": "Lecture 1",
  "slide_id": "Slide1",
  "paths": {
      "image": "...",
      "text": "..."
  },
  "models": {
      "ModelName": {
           "concepts": ...,
           "triples": ...
      },
      ...
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# ===============================================================
# CONFIGURATION
# ===============================================================
BY_SLIDE_ROOT = Path("F:/Research Files/SlideChain/Lectures/by_slide")
PROVENANCE_ROOT = Path("provenance")   # Output directory


# ===============================================================
# Utility functions
# ===============================================================

def safe_load_json(path: Path) -> Dict[str, Any]:
    """Load JSON safely, returning {} if any error occurs."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def canonical_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON using canonical formatting (sorted keys, compact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)
    path.write_text(txt + "\n", encoding="utf-8")


def extract_parsed(model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the 'parsed' section for concepts and triples.
    Robust to:
        - concepts = None
        - concepts = {}
        - concepts missing entirely
        - concepts = {"parsed": None}
        - triples with any of the same variations
    """
    out = {}

    # -------- Concepts --------
    c = model_entry.get("concepts", None)
    if isinstance(c, dict):
        out["concepts"] = c.get("parsed", None)
    else:
        out["concepts"] = None

    # -------- Triples --------
    t = model_entry.get("triples", None)
    if isinstance(t, dict):
        out["triples"] = t.get("parsed", None)
    else:
        out["triples"] = None

    return out



# ===============================================================
# Main builder
# ===============================================================

def build_provenance():
    print("Building provenance from by_slide...")

    if not BY_SLIDE_ROOT.exists():
        raise SystemExit(f"BY_SLIDE_ROOT does not exist: {BY_SLIDE_ROOT}")

    count = 0

    # Traverse Lecture folders
    for lecture_dir in sorted(BY_SLIDE_ROOT.glob("Lecture *")):
        if not lecture_dir.is_dir():
            continue

        lecture_name = lecture_dir.name  # e.g. "Lecture 1"

        # Traverse Slide JSONs
        for slide_json in sorted(lecture_dir.glob("Slide*.json")):
            slide_id = slide_json.stem  # "Slide1"

            # Load the merged JSON user provided
            src = safe_load_json(slide_json)
            if not src:
                continue

            # Extract paths (image/text)
            paths = src.get("paths", {})

            # Extract per-model parsed content
            models_src = src.get("models", {})
            cleaned_models = {}

            for model_name, model_entry in models_src.items():
                cleaned_models[model_name] = extract_parsed(model_entry)

            # Build provenance JSON
            prov = {
                "lecture": lecture_name,
                "slide_id": slide_id,
                "paths": {
                    "image": paths.get("image"),
                    "text": paths.get("text"),
                },
                "models": cleaned_models,
            }

            # Output path
            out_path = PROVENANCE_ROOT / lecture_name / f"{slide_id}.json"
            canonical_write_json(out_path, prov)
            count += 1

    print(f"Provenance generated for {count} slides.")
    print(f"Output written to: {PROVENANCE_ROOT.absolute()}")


# ===============================================================
# Entry point
# ===============================================================

if __name__ == "__main__":
    build_provenance()

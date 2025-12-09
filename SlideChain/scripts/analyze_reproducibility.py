import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Set, Any, List
from statistics import mean, median

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    """
    Jaccard similarity between two sets.
    Convention:
      - if both sets are empty, return 1.0 (they agree on predicting nothing).
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


def extract_concepts(model_block: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Normalize and extract concept set from a model block.

    Returns a set of (term_lower, category_lower).
    Handles both:
      - {"concepts": [ { "term": ..., "category": ... }, ... ]}
      - {"term": ..., "category": ...}
    """
    concepts_set: Set[Tuple[str, str]] = set()
    raw = model_block.get("concepts", None)

    if raw is None:
        return concepts_set

    # Case 1: dict with 'concepts' list
    if isinstance(raw, dict):
        if isinstance(raw.get("concepts"), list):
            for c in raw["concepts"]:
                if not isinstance(c, dict):
                    continue
                term = str(c.get("term", "")).strip()
                cat = str(c.get("category", "")).strip()
                if term:
                    concepts_set.add((term.lower(), cat.lower()))
        else:
            # Single dict: {"term": "...", "category": "..."}
            term = str(raw.get("term", "")).strip()
            cat = str(raw.get("category", "")).strip()
            if term:
                concepts_set.add((term.lower(), cat.lower()))

    # Case 2: raw is already a list of dicts (rare but possible)
    elif isinstance(raw, list):
        for c in raw:
            if not isinstance(c, dict):
                continue
            term = str(c.get("term", "")).strip()
            cat = str(c.get("category", "")).strip()
            if term:
                concepts_set.add((term.lower(), cat.lower()))

    return concepts_set


def extract_triples(model_block: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    """
    Normalize and extract triple set from a model block.

    Returns a set of (s_lower, p_lower, o_lower).
    Handles:
      - {"triples": [ { "s":..., "p":..., "o":... }, ... ]}
    """
    triple_set: Set[Tuple[str, str, str]] = set()
    raw = model_block.get("triples", None)

    if raw is None:
        return triple_set

    if isinstance(raw, dict):
        items = raw.get("triples", [])
    else:
        items = raw

    if not isinstance(items, list):
        return triple_set

    for t in items:
        if not isinstance(t, dict):
            continue
        s = str(t.get("s", "")).strip()
        p = str(t.get("p", "")).strip()
        o = str(t.get("o", "")).strip()
        if s and p and o:
            triple_set.add((s.lower(), p.lower(), o.lower()))

    return triple_set


def load_run(root: Path) -> Dict[Tuple[str, str, str], Dict[str, Set]]:
    """
    Load provenance for one run.

    Returns:
      mapping[(lecture, slide_id, model_name)] = {
          "concepts": set((term, category)),
          "triples": set((s, p, o))
      }

    Expects directory structure:
      root/
        Lecture 1/Slide1.json
        Lecture 1/Slide2.json
        ...
        Lecture 23/SlideX.json
    """
    data: Dict[Tuple[str, str, str], Dict[str, Set]] = {}

    if not root.exists():
        raise FileNotFoundError(f"Run root does not exist: {root}")

    lecture_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for lec_dir in lecture_dirs:
        lecture = lec_dir.name  # e.g., "Lecture 1"

        for slide_file in sorted(lec_dir.glob("Slide*.json")):
            with slide_file.open("r", encoding="utf-8") as f:
                slide_json = json.load(f)

            slide_id = slide_json.get("slide_id", slide_file.stem)  # e.g., "Slide1"
            models = slide_json.get("models", {})

            for model_name, model_block in models.items():
                key = (lecture, slide_id, model_name)
                cset = extract_concepts(model_block or {})
                tset = extract_triples(model_block or {})
                data[key] = {"concepts": cset, "triples": tset}

    return data


def summarize_by_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate slide-level reproducibility into per-model statistics.
    """
    per_model: Dict[str, Dict[str, List[float]]] = {}

    for r in rows:
        model_name = r["model_name"]
        cj = r["concept_jaccard"]
        tj = r["triple_jaccard"]
        m = per_model.setdefault(model_name, {"concept_j": [], "triple_j": []})
        if cj is not None:
            m["concept_j"].append(cj)
        if tj is not None:
            m["triple_j"].append(tj)

    out_rows: List[Dict[str, Any]] = []

    for model_name, vals in per_model.items():
        cj_list = vals["concept_j"]
        tj_list = vals["triple_j"]

        def safe_mean(xs: List[float]) -> float:
            return mean(xs) if xs else 0.0

        def safe_median(xs: List[float]) -> float:
            return median(xs) if xs else 0.0

        out_rows.append(
            {
                "model_name": model_name,
                "num_slides": max(len(cj_list), len(tj_list)),
                "concept_jaccard_mean": safe_mean(cj_list),
                "concept_jaccard_median": safe_median(cj_list),
                "triple_jaccard_mean": safe_mean(tj_list),
                "triple_jaccard_median": safe_median(tj_list),
            }
        )

    return out_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze reproducibility between two provenance runs "
            "(concepts/triples Jaccard across runs)."
        )
    )
    parser.add_argument(
        "--run1_root",
        type=str,
        default=str(Path(__file__).parent.parent / "provenance"),
        help="Path to first provenance root (default: ../provenance)",
    )
    parser.add_argument(
        "--run2_root",
        type=str,
        default=str(Path(__file__).parent.parent / "provenance_run2"),
        help="Path to second provenance root (default: ../provenance_run2)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "analysis_results"),
        help="Directory to write CSV outputs (default: ../analysis_results)",
    )

    args = parser.parse_args()

    run1_root = Path(args.run1_root).resolve()
    run2_root = Path(args.run2_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Run 1 root:", run1_root)
    print("[INFO] Run 2 root:", run2_root)
    print("[INFO] Output dir:", out_dir)

    print("[INFO] Loading run 1 provenance...")
    run1_data = load_run(run1_root)
    print(f"[INFO] Run 1 keys: {len(run1_data)} (lecture, slide, model) combos")

    print("[INFO] Loading run 2 provenance...")
    run2_data = load_run(run2_root)
    print(f"[INFO] Run 2 keys: {len(run2_data)} (lecture, slide, model) combos")

    keys1 = set(run1_data.keys())
    keys2 = set(run2_data.keys())
    common_keys = sorted(keys1 & keys2)

    print(f"[INFO] Common (lecture, slide, model) keys across runs: {len(common_keys)}")

    slide_rows: List[Dict[str, Any]] = []

    for lecture, slide_id, model_name in common_keys:
        r1 = run1_data[(lecture, slide_id, model_name)]
        r2 = run2_data[(lecture, slide_id, model_name)]

        c1 = r1["concepts"]
        c2 = r2["concepts"]
        t1 = r1["triples"]
        t2 = r2["triples"]

        cj = jaccard(c1, c2)
        tj = jaccard(t1, t2)

        slide_rows.append(
            {
                "lecture": lecture,
                "slide_id": slide_id,
                "model_name": model_name,
                "num_concepts_run1": len(c1),
                "num_concepts_run2": len(c2),
                "num_triples_run1": len(t1),
                "num_triples_run2": len(t2),
                "concept_jaccard": cj,
                "triple_jaccard": tj,
            }
        )

    # ----------------- Write slide-level CSV -----------------
    slide_csv = out_dir / "reproducibility_slide_level.csv"
    with slide_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lecture",
                "slide_id",
                "model_name",
                "num_concepts_run1",
                "num_concepts_run2",
                "num_triples_run1",
                "num_triples_run2",
                "concept_jaccard",
                "triple_jaccard",
            ],
        )
        writer.writeheader()
        for row in slide_rows:
            writer.writerow(row)

    print(f"[OK] Wrote slide-level reproducibility -> {slide_csv}")

    # ----------------- Write per-model summary -----------------
    model_summary_rows = summarize_by_model(slide_rows)
    summary_csv = out_dir / "reproducibility_model_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "num_slides",
                "concept_jaccard_mean",
                "concept_jaccard_median",
                "triple_jaccard_mean",
                "triple_jaccard_median",
            ],
        )
        writer.writeheader()
        for row in model_summary_rows:
            writer.writerow(row)

    print(f"[OK] Wrote model-level reproducibility summary -> {summary_csv}")

    # ----------------- Quick console summary -----------------
    if model_summary_rows:
        print("\n=== Reproducibility Summary (per model) ===")
        for r in model_summary_rows:
            print(
                f"{r['model_name']}: "
                f"concept Jaccard mean={r['concept_jaccard_mean']:.3f}, "
                f"triple Jaccard mean={r['triple_jaccard_mean']:.3f} "
                f"(slides={r['num_slides']})"
            )
        print("===========================================\n")
    else:
        print("[WARN] No overlapping (lecture, slide, model) keys between runs.")


if __name__ == "__main__":
    main()

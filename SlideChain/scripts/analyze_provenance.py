import os
import json
import csv
from pathlib import Path

PROVENANCE_ROOT = Path("F:/Research Files/SlideChain/SlideChain/provenance")
OUT_DIR = Path("F:/Research Files/SlideChain/SlideChain/analysis_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_slide(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------
#  ROBUST CONCEPT EXTRACTION (never errors)
# -------------------------------------------------------
def extract_concepts(model_block):
    """
    Returns a SET of (category, term) for all models.
    Supports:
    1. {"concepts": {"concepts":[...], "evidence":[...]}}
    2. {"concepts": {"category": "...", "term": "..."}}
    3. missing concepts key
    4. concepts = None
    5. concepts = {} empty
    """

    c = model_block.get("concepts", None)
    if c is None:
        return set()

    # Case 1: multi-concept list
    if isinstance(c, dict) and "concepts" in c and isinstance(c["concepts"], list):
        out = set()
        for item in c["concepts"]:
            if isinstance(item, dict) and "category" in item and "term" in item:
                out.add((item["category"], item["term"]))
        return out

    # Case 2: single {category, term}
    if isinstance(c, dict) and "category" in c and "term" in c:
        return {(c["category"], c["term"])}

    # Nothing usable
    return set()


# -------------------------------------------------------
#  ROBUST TRIPLE EXTRACTION (never errors)
# -------------------------------------------------------
def extract_triples(model_block):
    """
    Returns a SET of (s,p,o) triples.
    Supports:
    - {"triples": {"triples":[...]} }
    - {"triples": None}
    - {"triples": {}}
    - {"triples": {"triples": []}}
    - missing triples
    """

    t = model_block.get("triples", None)
    if t is None:
        return set()

    if not isinstance(t, dict):
        return set()

    triple_list = t.get("triples", None)
    if not isinstance(triple_list, list):
        return set()

    out = set()
    for trip in triple_list:
        if isinstance(trip, dict):
            s = trip.get("s", "")
            p = trip.get("p", "")
            o = trip.get("o", "")
            out.add((s, p, o))
    return out


# -------------------------------------------------------
#  MAIN PROCESSING
# -------------------------------------------------------
def main():
    per_model_rows = []
    per_slide_rows = []

    for lecture_dir in sorted(PROVENANCE_ROOT.iterdir()):
        if not lecture_dir.is_dir():
            continue

        lecture_name = lecture_dir.name

        for slide_file in sorted(lecture_dir.glob("Slide*.json")):
            slide_json = load_slide(slide_file)
            slide_id = slide_json.get("slide_id", slide_file.stem)

            models = slide_json.get("models", {})

            all_concepts = []
            all_triples = []

            # Per-model stats
            for model_name, block in models.items():
                concepts = extract_concepts(block)
                triples = extract_triples(block)

                all_concepts.append(concepts)
                all_triples.append(triples)

                per_model_rows.append([
                    lecture_name,
                    slide_id,
                    model_name,
                    len(concepts),
                    len(triples)
                ])

            # ------------------------------
            # Compute per-slide disagreement
            # ------------------------------

            # If a slide has 0 models, avoid crash
            if len(all_concepts) == 0:
                concept_dis = 0
                triple_dis = 0
            else:
                # Concept disagreement
                union_concepts = set().union(*all_concepts)
                intersection_concepts = set(all_concepts[0])
                for c in all_concepts[1:]:
                    intersection_concepts &= c

                concept_dis = len(union_concepts) - len(intersection_concepts)

                # Triple disagreement
                union_triples = set().union(*all_triples)
                intersection_triples = set(all_triples[0])
                for t in all_triples[1:]:
                    intersection_triples &= t

                triple_dis = len(union_triples) - len(intersection_triples)

            # Append per-slide line
            per_slide_rows.append([
                lecture_name,
                slide_id,
                concept_dis,
                triple_dis
            ])

    # ------------------------------
    # WRITE OUTPUT FILES
    # ------------------------------
    with open(OUT_DIR / "slide_level_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "slide_id", "model_name", "num_concepts", "num_triples"])
        w.writerows(per_model_rows)

    with open(OUT_DIR / "slide_level_disagreement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "slide_id", "concept_disagreement", "triple_disagreement"])
        w.writerows(per_slide_rows)

    print("\n[OK] CREATED FILES:")
    print("  - slide_level_stats.csv")
    print("  - slide_level_disagreement.csv")
    print("Slides processed:", len(per_slide_rows))


if __name__ == "__main__":
    main()

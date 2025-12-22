import json
import os
import pandas as pd

# -------------------- CONFIG --------------------
LECTURES_ROOT = r"F:\Research Files\SlideChain\Lectures\by_slide"
OUTPUT_DIR = r"F:\Research Files\SlideChain\SlideChain\analysis_results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "single_vs_multi_provenance.csv")

BASELINE_MODEL = "OpenGVLab__InternVL3-14B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- HELPERS --------------------
def _safe_get(d, *keys):
    """
    Safely descend through nested dict keys.
    Returns None if any step is missing or not a dict.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def _norm_text(x):
    return str(x).strip().lower()

def extract_concepts(model_entry):
    """
    Returns a set of normalized concept terms.
    Handles:
      - parsed == None
      - parsed as dict with "concepts": [...]
      - parsed as a single concept object {"term":..., "category":...}
    """
    parsed = _safe_get(model_entry, "concepts", "parsed")

    if parsed is None:
        return set()

    # Case A: normal form: {"concepts":[{term,...}, ...], "evidence":[...]}
    if isinstance(parsed, dict) and "concepts" in parsed and isinstance(parsed["concepts"], list):
        out = set()
        for c in parsed["concepts"]:
            if isinstance(c, dict) and c.get("term"):
                out.add(_norm_text(c["term"]))
        return out

    # Case B: single concept object: {"term": "...", "category": "..."}
    if isinstance(parsed, dict) and parsed.get("term"):
        return {_norm_text(parsed["term"])}

    # Anything else -> empty
    return set()

def extract_triples(model_entry):
    """
    Returns a set of normalized (s, p, o) triples.
    Handles:
      - parsed == None
      - parsed as dict with "triples":[...]
      - parsed as a single triple object {"s":...,"p":...,"o":...}
    """
    parsed = _safe_get(model_entry, "triples", "parsed")

    if parsed is None:
        return set()

    def _triple_from_obj(t):
        if not isinstance(t, dict):
            return None
        s, p, o = t.get("s"), t.get("p"), t.get("o")
        if s and p and o:
            return (_norm_text(s), _norm_text(p), _norm_text(o))
        return None

    # Case A: normal form: {"triples":[{s,p,o,...}, ...]}
    if isinstance(parsed, dict) and "triples" in parsed and isinstance(parsed["triples"], list):
        out = set()
        for t in parsed["triples"]:
            tri = _triple_from_obj(t)
            if tri is not None:
                out.add(tri)
        return out

    # Case B: single triple object
    if isinstance(parsed, dict) and parsed.get("s") and parsed.get("p") and parsed.get("o"):
        tri = _triple_from_obj(parsed)
        return {tri} if tri is not None else set()

    # Anything else -> empty
    return set()

def safe_jaccard(a, b):
    """
    Jaccard similarity for sets.
    If both empty, return 1.0 (identical absence).
    """
    if not a and not b:
        return 1.0
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0

# -------------------- MAIN LOOP --------------------
records = []

for lecture_name in sorted(os.listdir(LECTURES_ROOT)):
    lecture_path = os.path.join(LECTURES_ROOT, lecture_name)
    if not os.path.isdir(lecture_path):
        continue

    for fname in sorted(os.listdir(lecture_path)):
        if not fname.lower().endswith(".json"):
            continue

        fpath = os.path.join(lecture_path, fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Skip unreadable JSON
            continue

        slide_id = data.get("slide_id", fname.replace(".json", ""))
        models = data.get("models", {})

        if BASELINE_MODEL not in models:
            # Skip slides without baseline model
            continue

        # ---- Single-model (InternVL3) ----
        C_single = extract_concepts(models.get(BASELINE_MODEL, {}))
        T_single = extract_triples(models.get(BASELINE_MODEL, {}))

        # ---- Multi-model (union) ----
        C_multi = set()
        T_multi = set()
        for model_data in models.values():
            C_multi |= extract_concepts(model_data if isinstance(model_data, dict) else {})
            T_multi |= extract_triples(model_data if isinstance(model_data, dict) else {})

        # ---- Coverage-loss metrics (what multi has that single misses) ----
        concept_coverage_loss = (len(C_multi - C_single) / len(C_multi)) if len(C_multi) > 0 else 0.0
        triple_coverage_loss = (len(T_multi - T_single) / len(T_multi)) if len(T_multi) > 0 else 0.0

        # ---- Optional: similarity metrics (nice for a short paragraph/figure) ----
        concept_jaccard = safe_jaccard(C_single, C_multi)
        triple_jaccard = safe_jaccard(T_single, T_multi)

        records.append({
            "lecture": lecture_name,
            "slide_id": slide_id,
            "concepts_single": len(C_single),
            "concepts_multi": len(C_multi),
            "concept_coverage_loss": concept_coverage_loss,
            "concept_jaccard_single_vs_multi": concept_jaccard,
            "triples_single": len(T_single),
            "triples_multi": len(T_multi),
            "triple_coverage_loss": triple_coverage_loss,
            "triple_jaccard_single_vs_multi": triple_jaccard,
        })

# -------------------- SAVE --------------------
df = pd.DataFrame(records).sort_values(["lecture", "slide_id"])
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows:", len(df))
print(df[[
    "concepts_single","concepts_multi","concept_coverage_loss",
    "triples_single","triples_multi","triple_coverage_loss"
]].describe())

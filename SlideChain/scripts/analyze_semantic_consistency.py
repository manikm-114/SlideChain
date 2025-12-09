import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROVENANCE_ROOT = SCRIPT_DIR.parent / "provenance"
OUT_DIR = SCRIPT_DIR.parent / "analysis_results"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Helper: load parsed provenance JSON
# ---------------------------------------------------------
def load_provenance():
    rows = []
    for lecture_dir in sorted(PROVENANCE_ROOT.iterdir()):
        if not lecture_dir.is_dir(): 
            continue

        lecture = lecture_dir.name     # "Lecture 1"

        for slide_file in sorted(lecture_dir.glob("Slide*.json")):
            slide_id = slide_file.stem.replace("Slide", "")

            try:
                data = json.loads(slide_file.read_text(encoding="utf-8"))
            except:
                continue

            models = data.get("models", {})

            # Count concepts & triples per model
            for model_name, block in models.items():

                # concepts
                c = block.get("concepts", {})
                if isinstance(c, dict) and "concepts" in c:
                    num_c = len(c["concepts"])
                elif isinstance(c, dict) and ("category" in c and "term" in c):
                    num_c = 1
                else:
                    num_c = 0

                # triples
                t = block.get("triples", {})
                if isinstance(t, dict) and "triples" in t and isinstance(t["triples"], list):
                    num_t = len(t["triples"])
                else:
                    num_t = 0

                rows.append([lecture, slide_id, model_name, num_c, num_t])

    df = pd.DataFrame(rows, columns=[
        "lecture", "slide_id", "model_name", "num_concepts", "num_triples"
    ])
    return df


# ---------------------------------------------------------
# Part 1 — Compute per-model concept/triple statistics
# ---------------------------------------------------------
def compute_model_stats(df):
    concept_stats = (
        df.groupby("model_name")["num_concepts"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    triple_stats = (
        df.groupby("model_name")["num_triples"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    concept_stats.to_csv(OUT_DIR / "model_concept_stats.csv", index=False)
    triple_stats.to_csv(OUT_DIR / "model_triple_stats.csv", index=False)

    print("[OK] Wrote per-model stats:")
    print("    concepts ->", OUT_DIR / "model_concept_stats.csv")
    print("    triples  ->", OUT_DIR / "model_triple_stats.csv")


# ---------------------------------------------------------
# Part 2 — Compute Jaccard similarity across models
# ---------------------------------------------------------
def extract_sets(df):
    """
    Returns:
    concept_sets[model_name][slide] -> set of terms
    triple_sets[model_name][slide]  -> set of (s,p,o)
    """
    concept_sets = defaultdict(dict)
    triple_sets = defaultdict(dict)

    # Load provenance JSON again to recover actual sets
    for lecture_dir in sorted(PROVENANCE_ROOT.iterdir()):
        if not lecture_dir.is_dir(): 
            continue

        for slide_file in sorted(lecture_dir.glob("Slide*.json")):
            slide_key = f"{lecture_dir.name}_{slide_file.stem}"
            data = json.loads(slide_file.read_text())

            for model_name, block in data.get("models", {}).items():

                # concepts
                c = block.get("concepts", {})
                if isinstance(c, dict) and "concepts" in c:
                    terms = set([item["term"] for item in c["concepts"]])
                elif isinstance(c, dict) and ("category" in c and "term" in c):
                    terms = {c["term"]}
                else:
                    terms = set()
                concept_sets[model_name][slide_key] = terms

                # triples
                t = block.get("triples", {})
                if isinstance(t, dict) and "triples" in t:
                    triples = set()
                    for tr in t["triples"]:
                        try:
                            triples.add((tr["s"], tr["p"], tr["o"]))
                        except:
                            pass
                else:
                    triples = set()
                triple_sets[model_name][slide_key] = triples

    return concept_sets, triple_sets


def compute_jaccard_matrix(model_list, sets_dict):
    """
    model_list = list of model names
    sets_dict[model][slide_key] = set(...)
    """
    n = len(model_list)
    mat = np.zeros((n, n))

    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i == j:
                mat[i, j] = 1.0
                continue

            overlaps = []
            for slide_key in sets_dict[m1].keys():
                A = sets_dict[m1].get(slide_key, set())
                B = sets_dict[m2].get(slide_key, set())
                if len(A | B) == 0:
                    continue
                overlaps.append(len(A & B) / len(A | B))

            mat[i, j] = np.mean(overlaps) if overlaps else 0.0

    return mat


def save_matrix(mat, model_list, out_path):
    df = pd.DataFrame(mat, index=model_list, columns=model_list)
    df.to_csv(out_path, encoding="utf-8")


# ---------------------------------------------------------
# Part 3 — Lecture-level disagreement & stability
# ---------------------------------------------------------
def compute_lecture_disagreement_and_stability():

    stats_path = OUT_DIR / "slide_level_stats.csv"
    disagree_path = OUT_DIR / "slide_level_disagreement.csv"

    if not stats_path.exists() or not disagree_path.exists():
        raise FileNotFoundError("Required slide-level CSVs are missing.")

    df_stats = pd.read_csv(stats_path)
    df_dis = pd.read_csv(disagree_path)

    # FIX column names
    if "lecture" in df_dis.columns:
        df_dis.rename(columns={"lecture": "lecture_id"}, inplace=True)

    # MERGE HERE
    merged = df_dis.merge(df_stats[["lecture", "slide_id"]].drop_duplicates(),
                          left_on=["lecture_id", "slide_id"],
                          right_on=["lecture", "slide_id"],
                          how="left")

    merged.drop(columns=["lecture"], inplace=True)

    # Now compute lecture-level averages
    lecture_level = (
        merged.groupby("lecture_id")[["concept_disagreement", "triple_disagreement"]]
        .mean()
        .reset_index()
    )

    lecture_level.to_csv(OUT_DIR / "lecture_disagreement.csv", index=False)

    # Stability rule:
    # 0–25% percentile → stable
    # 25–75% → moderate
    # >75% → unstable
    cd = merged["concept_disagreement"].values
    t25, t75 = np.percentile(cd, [25, 75])

    labels = []
    for v in cd:
        if v <= t25:
            labels.append("stable")
        elif v <= t75:
            labels.append("moderate")
            continue
        else:
            labels.append("unstable")

    merged["stability"] = labels
    merged.to_csv(OUT_DIR / "semantic_stability.csv", index=False)

    print("[OK] Wrote lecture-level disagreement ->", OUT_DIR / "lecture_disagreement.csv")
    print("[OK] Wrote stability labels ->", OUT_DIR / "semantic_stability.csv")
    print(f"     (25th percentile={t25:.3f}, 75th percentile={t75:.3f})")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    print("[INFO] Loading provenance records...")
    df = load_provenance()
    print(f"[INFO] Loaded {len(df)} provenance JSON files.")

    # Model statistics
    compute_model_stats(df)

    # Jaccard matrices
    print("[INFO] Computing pairwise Jaccard similarity (concepts & triples)...")
    concept_sets, triple_sets = extract_sets(df)

    model_list = sorted(concept_sets.keys())

    concept_mat = compute_jaccard_matrix(model_list, concept_sets)
    triple_mat = compute_jaccard_matrix(model_list, triple_sets)

    save_matrix(concept_mat, model_list, OUT_DIR / "concept_jaccard_matrix.csv")
    save_matrix(triple_mat, model_list, OUT_DIR / "triple_jaccard_matrix.csv")

    print("[OK] Wrote Jaccard matrices:")
    print("     concepts ->", OUT_DIR / "concept_jaccard_matrix.csv")
    print("     triples  ->", OUT_DIR / "triple_jaccard_matrix.csv")

    # Lecture-level aggregation
    print("[INFO] Loading slide-level disagreement stats...")
    compute_lecture_disagreement_and_stability()

    print("\n=== analyze_semantic_consistency COMPLETE ===")


if __name__ == "__main__":
    main()

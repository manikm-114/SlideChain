import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set(style="whitegrid")

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / ".." / "analysis_results"
FIG_DIR = SCRIPT_DIR / ".." / "figures" / "provenance"

MODEL_CONCEPT_STATS = RESULTS_DIR / "model_concept_stats.csv"
MODEL_TRIPLE_STATS = RESULTS_DIR / "model_triple_stats.csv"
CONCEPT_JACCARD = RESULTS_DIR / "concept_jaccard_matrix.csv"
TRIPLE_JACCARD = RESULTS_DIR / "triple_jaccard_matrix.csv"
LECTURE_DISAGREE = RESULTS_DIR / "lecture_disagreement.csv"
SLIDE_DISAGREE = RESULTS_DIR / "slide_level_disagreement.csv"
SEM_STABILITY = RESULTS_DIR / "semantic_stability.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper: save figure
# -----------------------------
def savefig(name):
    out = FIG_DIR / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure: {name}")

# -----------------------------
# 0. Disagreement Histograms
# -----------------------------
def plot_disagreement_histograms():
    df = pd.read_csv(SLIDE_DISAGREE)

    # Concept disagreement
    plt.figure(figsize=(8, 5))
    sns.histplot(df["concept_disagreement"], bins=20, kde=True, color="steelblue")
    plt.title("Concept Disagreement Distribution")
    plt.xlabel("Concept Disagreement")
    plt.ylabel("Count")
    savefig("concept_disagreement_hist.png")
    savefig("concept_disagreement_hist.pdf")
    savefig("concept_disagreement_hist.svg")

    # Triple disagreement
    plt.figure(figsize=(8, 5))
    sns.histplot(df["triple_disagreement"], bins=20, kde=True, color="darkred")
    plt.title("Triple Disagreement Distribution")
    plt.xlabel("Triple Disagreement")
    plt.ylabel("Count")
    savefig("triple_disagreement_hist.png")
    savefig("triple_disagreement_hist.pdf")
    savefig("triple_disagreement_hist.svg")

# -----------------------------
# 1. Per-model concept statistics
# -----------------------------
def plot_model_concepts():
    df = pd.read_csv(MODEL_CONCEPT_STATS)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model_name", y="mean", hue="model_name", dodge=False, legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Average Concepts Per Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Concepts")
    savefig("model_concepts_mean.png")

# -----------------------------
# 2. Per-model triples statistics
#-----------------------------
def plot_model_triples():
    df = pd.read_csv(MODEL_TRIPLE_STATS)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model_name", y="mean", hue="model_name", dodge=False, legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Average Triples Per Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Triples")
    savefig("model_triples_mean.png")

# -----------------------------
# 3. Heatmaps (Jaccard similarity)
# -----------------------------
def plot_jaccard_heatmaps():
    df_c = pd.read_csv(CONCEPT_JACCARD, index_col=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_c, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Concept Jaccard Similarity Across Models")
    savefig("concept_jaccard_heatmap.png")

    df_t = pd.read_csv(TRIPLE_JACCARD, index_col=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_t, annot=True, cmap="Greens", fmt=".2f")
    plt.title("Triple Jaccard Similarity Across Models")
    savefig("triple_jaccard_heatmap.png")

# -----------------------------
# 4. Lecture-level disagreement
# -----------------------------
def plot_lecture_disagreement():
    df = pd.read_csv(LECTURE_DISAGREE)
    
    df["lecture_num"] = df["lecture_id"].astype(str).str.extract(r"(\d+)").astype(int)
    df = df.sort_values("lecture_num")

    plt.figure(figsize=(14, 5))
    sns.barplot(data=df, x="lecture_num", y="concept_disagreement", color="steelblue")
    plt.xlabel("Lecture")
    plt.ylabel("Avg Concept Disagreement")
    savefig("lecture_concept_disagreement.png")

    plt.figure(figsize=(14, 5))
    sns.barplot(data=df, x="lecture_num", y="triple_disagreement", color="darkorange")
    plt.xlabel("Lecture")
    plt.ylabel("Avg Triple Disagreement")
    savefig("lecture_triple_disagreement.png")

    # Combined plot
    plt.figure(figsize=(14, 5))
    sns.lineplot(data=df, x="lecture_num", y="concept_disagreement", label="Concept", marker="o")
    sns.lineplot(data=df, x="lecture_num", y="triple_disagreement", label="Triple", marker="s")
    plt.title("Lecture-Level Disagreement")
    plt.xlabel("Lecture")
    plt.ylabel("Disagreement")
    savefig("lecture_disagreement.png")

# -----------------------------
# 5. Semantic Stability
# -----------------------------
def plot_semantic_stability():
    df = pd.read_csv(SEM_STABILITY)
    counts = df["stability"].value_counts()

    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, dodge=False, legend=False)
    plt.xlabel("Stability")
    plt.ylabel("Slides")
    plt.title("Semantic Stability Distribution")
    savefig("semantic_stability_distribution.png")

# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"[INFO] Saving provenance figures to: {FIG_DIR}")

    plot_disagreement_histograms()
    plot_model_concepts()
    plot_model_triples()
    plot_jaccard_heatmaps()
    plot_lecture_disagreement()
    plot_semantic_stability()

if __name__ == "__main__":
    main()

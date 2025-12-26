import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
INPUT_CSV = Path(r"F:\Research Files\SlideChain\SlideChain\analysis_results\single_vs_multi_provenance.csv")
OUTPUT_DIR = Path(r"F:\Research Files\SlideChain\SlideChain\figures\provenance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

# ======================================================
# Figure 1: Concept Coverage Loss (with mean & median)
# ======================================================
concept_mean = df["concept_coverage_loss"].mean()
concept_median = df["concept_coverage_loss"].median()

plt.figure(figsize=(5, 3.8))
plt.hist(df["concept_coverage_loss"], bins=20)
plt.axvline(concept_mean, linestyle="--", linewidth=1, label=f"Mean = {concept_mean:.2f}")
plt.axvline(concept_median, linestyle=":", linewidth=1, label=f"Median = {concept_median:.2f}")

plt.xlabel("Concept coverage loss")
plt.ylabel("Number of slides")
plt.title("Single-model vs. multi-model concept coverage loss")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "concept_coverage_loss_hist.png", dpi=300)
plt.close()

# ======================================================
# Figure 2: Triple Coverage Loss (with mean & median)
# ======================================================
triple_mean = df["triple_coverage_loss"].mean()
triple_median = df["triple_coverage_loss"].median()

plt.figure(figsize=(5, 3.8))
plt.hist(df["triple_coverage_loss"], bins=20)
plt.axvline(triple_mean, linestyle="--", linewidth=1, label=f"Mean = {triple_mean:.2f}")
plt.axvline(triple_median, linestyle=":", linewidth=1, label=f"Median = {triple_median:.2f}")

plt.xlabel("Triple coverage loss")
plt.ylabel("Number of slides")
plt.title("Single-model vs. multi-model triple coverage loss")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "triple_coverage_loss_hist.png", dpi=300)
plt.close()

print("Figures saved to:", OUTPUT_DIR)
print(f"Concept coverage loss: mean={concept_mean:.3f}, median={concept_median:.3f}")
print(f"Triple coverage loss:  mean={triple_mean:.3f}, median={triple_median:.3f}")

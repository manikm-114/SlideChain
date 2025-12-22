import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
INPUT_CSV = Path(r"F:\Research Files\SlideChain\SlideChain\analysis_results\single_vs_multi_provenance.csv")
OUTPUT_DIR = Path(r"F:\Research Files\SlideChain\SlideChain\figures\provenance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

# -------------------------------
# Figure 1: Concept Coverage Loss
# -------------------------------
plt.figure(figsize=(5, 3.8))
plt.hist(df["concept_coverage_loss"], bins=20)
plt.xlabel("Concept coverage loss")
plt.ylabel("Number of slides")
plt.title("Single-model vs. multi-model concept coverage loss")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "concept_coverage_loss_hist.png", dpi=300)
plt.close()

# ------------------------------
# Figure 2: Triple Coverage Loss
# ------------------------------
plt.figure(figsize=(5, 3.8))
plt.hist(df["triple_coverage_loss"], bins=20)
plt.xlabel("Triple coverage loss")
plt.ylabel("Number of slides")
plt.title("Single-model vs. multi-model triple coverage loss")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "triple_coverage_loss_hist.png", dpi=300)
plt.close()

print("Figures saved to:", OUTPUT_DIR)

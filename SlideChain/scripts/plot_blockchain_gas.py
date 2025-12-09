import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV = os.path.join(ROOT, "chain_results", "gas_detailed.csv")

OUT_DIR = os.path.join(ROOT, "figures", "blockchain")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("[INFO] Loading:", CSV)
    df = pd.read_csv(CSV)

    required_cols = {"tx_hash", "block", "gas_used", "effective_gas_price"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Create slide index (monotonic ordering)
    df = df.reset_index().rename(columns={"index": "slide_index"})

    # -----------------------------------------------------------
    # Plot 1 — Gas Usage Histogram
    # -----------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(df["gas_used"], bins=20, edgecolor="black")
    plt.xlabel("Gas Used")
    plt.ylabel("Count")
    plt.title("Gas Usage Distribution Across Slides")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gas_usage_histogram.png")
    plt.savefig(path)
    plt.close()
    print("[OK] Saved:", path)

    # -----------------------------------------------------------
    # Plot 2 — Gas Used Per Slide
    # -----------------------------------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(df["slide_index"], df["gas_used"], marker="o", markersize=2)
    plt.xlabel("Slide Index")
    plt.ylabel("Gas Used")
    plt.title("Gas Used Per Slide")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gas_per_slide.png")
    plt.savefig(path)
    plt.close()
    print("[OK] Saved:", path)

    # -----------------------------------------------------------
    # Plot 3 — Cumulative Gas Consumption
    # -----------------------------------------------------------
    df["cum_gas"] = df["gas_used"].cumsum()
    plt.figure(figsize=(12,5))
    plt.plot(df["slide_index"], df["cum_gas"], linewidth=2)
    plt.xlabel("Slide Index")
    plt.ylabel("Cumulative Gas")
    plt.title("Cumulative Gas Consumption Over Dataset")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cumulative_gas.png")
    plt.savefig(path)
    plt.close()
    print("[OK] Saved:", path)

    # -----------------------------------------------------------
    # Plot 4 — Gas vs Block Number (scatter)
    # -----------------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.scatter(df["block"], df["gas_used"], s=10)
    plt.xlabel("Block Number")
    plt.ylabel("Gas Used")
    plt.title("Gas Used per Block")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gas_vs_block.png")
    plt.savefig(path)
    plt.close()
    print("[OK] Saved:", path)

    # -----------------------------------------------------------
    # Plot 5 — Effective Gas Price Distribution
    # -----------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(df["effective_gas_price"], bins=20, edgecolor="black")
    plt.xlabel("Effective Gas Price (wei)")
    plt.ylabel("Count")
    plt.title("Gas Price Distribution")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gas_price_distribution.png")
    plt.savefig(path)
    plt.close()
    print("[OK] Saved:", path)

    print("\n[COMPLETE] All blockchain gas figures generated.")

if __name__ == "__main__":
    main()

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

CHAIN_RESULTS_DIR = ROOT / "chain_results"
FIG_ROOT = ROOT / "figures" / "blockchain"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

BLOCK_DIST_CSV = CHAIN_RESULTS_DIR / "block_distribution.csv"
COST_PER_SLIDE_CSV = CHAIN_RESULTS_DIR / "cost_per_slide.csv"
SCALABILITY_CSV = CHAIN_RESULTS_DIR / "scalability_summary.csv"
TIMEGAP_CSV = CHAIN_RESULTS_DIR / "integrity_timegap.csv"
TAMPER_CSV = CHAIN_RESULTS_DIR / "tamper_detection.csv"


# ---------------------------------------------------------------------------
# 1. Block distribution plots
# ---------------------------------------------------------------------------

def plot_block_distribution():
    print("=== BLOCK DISTRIBUTION PLOTS ===")

    if not BLOCK_DIST_CSV.exists():
        print(f"[WARN] {BLOCK_DIST_CSV} not found, skipping block plots.")
        return

    df = pd.read_csv(BLOCK_DIST_CSV)
    print(f"[INFO] Loaded: {BLOCK_DIST_CSV}")

    # Expect columns: lecture_id, slide_id, block_number, timestamp, tx_hash, log_index
    if "block_number" not in df.columns:
        raise ValueError("block_distribution.csv must contain 'block_number' column")

    # 1) Histogram of block numbers
    plt.figure(figsize=(8, 4))
    plt.hist(df["block_number"], bins=30, edgecolor="black")
    plt.xlabel("Block number")
    plt.ylabel("Number of slide registrations")
    plt.title("Distribution of Slide Registrations Across Blocks")
    out_path = FIG_ROOT / "block_histogram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 2) Transactions per block (bar plot)
    counts = df["block_number"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Block number")
    plt.ylabel("Registrations in block")
    plt.title("Transactions per Block")
    out_path = FIG_ROOT / "transactions_per_block.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 3) Block vs slide index
    df_sorted = df.sort_values(["block_number", "log_index"]).reset_index(drop=True)
    df_sorted["slide_index"] = df_sorted.index + 1
    plt.figure(figsize=(8, 4))
    plt.scatter(df_sorted["slide_index"], df_sorted["block_number"], s=4)
    plt.xlabel("Slide index (sorted by block)")
    plt.ylabel("Block number")
    plt.title("Block Assignment Over Slide Index")
    out_path = FIG_ROOT / "block_vs_slide_index.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 4) Registration timeline (time vs slide index)
    if "timestamp" in df_sorted.columns:
        df_sorted["time"] = pd.to_datetime(df_sorted["timestamp"], unit="s")
        plt.figure(figsize=(8, 4))
        plt.plot(df_sorted["time"], df_sorted["slide_index"])
        plt.xlabel("Time")
        plt.ylabel("Cumulative registered slides")
        plt.title("Registration Timeline")
        plt.tight_layout()
        out_path = FIG_ROOT / "registration_timeline.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    print("[OK] Block distribution figures generated.")


# ---------------------------------------------------------------------------
# 2. Cost analysis plots
# ---------------------------------------------------------------------------

def plot_costs():
    print("=== COST ANALYSIS PLOTS ===")

    if not COST_PER_SLIDE_CSV.exists():
        print(f"[WARN] {COST_PER_SLIDE_CSV} not found, skipping cost plots.")
        return

    df = pd.read_csv(COST_PER_SLIDE_CSV)

    # Expect columns like: lecture, slide, gas_used, gas_price_gwei, cost_eth, cost_usd
    cost_col = None
    for cand in ["cost_usd", "cost_per_slide_usd", "cost_eth"]:
        if cand in df.columns:
            cost_col = cand
            break

    if cost_col is None:
        raise ValueError("cost_per_slide.csv must contain a per-slide cost column (e.g., 'cost_usd').")

    # 1) Cost-per-slide histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df[cost_col], bins=30, edgecolor="black")
    plt.xlabel(f"Cost per slide ({cost_col})")
    plt.ylabel("Number of slides")
    plt.title("Distribution of Cost per Slide")
    plt.tight_layout()
    out_path = FIG_ROOT / "cost_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 2) Cumulative cost curve
    df_sorted = df.sort_values(cost_col).reset_index(drop=True)
    df_sorted["slide_index"] = df_sorted.index + 1
    df_sorted["cumulative_cost"] = df_sorted[cost_col].cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(df_sorted["slide_index"], df_sorted["cumulative_cost"])
    plt.xlabel("Slide index (sorted by cost)")
    plt.ylabel(f"Cumulative cost ({cost_col})")
    plt.title("Cumulative Registration Cost Across Dataset")
    plt.tight_layout()
    out_path = FIG_ROOT / "cumulative_cost_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[OK] Cost figures generated.")


# ---------------------------------------------------------------------------
# 3. Scalability plots
# ---------------------------------------------------------------------------

def plot_scalability():
    print("=== SCALABILITY PLOTS ===")

    if not SCALABILITY_CSV.exists():
        print(f"[WARN] {SCALABILITY_CSV} not found, skipping scalability plots.")
        return

    df = pd.read_csv(SCALABILITY_CSV)

    # Expect columns: scenario, network, dataset_size, gas_per_slide, total_gas, total_cost_usd
    required_cols = {"scenario", "network", "dataset_size", "gas_per_slide", "total_gas", "total_cost_usd"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"scalability_summary.csv missing required columns: {missing}")

    # 1) Gas scaling across dataset sizes, per network
    plt.figure(figsize=(8, 4))
    for network, sub in df.groupby("network"):
        plt.plot(sub["dataset_size"], sub["total_gas"], marker="o", label=network)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (slides, log scale)")
    plt.ylabel("Total gas (log scale)")
    plt.title("Gas Scaling Across Dataset Sizes and Networks")
    plt.legend()
    plt.tight_layout()
    out_path = FIG_ROOT / "scaling_gas.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 2) Cost scaling across dataset sizes, per network
    plt.figure(figsize=(8, 4))
    for network, sub in df.groupby("network"):
        plt.plot(sub["dataset_size"], sub["total_cost_usd"], marker="o", label=network)
    plt.xscale("log")
    plt.xlabel("Dataset size (slides, log scale)")
    plt.ylabel("Total cost (USD)")
    plt.title("Total Cost Scaling Across Dataset Sizes and Networks")
    plt.legend()
    plt.tight_layout()
    out_path = FIG_ROOT / "scaling_cost.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[OK] Scalability figures generated.")


# ---------------------------------------------------------------------------
# 4. Integrity plots (tamper + time-gap)
# ---------------------------------------------------------------------------

def plot_integrity():
    print("=== INTEGRITY PLOTS ===")

    # --- Time-gap violin plot ---
    if TIMEGAP_CSV.exists():
        df_tgap = pd.read_csv(TIMEGAP_CSV)
        # Expect column: time_gap_sec
        time_col = "time_gap_sec" if "time_gap_sec" in df_tgap.columns else None
        if time_col is None:
            raise ValueError("integrity_timegap.csv must contain 'time_gap_sec' column.")

        plt.figure(figsize=(5, 4))
        sns.violinplot(y=df_tgap[time_col])
        plt.ylabel("Time gap (seconds)")
        plt.title("Distribution of Local→Chain Registration Time Gaps")
        plt.tight_layout()
        out_path = FIG_ROOT / "time_gap_violin.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        print(f"[WARN] {TIMEGAP_CSV} not found, skipping time-gap plot.")

    # --- Tamper detection bar chart ---
    if TAMPER_CSV.exists():
        df_t = pd.read_csv(TAMPER_CSV)
        # Expected columns from your file:
        # lecture, slide, good_hash, chain_hash, match_good, corrupted_hash, match_corrupted

        # Normalize boolean-like columns
        for col in ["match_good", "match_corrupted"]:
            if col in df_t.columns:
                df_t[col] = (
                    df_t[col]
                    .astype(str)
                    .str.upper()
                    .map({"TRUE": True, "FALSE": False})
                )

        if "match_corrupted" not in df_t.columns:
            raise ValueError("tamper_detection.csv must contain 'match_corrupted' column.")

        # Define "detection success" as: corrupted hash does NOT match on-chain hash
        detected_mask = ~df_t["match_corrupted"]
        detected = int(detected_mask.sum())
        total = len(df_t)
        missed = total - detected
        detection_rate = detected / total if total > 0 else 0.0

        # Bar chart: detected vs missed
        plt.figure(figsize=(5, 4))
        plt.bar(["Detected", "Missed"], [detected, missed], color=["green", "red"])
        plt.ylabel("Number of tampered slides")
        plt.title(f"Tamper Detection Success (rate={detection_rate:.2%})")
        plt.tight_layout()
        out_path = FIG_ROOT / "tamper_detection_bar.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        print(f"[WARN] {TAMPER_CSV} not found, skipping tamper detection plot.")

    print("[OK] Integrity figures generated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plot_block_distribution()
    plot_costs()
    plot_scalability()
    plot_integrity()


if __name__ == "__main__":
    main()

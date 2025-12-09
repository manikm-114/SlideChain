import csv
import math
from pathlib import Path
from statistics import mean

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CHAIN_RESULTS_DIR = ROOT_DIR / "chain_results"
CHAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAS_DETAILED_PATH = CHAIN_RESULTS_DIR / "gas_detailed.csv"

# -------------------------------------------------------------------
# Hyperparameters / Assumptions
# -------------------------------------------------------------------
# You can adjust these if you want to explore different economic regimes.
ETH_PRICE_USD = 3000.0  # same as analyze_costs.py, for consistency

# These are *simulation configs*, not live on-chain prices.
# You will describe them explicitly in the paper as assumed values.
NETWORKS = {
    "ethereum_l1": {
        "block_gas_limit": 30_000_000,  # typical order of magnitude
        "block_time_sec": 12.0,
        "gas_price_gwei": 30.0,
    },
    "polygon_pos": {
        "block_gas_limit": 30_000_000,
        "block_time_sec": 2.0,
        "gas_price_gwei": 5.0,
    },
    "optimism_like": {
        "block_gas_limit": 30_000_000,
        "block_time_sec": 2.0,
        # effective gas including DA (simplified)
        "gas_price_gwei": 1.0,
    },
    "arbitrum_like": {
        "block_gas_limit": 30_000_000,
        "block_time_sec": 0.25,
        "gas_price_gwei": 0.5,
    },
}

# Dataset sizes to simulate
DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000]

# Scenarios: how much more gas a richer on-chain representation would cost
SCENARIOS = {
    "hash_only": 1.0,               # your current SlideChain (store only keccak + URI)
    "hash_plus_metadata_x2": 2.0,   # double gas per slide
    "hash_plus_rich_metadata_x4": 4.0,  # 4x gas per slide
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_mean_gas_per_slide(gas_csv_path: Path) -> float:
    """
    Load gas_detailed.csv and compute mean gasUsed across all SlideRegistered events.
    Assumes a column named 'gas_used'.
    """
    if not gas_csv_path.exists():
        raise FileNotFoundError(f"gas_detailed.csv not found at {gas_csv_path}")

    gas_values = []
    with gas_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Try a few possible column names for robustness
        cols = reader.fieldnames or []
        # Normalize possible names
        gas_col = None
        for cand in ["gas_used", "gasUsed", "gas"]:
            if cand in cols:
                gas_col = cand
                break

        if gas_col is None:
            raise ValueError(
                f"Could not find a gas column in {gas_csv_path}. "
                f"Available columns: {cols}"
            )

        for row in reader:
            try:
                g = int(row[gas_col])
                gas_values.append(g)
            except Exception:
                continue

    if not gas_values:
        raise ValueError("No valid gas values found in gas_detailed.csv")

    return mean(gas_values)


def gas_to_eth(gas_units: float, gas_price_gwei: float) -> float:
    """
    Convert gas * gas_price (in gwei) to ETH.
    1 gwei = 1e-9 ETH.
    """
    return gas_units * gas_price_gwei * 1e-9


def eth_to_usd(eth_amount: float, eth_price_usd: float) -> float:
    return eth_amount * eth_price_usd


def format_float(x: float, ndigits: int = 6) -> float:
    """
    For pretty printing / CSV, round but keep as float.
    """
    return round(float(x), ndigits)


# -------------------------------------------------------------------
# Core simulation
# -------------------------------------------------------------------
def simulate_scaling(
    mean_gas_per_slide: float,
    out_path: Path,
    eth_price_usd: float = ETH_PRICE_USD,
):
    """
    For each SCENARIO × NETWORK × DATASET_SIZE, compute:
      - total gas
      - blocks needed
      - time span (sec, hours, days)
      - cost in ETH and USD

    Write results to CSV.
    """
    headers = [
        "scenario",
        "network",
        "dataset_size",
        "gas_per_slide",
        "total_gas",
        "blocks_needed",
        "time_seconds",
        "time_hours",
        "time_days",
        "gas_price_gwei",
        "total_cost_eth",
        "total_cost_usd",
    ]

    rows = []

    for scenario_name, multiplier in SCENARIOS.items():
        scenario_gas_per_slide = mean_gas_per_slide * multiplier

        for net_name, cfg in NETWORKS.items():
            block_gas_limit = cfg["block_gas_limit"]
            block_time_sec = cfg["block_time_sec"]
            gas_price_gwei = cfg["gas_price_gwei"]

            for n_slides in DATASET_SIZES:
                total_gas = scenario_gas_per_slide * n_slides

                # Blocks needed (ceil)
                blocks_needed = math.ceil(total_gas / block_gas_limit)

                # Time = blocks * block_time
                time_seconds = blocks_needed * block_time_sec
                time_hours = time_seconds / 3600.0
                time_days = time_hours / 24.0

                # Cost
                total_cost_eth = gas_to_eth(total_gas, gas_price_gwei)
                total_cost_usd = eth_to_usd(total_cost_eth, eth_price_usd)

                rows.append(
                    {
                        "scenario": scenario_name,
                        "network": net_name,
                        "dataset_size": n_slides,
                        "gas_per_slide": format_float(scenario_gas_per_slide, 3),
                        "total_gas": format_float(total_gas, 3),
                        "blocks_needed": int(blocks_needed),
                        "time_seconds": format_float(time_seconds, 3),
                        "time_hours": format_float(time_hours, 3),
                        "time_days": format_float(time_days, 3),
                        "gas_price_gwei": format_float(gas_price_gwei, 3),
                        "total_cost_eth": format_float(total_cost_eth, 8),
                        "total_cost_usd": format_float(total_cost_usd, 4),
                    }
                )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote scalability summary -> {out_path}")


def compute_throughput_limits(
    mean_gas_per_slide: float,
    out_path: Path,
):
    """
    For each NETWORK, compute:
      - max slides per block
      - max slides per second
      - max slides per hour / day
    Based on current mean gas and block gas limits.
    """
    headers = [
        "network",
        "block_gas_limit",
        "block_time_sec",
        "gas_price_gwei",
        "gas_per_slide",
        "max_slides_per_block",
        "max_slides_per_second",
        "max_slides_per_hour",
        "max_slides_per_day",
    ]

    rows = []

    for net_name, cfg in NETWORKS.items():
        block_gas_limit = cfg["block_gas_limit"]
        block_time_sec = cfg["block_time_sec"]
        gas_price_gwei = cfg["gas_price_gwei"]

        max_slides_block = block_gas_limit // math.ceil(mean_gas_per_slide)
        if max_slides_block <= 0:
            max_slides_block = 0

        if block_time_sec > 0 and max_slides_block > 0:
            max_slides_sec = max_slides_block / block_time_sec
        else:
            max_slides_sec = 0.0

        max_slides_hour = max_slides_sec * 3600.0
        max_slides_day = max_slides_hour * 24.0

        rows.append(
            {
                "network": net_name,
                "block_gas_limit": int(block_gas_limit),
                "block_time_sec": format_float(block_time_sec, 3),
                "gas_price_gwei": format_float(gas_price_gwei, 3),
                "gas_per_slide": format_float(mean_gas_per_slide, 3),
                "max_slides_per_block": int(max_slides_block),
                "max_slides_per_second": format_float(max_slides_sec, 6),
                "max_slides_per_hour": format_float(max_slides_hour, 3),
                "max_slides_per_day": format_float(max_slides_day, 3),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote throughput limits -> {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("[INFO] Loading gas_detailed.csv from:", GAS_DETAILED_PATH)
    mean_gas = load_mean_gas_per_slide(GAS_DETAILED_PATH)
    print(f"[INFO] Mean gas per slide (from chain): {mean_gas:.3f}")

    scalability_out = CHAIN_RESULTS_DIR / "scalability_summary.csv"
    throughput_out = CHAIN_RESULTS_DIR / "throughput_limits.csv"

    print("[INFO] Simulating scaling across scenarios and networks...")
    simulate_scaling(mean_gas, scalability_out, eth_price_usd=ETH_PRICE_USD)

    print("[INFO] Computing theoretical throughput limits...")
    compute_throughput_limits(mean_gas, throughput_out)

    print("\n=== analyze_scalability COMPLETE ===")


if __name__ == "__main__":
    main()

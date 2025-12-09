import json
import csv
from pathlib import Path
from web3 import Web3
import statistics

# -----------------------
# Config
# -----------------------
RPC_URL = "http://127.0.0.1:8545"
ETH_PRICE_USD = 3000   # <-- change for paper if needed

SCRIPT_DIR = Path(__file__).parent
ABI_PATH = SCRIPT_DIR / "SlideChain_abi.json"
ADDR_PATH = SCRIPT_DIR / "SlideChain_address.txt"
OUT_DIR = SCRIPT_DIR / ".." / "chain_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------
# Load ABI + Contract
# -----------------------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
assert w3.is_connected(), "Web3 not connected"

abi = json.loads(ABI_PATH.read_text())
address = ADDR_PATH.read_text().strip()
contract = w3.eth.contract(address=address, abi=abi)

# -----------------------
# Load gas_detailed.csv
# -----------------------
gas_csv = OUT_DIR / "gas_detailed.csv"

if not gas_csv.exists():
    raise FileNotFoundError("gas_detailed.csv not found — run analyze_gas_detailed.py first.")

records = []
with open(gas_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gas_used = int(row["gas_used"])
        eff_price = int(row["effective_gas_price"])
        records.append((gas_used, eff_price))

# -----------------------
# Compute costs
# -----------------------
costs_eth = []
costs_usd = []

for gas, price in records:
    cost_eth = gas * price / 1e18
    cost_usd = cost_eth * ETH_PRICE_USD

    costs_eth.append(cost_eth)
    costs_usd.append(cost_usd)

# -----------------------
# Write CSV
# -----------------------
out_csv = OUT_DIR / "cost_per_slide.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["slide_index", "gas_used", "gas_price", "cost_eth", "cost_usd"])

    for i, ((gas, price), eth, usd) in enumerate(zip(records, costs_eth, costs_usd), start=1):
        writer.writerow([i, gas, price, eth, usd])

print("[OK] Wrote:", out_csv)

# -----------------------
# Summary stats
# -----------------------
print("\n=== Cost Summary ===")
print(f"Slides: {len(records)}")
print(f"ETH price used: ${ETH_PRICE_USD} per ETH\n")

print("Min cost per slide (USD):", min(costs_usd))
print("Max cost per slide (USD):", max(costs_usd))
print("Mean cost per slide (USD):", statistics.mean(costs_usd))

print("\nTotal dataset cost (ETH):", sum(costs_eth))
print("Total dataset cost (USD):", sum(costs_usd))
print("=======================")

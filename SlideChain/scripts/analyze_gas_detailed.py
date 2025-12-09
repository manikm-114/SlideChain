import json
from pathlib import Path
from web3 import Web3
import csv
import statistics

# -----------------------
# Config
# -----------------------
RPC_URL = "http://127.0.0.1:8545"

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
# Fetch SlideRegistered logs
# -----------------------
event_sig = contract.events.SlideRegistered._get_event_abi()
topic = w3.keccak(text="SlideRegistered(uint256,uint256,string,string,address,uint256)").hex()

logs = w3.eth.get_logs({"fromBlock": 0, "toBlock": "latest", "topics": [topic]})
print(f"[INFO] Found {len(logs)} SlideRegistered events")

# -----------------------
# Process logs → get receipts
# -----------------------
records = []

for log in logs:
    tx_hash = log["transactionHash"].hex()
    receipt = w3.eth.get_transaction_receipt(tx_hash)

    gas_used = receipt["gasUsed"]
    eff_price = receipt.get("effectiveGasPrice", None)
    block = receipt["blockNumber"]

    records.append({
        "tx_hash": tx_hash,
        "block": block,
        "gas_used": gas_used,
        "effective_gas_price": eff_price
    })

# -----------------------
# Write out CSV
# -----------------------
csv_path = OUT_DIR / "gas_detailed.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tx_hash", "block", "gas_used", "effective_gas_price"])
    
    for r in records:
        writer.writerow([r["tx_hash"], r["block"], r["gas_used"], r["effective_gas_price"]])

print("[OK] Wrote:", csv_path)

# -----------------------
# Summary Statistics
# -----------------------
gas_list = [r["gas_used"] for r in records]

print("\n=== Detailed Gas Summary ===")
print("Count:", len(gas_list))
print("Min:", min(gas_list))
print("Max:", max(gas_list))
print("Mean:", statistics.mean(gas_list))
print("Median:", statistics.median(gas_list))
print("============================")

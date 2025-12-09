import json
import csv
from pathlib import Path
from web3 import Web3
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

event_signature_hash = w3.keccak(text="SlideRegistered(uint256,uint256,string,string,address,uint256)").hex()

# -----------------------
# Fetch all logs
# -----------------------
latest = w3.eth.block_number
logs = w3.eth.get_logs({"fromBlock": 0, "toBlock": latest, "address": address})

print(f"[INFO] Found {len(logs)} SlideRegistered logs")

# -----------------------
# Decode log entries
# -----------------------
records = []

for lg in logs:
    block_num = lg["blockNumber"]
    tx_hash = lg["transactionHash"].hex()
    log_index = lg["logIndex"]

    # Decode topics:
    lecture_id = int(lg["topics"][1].hex(), 16)
    slide_id = int(lg["topics"][2].hex(), 16)

    # Fetch timestamp
    ts = w3.eth.get_block(block_num).timestamp

    records.append((lecture_id, slide_id, block_num, ts, tx_hash, log_index))

# -----------------------
# Sort by block then log index
# -----------------------
records.sort(key=lambda x: (x[2], x[5]))

# -----------------------
# Write block mapping CSV
# -----------------------
out_csv = OUT_DIR / "block_distribution.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["lecture_id", "slide_id", "block_number", "timestamp", "tx_hash", "log_index"])
    writer.writerows(records)

print("[OK] Wrote:", out_csv)

# -----------------------
# Analyze throughput
# -----------------------
timestamps = [r[3] for r in records]
first_ts, last_ts = min(timestamps), max(timestamps)
time_span = last_ts - first_ts

slides = len(records)
slides_per_second = slides / time_span if time_span > 0 else None

# -----------------------
# Blocks-per-slide statistics
# -----------------------
block_numbers = [r[2] for r in records]
unique_blocks = set(block_numbers)

# Count transactions per block
tx_per_block = {}
for b in block_numbers:
    tx_per_block[b] = tx_per_block.get(b, 0) + 1

tx_counts = list(tx_per_block.values())

print("\n=== Block Distribution Summary ===")
print("Total slides:", slides)
print("Total blocks touched:", len(unique_blocks))
print("Min tx/block:", min(tx_counts))
print("Max tx/block:", max(tx_counts))
print("Mean tx/block:", statistics.mean(tx_counts))
print("Median tx/block:", statistics.median(tx_counts))

print("\n=== Throughput Summary ===")
print(f"First ts: {first_ts}")
print(f"Last ts:  {last_ts}")
print(f"Time span: {time_span} seconds")
print(f"Approx throughput: {slides_per_second:.4f} slides/sec")
print("=================================")

# scripts/analyze_chain_metrics.py

import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

from web3 import Web3

from config import RPC_URL, ABI_PATH, ADDRESS_PATH


def load_contract():
    """Connect to RPC, load ABI + contract instance."""
    print(f"[DEBUG] Loading ABI from: {ABI_PATH}")
    if not ABI_PATH.exists():
        raise FileNotFoundError(f"ABI file not found: {ABI_PATH}")
    if not ADDRESS_PATH.exists():
        raise FileNotFoundError(f"Address file not found: {ADDRESS_PATH}")

    abi = json.loads(ABI_PATH.read_text(encoding="utf-8"))
    address = ADDRESS_PATH.read_text(encoding="utf-8").strip()

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise RuntimeError(f"Could not connect to RPC: {RPC_URL}")
    print(f"[OK] Connected to RPC: {RPC_URL}")

    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
    print(f"[OK] Contract loaded at: {address}")
    return w3, contract


def fetch_slide_registered_events(w3: Web3, contract) -> List[Any]:
    """Fetch all SlideRegistered events using web3.py v6-compatible get_logs."""
    latest_block = w3.eth.block_number
    print(f"[INFO] Latest block: {latest_block}")

    event_abi = None
    # Find ABI for SlideRegistered event
    for item in contract.abi:
        if item.get("type") == "event" and item.get("name") == "SlideRegistered":
            event_abi = item
            break
    if event_abi is None:
        raise RuntimeError("SlideRegistered ABI not found in contract ABI.")

    # Build filter for get_logs
    event_signature_hash = w3.keccak(
        text="SlideRegistered(uint256,uint256,string,string,address,uint256)"
    ).hex()

    filter_params = {
        "fromBlock": 0,
        "toBlock": latest_block,
        "address": contract.address,
        "topics": [event_signature_hash],  # topic0 = event signature
    }

    print("[INFO] Fetching logs via w3.eth.get_logs()…")
    raw_logs = w3.eth.get_logs(filter_params)

    # Decode logs
    events = []
    for log in raw_logs:
        decoded = contract.events.SlideRegistered().process_log(log)
        events.append(decoded)

    print(f"[INFO] Found {len(events)} SlideRegistered events.")
    return events



def build_metrics(w3: Web3, events: List[Any]) -> List[Dict[str, Any]]:
    """Build per-slide metrics from event logs + receipts + blocks."""
    metrics = []

    for ev in events:
        args = ev["args"]
        lecture_id = int(args["lectureId"])
        slide_id = int(args["slideId"])
        slide_hash = str(args["slideHash"])
        uri = str(args["uri"])
        registrant = str(args["registrant"])

        block_number = ev["blockNumber"]
        tx_hash = ev["transactionHash"]

        # Block for timestamp
        block = w3.eth.get_block(block_number)
        timestamp = int(block["timestamp"])

        # Tx receipt for gas usage
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        gas_used = int(receipt["gasUsed"])

        # effectiveGasPrice is available on EIP-1559 chains (Hardhat)
        # Fallback to tx.gasPrice if not present
        try:
            gas_price = int(receipt.get("effectiveGasPrice"))  # type: ignore
        except Exception:
            tx = w3.eth.get_transaction(tx_hash)
            gas_price = int(tx.get("gasPrice", 0))

        metrics.append(
            {
                "lecture": lecture_id,
                "slide": slide_id,
                "slide_hash": slide_hash,
                "uri": uri,
                "registrant": registrant,
                "block_number": block_number,
                "timestamp": timestamp,
                "tx_hash": tx_hash.hex(),
                "gas_used": gas_used,
                "gas_price": gas_price,
                "gas_cost_wei": gas_used * gas_price,
            }
        )

    # Sort by (lecture, slide) just for deterministic CSV ordering
    metrics.sort(key=lambda m: (m["lecture"], m["slide"]))
    return metrics


def write_metrics_csv(metrics: List[Dict[str, Any]], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lecture",
                "slide",
                "slide_hash",
                "uri",
                "registrant",
                "block_number",
                "timestamp",
                "tx_hash",
                "gas_used",
                "gas_price",
                "gas_cost_wei",
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m["lecture"],
                    m["slide"],
                    m["slide_hash"],
                    m["uri"],
                    m["registrant"],
                    m["block_number"],
                    m["timestamp"],
                    m["tx_hash"],
                    m["gas_used"],
                    m["gas_price"],
                    m["gas_cost_wei"],
                ]
            )
    print(f"[OK] Wrote metrics CSV -> {out_path}")


def summarize_metrics(metrics: List[Dict[str, Any]]) -> None:
    if not metrics:
        print("[WARN] No metrics to summarize.")
        return

    gas_list = [m["gas_used"] for m in metrics]
    cost_list = [m["gas_cost_wei"] for m in metrics]
    ts_list = [m["timestamp"] for m in metrics]

    total_slides = len(metrics)
    total_gas = sum(gas_list)
    total_cost_wei = sum(cost_list)

    gas_min = min(gas_list)
    gas_max = max(gas_list)
    gas_mean = statistics.mean(gas_list)
    gas_median = statistics.median(gas_list)

    print("\n=== On-Chain Gas Usage Summary ===")
    print(f"Total slides registered:  {total_slides}")
    print(f"Total gas used:          {total_gas}")
    print(f"Min gas per slide:       {gas_min}")
    print(f"Max gas per slide:       {gas_max}")
    print(f"Mean gas per slide:      {gas_mean:.2f}")
    print(f"Median gas per slide:    {gas_median:.2f}")
    print(f"Total gas cost (wei):    {total_cost_wei}")
    print("==================================")

    # Throughput estimate (slides per second) based on timestamps
    t_min = min(ts_list)
    t_max = max(ts_list)
    span = max(1, t_max - t_min)  # avoid divide-by-zero for fast runs

    slides_per_sec = total_slides / span
    print("\n=== Registration Throughput (approx) ===")
    print(f"First registration ts:   {t_min}")
    print(f"Last registration ts:    {t_max}")
    print(f"Time span (seconds):     {span}")
    print(f"Approx slides / second:  {slides_per_sec:.4f}")
    print("========================================")


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "chain_results"
    out_csv = out_dir / "chain_metrics.csv"

    w3, contract = load_contract()
    events = fetch_slide_registered_events(w3, contract)

    if not events:
        print("[WARN] No SlideRegistered events found. Did you run register_all_slides.py?")
        return

    metrics = build_metrics(w3, events)
    write_metrics_csv(metrics, out_csv)
    summarize_metrics(metrics)


if __name__ == "__main__":
    main()

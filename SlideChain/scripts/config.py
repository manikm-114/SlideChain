from pathlib import Path

# ============================================================
# FILESYSTEM PATHS
# ============================================================

# Source: by-slide merged model outputs (your final integrated data)
BY_SLIDE_ROOT = Path(
    r"F:\Research Files\SlideChain\Lectures\by_slide"
)

# Provenance output (merged per-slide provenance)
PROVENANCE_ROOT = Path(
    r"F:\Research Files\SlideChain\SlideChain\provenance"
)

# Directory for local analysis results
ANALYSIS_RESULTS_DIR = Path(
    r"F:\Research Files\SlideChain\SlideChain\analysis_results"
)

# Directory for blockchain analysis results
CHAIN_RESULTS_DIR = Path(
    r"F:\Research Files\SlideChain\SlideChain\chain_results"
)

ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# BLOCKCHAIN CONFIG
# ============================================================

# Your local Hardhat node
RPC_URL = "http://127.0.0.1:8545"

# Path to ABI JSON (renamed to match the actual file)
ABI_PATH = Path(
    r"F:\Research Files\SlideChain\SlideChain\scripts\SlideChain_abi.json"
)

# Path containing deployed contract address produced by Hardhat
ADDRESS_PATH = Path(
    r"F:\Research Files\SlideChain\SlideChain\scripts\SlideChain_address.txt"
)

# Use the first local Hardhat account
ACCOUNT_INDEX = 0

# Reasonable default gas limit
GAS_LIMIT = 3_000_000

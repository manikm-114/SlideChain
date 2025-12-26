from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ⚠️ Adjust this if your Lectures live elsewhere
PROJECT_ROOT = Path(r"G:\\")

LECTURES_ROOT = PROJECT_ROOT / "Lectures"
PROVENANCE_ROOT = PROJECT_ROOT / "slidechain" / "provenance_records"

DATASET_ID = "MILU23"

MODEL_IDS = [
    "llava-hf__llava-onevision-qwen2-7b-ov-hf",
    "OpenGVLab__InternVL3-14B",
    "Qwen__Qwen2-VL-7B-Instruct",
    "Qwen__Qwen3-VL-4B-Instruct",
]

RPC_URL = os.getenv("SLIDECHAIN_RPC_URL", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv("SLIDECHAIN_PRIVATE_KEY", "")
CONTRACT_ADDRESS = os.getenv("SLIDECHAIN_CONTRACT_ADDRESS", "")

CONTRACT_ABI_PATH = BASE_DIR / "artifacts" / "SlideChain.abi.json"

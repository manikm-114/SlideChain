from pathlib import Path

# === ROOT PATHS ===

# by-slide source directory (raw model outputs)
BY_SLIDE_ROOT = Path(
    r"F:\Research Files\SlideChain\Lectures\by_slide"
)

# provenance directory (computed merged provenance)
PROVENANCE_ROOT = Path(
    r"F:\Research Files\SlideChain\SlideChain\provenance"
)

# output results
ANALYSIS_RESULTS_DIR = Path(
    r"F:\Research Files\SlideChain\SlideChain\analysis_results"
)

CHAIN_RESULTS_DIR = Path(
    r"F:\Research Files\SlideChain\SlideChain\chain_results"
)

# ensure directories exist
ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

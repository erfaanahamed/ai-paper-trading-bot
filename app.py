"""Streamlit entry point for AI Brain V7 paper-trading research app."""
from pathlib import Path
import sys

# Ensure imports work when Streamlit starts from a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACKAGE_DIR = PROJECT_ROOT / "trading_engine"
if not PACKAGE_DIR.is_dir() or not (PACKAGE_DIR / "__init__.py").is_file():
    raise ModuleNotFoundError(
        "The 'trading_engine' package is missing. Upload the CONTENTS of the full ZIP "
        "to the GitHub repository root, preserving the trading_engine/ folder; do not "
        "upload only app.py or upload the ZIP as a single file."
    )

from trading_engine.app import *  # noqa: F401,F403,E402

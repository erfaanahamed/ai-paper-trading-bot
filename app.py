"""Streamlit entry point for AI Brain V7 paper-trading research app.

Supports a normal repository-root layout and one accidentally nested project
folder. If the package is absent, the error explains exactly what to upload.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

def find_package_root(root: Path):
    # Expected layout: root/trading_engine/__init__.py
    expected = root / "trading_engine" / "__init__.py"
    if expected.is_file():
        return root
    # Recovery for projects accidentally uploaded one directory too deep.
    # Do not search hidden folders or virtual environments.
    for init_file in root.glob("*/trading_engine/__init__.py"):
        parent = init_file.parent.parent
        if parent.is_dir():
            return parent
    return None

PACKAGE_ROOT = find_package_root(PROJECT_ROOT)
if PACKAGE_ROOT is None:
    raise ModuleNotFoundError(
        "Cannot find trading_engine/__init__.py. In GitHub, extract the ZIP on your "
        "phone and upload the ZIP CONTENTS (not the ZIP file) to the repository root. "
        "The repository must contain app.py, requirements.txt, and the trading_engine/ "
        "folder with __init__.py. Then commit the changes and reboot Streamlit."
    )
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from trading_engine.app import *  # noqa: F401,F403,E402

"""
Streamlit entrypoint for Streamlit Community Cloud deployment.
Redirects to the main UI application at src/interfaces/ui/app.py.
"""

import sys
import runpy
from pathlib import Path

# Ensure repo root is on Python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

app_path = repo_root / "src" / "interfaces" / "ui" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")

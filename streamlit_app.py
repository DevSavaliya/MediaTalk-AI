"""Community Cloud entry point with CPU transcription and visitor isolation."""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["WHISPER_ENGINE"] = "faster"
runpy.run_path(str(ROOT / "streamlit_cloud.py"), run_name="__main__")

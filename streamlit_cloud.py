"""Public-demo entry point: always enable isolated visitor sessions."""

import os
import runpy
from pathlib import Path

os.environ["MEDIATALK_PUBLIC_DEMO"] = "1"
runpy.run_path(str(Path(__file__).resolve().with_name("streamlit_app.py")), run_name="__main__")

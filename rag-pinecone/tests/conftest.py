"""
Pytest configuration — auto-loads .env before test collection.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))

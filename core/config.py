import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file in the project root
load_dotenv()

# --- API Configuration ---
EODHD_API_KEY: Optional[str] = os.getenv("EODHD_API_KEY")

# --- Directory Configuration ---
DATA_DIR: str = "data/"
RAW_DIR: str = "data/raw/"
PROCESSED_DIR: str = "data/processed/"

# --- Initialization ---
def _init_directories() -> None:
    """Create necessary data directories if they do not exist."""
    directories = [DATA_DIR, RAW_DIR, PROCESSED_DIR]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

_init_directories()

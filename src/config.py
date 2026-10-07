from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# HR and IT document directories
HR_DATA_DIR = RAW_DATA_DIR / "hr"
IT_DATA_DIR = RAW_DATA_DIR / "it"
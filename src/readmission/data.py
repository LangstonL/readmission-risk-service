"""Download and load the UCI Diabetes 130-US Hospitals dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
DEFAULT_DATA_DIR = Path("data/raw")


def download(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the dataset zip and extract the patient-level CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "diabetic_data.csv"
    if csv_path.exists():
        return csv_path

    with urllib.request.urlopen(DATASET_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    archive.extract("diabetic_data.csv", data_dir)
    return csv_path


def load(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the patient-level encounters, treating '?' as missing."""
    return pd.read_csv(download(data_dir), na_values=["?"], low_memory=False)

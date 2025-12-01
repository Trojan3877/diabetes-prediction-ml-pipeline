import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Save processed CSV dataset."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
"""Root-level preprocessing module (delegates to src.data_preprocessing)."""

from src.data_preprocessing import load_data, preprocess_data, train_test_split_data

__all__ = ["load_data", "preprocess_data", "train_test_split_data"]

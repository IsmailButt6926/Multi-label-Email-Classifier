# Any extra functionality that needs to be reused will go here
import pandas as pd
import numpy as np


def remove_low_frequency_classes(df, X, target_col, min_count):
    """Remove rows whose target class has fewer than min_count instances.

    Args:
        df: DataFrame containing the target column.
        X: Feature matrix (numpy array), aligned row-wise with df.
        target_col: Name of the target column to check class frequencies.
        min_count: Minimum number of instances a class must have.

    Returns:
        Tuple of (filtered X, filtered df) with low-frequency classes removed.
    """
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    mask = df[target_col].isin(valid_classes)
    return X[mask.values], df[mask].reset_index(drop=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from utils import remove_low_frequency_classes
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():
    """Encapsulates training and testing data for a classification target.

    This class handles:
    - Dropping rows where the target column is NaN
    - Removing classes with too few instances
    - Performing stratified train/test split
    """

    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None) -> None:
        if target_col is None:
            target_col = Config.CLASS_COL

        # Drop rows where target is NaN
        valid_mask = df[target_col].notna()
        X_valid = X[valid_mask.values]
        df_valid = df[valid_mask].reset_index(drop=True)

        # Remove classes with too few instances
        X_valid, df_valid = remove_low_frequency_classes(
            df_valid, X_valid, target_col, Config.MIN_CLASS_COUNT
        )

        self.embeddings = X_valid
        self.y = df_valid[target_col].values
        self.target_col = target_col

        # Stratified train/test split using indices
        indices = np.arange(len(df_valid))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2, random_state=Config.SEED, stratify=self.y
        )

        self.X_train = X_valid[train_idx]
        self.X_test = X_valid[test_idx]
        self.y_train = self.y[train_idx]
        self.y_test = self.y[test_idx]

        self.train_df = df_valid.iloc[train_idx].reset_index(drop=True)
        self.test_df = df_valid.iloc[test_idx].reset_index(drop=True)

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df


class FilteredData():
    """Data object created from pre-filtered subsets for hierarchical modelling.

    Unlike Data, this class receives pre-split X_train/X_test arrays directly,
    used when the parent model's class determines which subset to use.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.embeddings = np.vstack([X_train, X_test]) if len(X_train) > 0 and len(X_test) > 0 else X_train
        self.y = np.concatenate([y_train, y_test]) if len(y_train) > 0 and len(y_test) > 0 else y_train

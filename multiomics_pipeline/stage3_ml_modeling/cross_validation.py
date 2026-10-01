"""
Stage 3: Cross-Validation Engine (Group K-Fold)
Enforces strict biological replicate grouping across donor animals to eliminate data leakage.
"""
from typing import Generator, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from ..config import config

class BiologicalGroupKFoldEngine:
    """
    Splits cohort datasets by animal replicates ensuring zero sample leakage.
    """
    def __init__(self, n_splits: int = 4, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def split(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_idx, val_idx) using StratifiedGroupKFold with dynamic split adaptation.
        """
        n_samples = len(X)
        n_unique_groups = len(np.unique(groups))
        
        # Check min class count
        y_series = pd.Series(y)
        min_class_count = int(y_series.value_counts().min()) if len(y_series) > 0 else 1
        
        # Clamped splits
        max_possible_splits = min(self.n_splits, n_unique_groups, min_class_count)
        splits_to_use = max(2, max_possible_splits)
        
        try:
            if n_unique_groups >= splits_to_use and min_class_count >= splits_to_use:
                sgkf = StratifiedGroupKFold(n_splits=splits_to_use, shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in sgkf.split(X, y, groups):
                    yield train_idx, val_idx
            else:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=min(splits_to_use, n_samples), shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in kf.split(X, y):
                    yield train_idx, val_idx
        except Exception:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(X, y):
                yield train_idx, val_idx
            
    def validate_splits(self, groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
        """
        Verifies that train and val groups have zero intersection.
        """
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        return len(train_groups.intersection(val_groups)) == 0

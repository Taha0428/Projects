"""
Stage 3: Multi-Omics Machine Learning Model Zoo
Implements XGBoost, Random Forest, and Multi-Layer Perceptron (MLP) architectures.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
from ..config import config

class MultiOmicsModelZoo:
    """
    Factory and training suite for XGBoost, Random Forest, and MLP models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def get_classifiers(self) -> Dict[str, Any]:
        return {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                eval_metric="mlogloss",
                n_jobs=-1
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "MLP_Neural_Net": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=350,
                early_stopping=False,
                random_state=self.random_state
            )
        }
        
    def get_regressors(self) -> Dict[str, Any]:
        return {
            "XGBoost_Regressor": xgb.XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Random_Forest_Regressor": RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "MLP_Regressor": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=350,
                early_stopping=False,
                random_state=self.random_state
            )
        }
